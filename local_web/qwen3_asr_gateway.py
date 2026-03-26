"""Unified Qwen3-ASR gateway: one model process, multiple HTTP capabilities.

This service loads a single vLLM-backed Qwen3-ASR model and exposes both:

- offline transcription for uploaded audio
- live streaming transcription with committed-history subtitles

The point is to avoid the earlier "one container per capability" setup, which
duplicates model memory and GPU KV cache. Frontends talk to one base URL and
discover capabilities from ``/api/info`` instead of hard-coding multiple ports.
"""

from __future__ import annotations

import argparse
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from flask import Flask, Response, jsonify, request
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output


@dataclass
class Session:
    """Server-side state for one streaming browser session."""

    state: object
    created_at: float
    last_seen: float


app = Flask(__name__)

global asr
SESSIONS: Dict[str, Session] = {}
SESSION_TTL_SEC = 10 * 60


def _json_error(message: str, status: int = 400):
    """Return a compact JSON error payload."""
    return jsonify({"error": message}), status


@app.after_request
def _add_cors_headers(resp):
    """Allow local frontends to call this gateway from a different origin."""
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


def _gc_sessions() -> None:
    """Expire inactive sessions and flush any buffered tail audio."""
    now = time.time()
    dead = [sid for sid, sess in SESSIONS.items() if now - sess.last_seen > SESSION_TTL_SEC]
    for sid in dead:
        try:
            asr.finish_streaming_transcribe(SESSIONS[sid].state)
        except Exception:
            pass
        SESSIONS.pop(sid, None)


def _get_session(session_id: str) -> Optional[Session]:
    """Fetch a session by id and refresh its liveness timestamp."""
    _gc_sessions()
    sess = SESSIONS.get(session_id)
    if sess:
        sess.last_seen = time.time()
    return sess


def _longest_common_prefix_len(a: str, b: str) -> int:
    """Return the number of leading characters shared by both strings."""
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def _decode_clean_prefix(tokenizer, token_ids, end_idx: int) -> str:
    """Decode a token prefix while avoiding broken Unicode boundaries."""
    end_idx = max(0, int(end_idx))
    while end_idx > 0:
        prefix = tokenizer.decode(token_ids[:end_idx])
        if "\ufffd" not in prefix:
            return prefix
        end_idx -= 1
    return ""


def _split_committed_and_live_text(state) -> Tuple[str, str, str]:
    """Split the current streaming hypothesis into committed and live regions."""
    full_language = getattr(state, "language", "") or ""
    full_text = getattr(state, "text", "") or ""
    full_raw = getattr(state, "_raw_decoded", "") or ""

    if not full_text:
        return full_language, "", ""

    if not full_raw or state.chunk_id < state.unfixed_chunk_num:
        return full_language, "", full_text

    tokenizer = asr.processor.tokenizer
    token_ids = tokenizer.encode(full_raw)
    stable_end_idx = max(0, len(token_ids) - int(state.unfixed_token_num))
    stable_raw = _decode_clean_prefix(tokenizer, token_ids, stable_end_idx)

    if not stable_raw:
        return full_language, "", full_text

    stable_language, stable_text = parse_asr_output(stable_raw, user_language=state.force_language)
    language = full_language or stable_language

    if not stable_text:
        return language, "", full_text

    if full_text.startswith(stable_text):
        return language, stable_text, full_text[len(stable_text) :]

    lcp = _longest_common_prefix_len(stable_text, full_text)
    return language, full_text[:lcp], full_text[lcp:]


def _stream_payload(state, final: bool = False) -> dict:
    """Build the JSON payload returned for streaming requests."""
    language = getattr(state, "language", "") or ""
    text = getattr(state, "text", "") or ""

    if final:
        committed_text, live_text = text, ""
    else:
        language, committed_text, live_text = _split_committed_and_live_text(state)

    return {
        "language": language,
        "text": text,
        "committed_text": committed_text,
        "live_text": live_text,
        "chunk_id": int(getattr(state, "chunk_id", 0)),
        "unfixed_chunk_num": int(getattr(state, "unfixed_chunk_num", 0)),
        "unfixed_token_num": int(getattr(state, "unfixed_token_num", 0)),
    }


@app.get("/")
def index():
    """Serve the unified local web UI."""
    html_path = Path(__file__).with_name("qwen3_asr_gateway.html")
    return Response(html_path.read_text(encoding="utf-8"), mimetype="text/html; charset=utf-8")


@app.get("/healthz")
def healthz():
    """Cheap liveness probe."""
    return jsonify({"ok": True})


@app.get("/api/info")
def api_info():
    """Describe the capabilities exposed by this gateway."""
    return jsonify(
        {
            "service": "qwen3-asr-gateway",
            "model": app.config["MODEL_NAME"],
            "backend": "vllm",
            "capabilities": {
                "transcribe": True,
                "streaming": True,
                "timestamps": False,
                "committed_live_split": True,
            },
            "routes": {
                "ui": "/",
                "healthz": "/healthz",
                "info": "/api/info",
                "transcribe": "/api/transcribe",
                "stream_start": "/api/stream/start",
                "stream_chunk": "/api/stream/chunk",
                "stream_finish": "/api/stream/finish",
            },
        }
    )


@app.route("/api/transcribe", methods=["POST", "OPTIONS"])
def api_transcribe():
    """Run one-shot transcription on a data-url audio payload."""
    if request.method == "OPTIONS":
        return Response(status=204)

    payload = request.get_json(silent=True) or {}
    audio_data_url = str(payload.get("audio_data_url") or payload.get("audio_url") or "").strip()
    context = str(payload.get("context") or "")
    language = str(payload.get("language") or "").strip() or None

    if not audio_data_url:
        return _json_error("audio_data_url is required")

    start = time.time()
    try:
        result = asr.transcribe(
            audio=audio_data_url,
            context=context,
            language=language,
            return_time_stamps=False,
        )[0]
    except Exception as exc:
        return _json_error(str(exc), status=500)

    return jsonify(
        {
            "mode": "transcribe",
            "language": result.language or "",
            "text": result.text or "",
            "request_ms": int((time.time() - start) * 1000),
        }
    )


@app.route("/api/stream/start", methods=["POST", "OPTIONS"])
def api_stream_start():
    """Create a fresh decoder state for one browser streaming session."""
    if request.method == "OPTIONS":
        return Response(status=204)

    payload = request.get_json(silent=True) or {}
    context = str(payload.get("context") or "")
    language = str(payload.get("language") or "").strip() or None

    session_id = uuid.uuid4().hex
    state = asr.init_streaming_state(
        context=context,
        language=language,
        unfixed_chunk_num=app.config["UNFIXED_CHUNK_NUM"],
        unfixed_token_num=app.config["UNFIXED_TOKEN_NUM"],
        chunk_size_sec=app.config["CHUNK_SIZE_SEC"],
    )
    now = time.time()
    SESSIONS[session_id] = Session(state=state, created_at=now, last_seen=now)
    out = _stream_payload(state)
    out["session_id"] = session_id
    return jsonify(out)


@app.route("/api/stream/chunk", methods=["POST", "OPTIONS"])
def api_stream_chunk():
    """Accept one float32 PCM chunk and return the updated streaming hypothesis."""
    if request.method == "OPTIONS":
        return Response(status=204)

    session_id = request.args.get("session_id", "")
    sess = _get_session(session_id)
    if not sess:
        return _json_error("invalid session_id")

    if request.mimetype != "application/octet-stream":
        return _json_error("expect application/octet-stream")

    raw = request.get_data(cache=False)
    if len(raw) % 4 != 0:
        return _json_error("float32 bytes length not multiple of 4")

    wav = np.frombuffer(raw, dtype=np.float32).reshape(-1)
    try:
        asr.streaming_transcribe(wav, sess.state)
    except Exception as exc:
        return _json_error(str(exc), status=500)

    return jsonify(_stream_payload(sess.state))


@app.route("/api/stream/finish", methods=["POST", "OPTIONS"])
def api_stream_finish():
    """Flush any buffered tail audio and return the final streaming result."""
    if request.method == "OPTIONS":
        return Response(status=204)

    session_id = request.args.get("session_id", "")
    sess = _get_session(session_id)
    if not sess:
        return _json_error("invalid session_id")

    try:
        asr.finish_streaming_transcribe(sess.state)
    except Exception as exc:
        return _json_error(str(exc), status=500)

    out = _stream_payload(sess.state, final=True)
    SESSIONS.pop(session_id, None)
    return jsonify(out)


def parse_args():
    """Parse server and vLLM runtime arguments."""
    parser = argparse.ArgumentParser("Qwen3-ASR Unified Gateway")
    parser.add_argument("--asr-model-path", default="Qwen/Qwen3-ASR-0.6B", help="Model name or local path")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--unfixed-chunk-num", type=int, default=4)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument("--chunk-size-sec", type=float, default=1.0)
    return parser.parse_args()


def main():
    """Load the model once and start the gateway."""
    args = parse_args()

    app.config["MODEL_NAME"] = args.asr_model_path
    app.config["UNFIXED_CHUNK_NUM"] = args.unfixed_chunk_num
    app.config["UNFIXED_TOKEN_NUM"] = args.unfixed_token_num
    app.config["CHUNK_SIZE_SEC"] = args.chunk_size_sec

    llm_kwargs = dict(
        model=args.asr_model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
    )
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True

    global asr
    asr = Qwen3ASRModel.LLM(**llm_kwargs)

    print(
        "[gateway] model loaded "
        f"(gpu_memory_utilization={args.gpu_memory_utilization}, "
        f"max_model_len={args.max_model_len}, enforce_eager={args.enforce_eager})"
    )
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
