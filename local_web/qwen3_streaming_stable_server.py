"""Streaming web demo with committed-history subtitles for Qwen3-ASR.

The upstream streaming demo intentionally shows the model's latest full
hypothesis. Because the decoder rolls back a few trailing tokens and re-decodes
them on each chunk, previously displayed text can change or disappear.

For subtitle-like UX that behavior feels jarring. This server keeps the same
streaming backend, but splits the current hypothesis into:

- committed_text: a stable prefix that is unlikely to change again
- live_text: the unstable tail that may still be revised

The split is derived from the same rollback heuristic used by the model:
everything except the last ``unfixed_token_num`` decoded tokens is treated as
committed once ``chunk_id >= unfixed_chunk_num``.
"""

from __future__ import annotations

import argparse
import time
import uuid
from dataclasses import dataclass
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
    """Decode a token prefix while avoiding a broken Unicode replacement char."""
    end_idx = max(0, int(end_idx))
    while end_idx > 0:
        prefix = tokenizer.decode(token_ids[:end_idx])
        if "\ufffd" not in prefix:
            return prefix
        end_idx -= 1
    return ""


def _split_committed_and_live_text(state) -> Tuple[str, str, str]:
    """Split the current hypothesis into committed and live regions.

    Returns:
        Tuple[str, str, str]:
            ``(language, committed_text, live_text)``
    """
    full_language = getattr(state, "language", "") or ""
    full_text = getattr(state, "text", "") or ""
    full_raw = getattr(state, "_raw_decoded", "") or ""

    if not full_text:
        return full_language, "", ""

    # During the earliest chunks the model intentionally does not reuse a text
    # prefix yet, so we keep the whole hypothesis in the live area.
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

    # Fallback for parser normalization edge cases: keep only the common prefix
    # as committed and leave the rest in the live tail.
    lcp = _longest_common_prefix_len(stable_text, full_text)
    return language, full_text[:lcp], full_text[lcp:]


def _session_payload(state, final: bool = False) -> dict:
    """Build the JSON payload returned to the browser."""
    language = getattr(state, "language", "") or ""
    text = getattr(state, "text", "") or ""
    committed_text, live_text = "", text

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


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Qwen3-ASR Streaming Subtitles</title>
  <style>
    :root{
      --bg:#f5f7fb;
      --card:#ffffff;
      --text:#0f172a;
      --muted:#526072;
      --border:#dbe3ef;
      --ok:#0f8a5f;
      --warn:#b97708;
      --err:#c42b58;
      --live-bg:#eef4ff;
      --live-text:#1d4ed8;
      --stable-bg:#f8fafc;
    }

    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body{
      margin:0;
      color:var(--text);
      background:
        radial-gradient(circle at top right, rgba(29,78,216,.08), transparent 28%),
        linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    }

    .wrap{
      min-height: 100%;
      padding: 20px;
      display: flex;
    }

    .card{
      width: 100%;
      min-height: calc(100vh - 40px);
      display: flex;
      flex-direction: column;
      gap: 14px;
      padding: 20px;
      border: 1px solid var(--border);
      border-radius: 20px;
      background: rgba(255,255,255,.92);
      box-shadow: 0 24px 80px rgba(15,23,42,.08);
      backdrop-filter: blur(10px);
    }

    .topbar{
      display:flex;
      gap:12px;
      align-items:center;
      flex-wrap:wrap;
    }

    .titlebox{
      display:flex;
      flex-direction:column;
      gap:4px;
      margin-right:auto;
    }

    h1{
      margin:0;
      font-size:20px;
      letter-spacing:.2px;
    }

    .subtitle{
      color:var(--muted);
      font-size:13px;
      line-height:1.5;
    }

    button{
      border:1px solid var(--border);
      border-radius: 999px;
      padding: 11px 16px;
      background: #fff;
      color:var(--text);
      font-weight:700;
      cursor:pointer;
      transition: background .15s ease, transform .05s ease, border-color .15s ease;
    }

    button:hover{
      background:#f8fbff;
      border-color:#bfd0e8;
    }

    button:active{
      transform: translateY(1px);
    }

    button.primary{
      background: rgba(15,138,95,.10);
      border-color: rgba(15,138,95,.28);
      color:#0b6a49;
    }

    button.danger{
      background: rgba(196,43,88,.10);
      border-color: rgba(196,43,88,.22);
      color:#9f1f49;
    }

    button:disabled{
      opacity:.55;
      cursor:not-allowed;
    }

    .pill{
      border:1px solid var(--border);
      border-radius: 999px;
      padding: 7px 11px;
      font-size:12px;
      color:var(--muted);
      background:#fff;
      user-select:none;
    }

    .pill.ok{
      color:#0b6a49;
      background: rgba(15,138,95,.10);
      border-color: rgba(15,138,95,.22);
    }

    .pill.warn{
      color:#9a6706;
      background: rgba(185,119,8,.10);
      border-color: rgba(185,119,8,.20);
    }

    .pill.err{
      color:#9f1f49;
      background: rgba(196,43,88,.10);
      border-color: rgba(196,43,88,.20);
    }

    .meta{
      display:grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap:12px;
    }

    .panel{
      border:1px solid var(--border);
      border-radius: 16px;
      background: #fff;
      padding: 14px 16px;
    }

    .label{
      color:var(--muted);
      font-size:12px;
      text-transform: uppercase;
      letter-spacing:.08em;
      margin-bottom:8px;
    }

    .mono{
      font-family: Consolas, "SFMono-Regular", Menlo, monospace;
    }

    .subtitle-area{
      flex:1;
      min-height:0;
      display:grid;
      grid-template-rows: minmax(0, 1fr) auto;
      gap:12px;
    }

    .subtitle-box{
      min-height:0;
      display:flex;
      flex-direction:column;
      overflow:hidden;
    }

    .subtitle-content{
      flex:1;
      min-height:0;
      overflow:auto;
      white-space:pre-wrap;
      font-size:18px;
      line-height:1.8;
      border-radius: 14px;
      padding: 16px;
    }

    #committed{
      background:var(--stable-bg);
      border:1px solid var(--border);
    }

    #live{
      min-height:88px;
      background:var(--live-bg);
      border:1px dashed rgba(29,78,216,.28);
      color:var(--live-text);
      font-weight:600;
    }

    .hint{
      color:var(--muted);
      font-size:13px;
      line-height:1.6;
    }

    @media (max-width: 800px){
      .wrap{ padding: 12px; }
      .card{
        min-height: calc(100vh - 24px);
        padding: 16px;
      }
      .subtitle-content{ font-size:16px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="topbar">
        <div class="titlebox">
          <h1>Qwen3-ASR Streaming Subtitles</h1>
          <div class="subtitle">
            Stable history stays in the main subtitle area. The blue live tail may still be revised by the model.
          </div>
        </div>
        <button id="btnStart" class="primary">Start Mic</button>
        <button id="btnStop" class="danger" disabled>Stop</button>
        <button id="btnClear">Clear</button>
        <span id="status" class="pill warn">Idle</span>
      </div>

      <div class="meta">
        <div class="panel">
          <div class="label">Language</div>
          <div id="lang" class="mono">-</div>
        </div>
        <div class="panel">
          <div class="label">Streaming Rule</div>
          <div id="rule" class="mono">Commit after rollback window</div>
        </div>
      </div>

      <div class="subtitle-area">
        <div class="subtitle-box">
          <div class="label">Committed Subtitle History</div>
          <div id="committed" class="subtitle-content"></div>
        </div>
        <div class="subtitle-box">
          <div class="label">Live Tail (may still change)</div>
          <div id="live" class="subtitle-content"></div>
        </div>
      </div>

      <div class="hint">
        Why old text used to disappear: Qwen3-ASR streaming re-decodes the trailing part of the transcript on every chunk.
        This page keeps the stable prefix visible and only lets the tail refresh.
      </div>
    </div>
  </div>

<script>
(() => {
  const $ = (id) => document.getElementById(id);

  const btnStart = $("btnStart");
  const btnStop = $("btnStop");
  const btnClear = $("btnClear");
  const statusEl = $("status");
  const langEl = $("lang");
  const ruleEl = $("rule");
  const committedEl = $("committed");
  const liveEl = $("live");

  const CHUNK_MS = 500;
  const TARGET_SR = 16000;

  let audioCtx = null;
  let processor = null;
  let source = null;
  let mediaStream = null;

  let sessionId = null;
  let running = false;
  let pushing = false;
  let buf = new Float32Array(0);

  function setStatus(text, cls){
    statusEl.textContent = text;
    statusEl.className = "pill " + (cls || "");
  }

  function setRunningUI(isRunning){
    btnStart.disabled = isRunning;
    btnStop.disabled = !isRunning;
  }

  function resetText(){
    committedEl.textContent = "";
    liveEl.textContent = "";
    langEl.textContent = "-";
  }

  function concatFloat32(a, b){
    const out = new Float32Array(a.length + b.length);
    out.set(a, 0);
    out.set(b, a.length);
    return out;
  }

  function resampleLinear(input, srcSr, dstSr){
    if (srcSr === dstSr) return input;
    const ratio = dstSr / srcSr;
    const outLen = Math.max(0, Math.round(input.length * ratio));
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++){
      const x = i / ratio;
      const x0 = Math.floor(x);
      const x1 = Math.min(x0 + 1, input.length - 1);
      const t = x - x0;
      out[i] = input[x0] * (1 - t) + input[x1] * t;
    }
    return out;
  }

  function scrollToBottom(el){
    el.scrollTop = el.scrollHeight;
  }

  function renderPayload(payload){
    langEl.textContent = payload.language || "-";
    committedEl.textContent = payload.committed_text || "";
    liveEl.textContent = payload.live_text || "";
    ruleEl.textContent =
      "unfixed_chunk_num=" + payload.unfixed_chunk_num +
      ", unfixed_token_num=" + payload.unfixed_token_num;
    scrollToBottom(committedEl);
    scrollToBottom(liveEl);
  }

  async function apiStart(){
    const resp = await fetch("/api/start", { method: "POST" });
    if (!resp.ok) throw new Error(await resp.text());
    const payload = await resp.json();
    sessionId = payload.session_id;
    renderPayload(payload);
  }

  async function apiPushChunk(float32_16k){
    const resp = await fetch("/api/chunk?session_id=" + encodeURIComponent(sessionId), {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: float32_16k.buffer,
    });
    if (!resp.ok) throw new Error(await resp.text());
    return await resp.json();
  }

  async function apiFinish(){
    const resp = await fetch("/api/finish?session_id=" + encodeURIComponent(sessionId), {
      method: "POST",
    });
    if (!resp.ok) throw new Error(await resp.text());
    return await resp.json();
  }

  async function stopAudioPipeline(){
    try{
      if (processor){
        processor.disconnect();
        processor.onaudioprocess = null;
      }
      if (source) source.disconnect();
      if (audioCtx) await audioCtx.close();
      if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
    }catch(e){}

    processor = null;
    source = null;
    audioCtx = null;
    mediaStream = null;
  }

  btnClear.onclick = () => {
    resetText();
    setStatus(running ? "Listening" : "Idle", running ? "ok" : "warn");
  };

  btnStart.onclick = async () => {
    if (running) return;

    resetText();
    buf = new Float32Array(0);

    try{
      setStatus("Starting", "warn");
      setRunningUI(true);

      await apiStart();

      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });

      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      source = audioCtx.createMediaStreamSource(mediaStream);
      processor = audioCtx.createScriptProcessor(4096, 1, 1);

      processor.onaudioprocess = (e) => {
        if (!running) return;
        const input = e.inputBuffer.getChannelData(0);
        const resampled = resampleLinear(input, audioCtx.sampleRate, TARGET_SR);
        buf = concatFloat32(buf, resampled);
        if (!pushing) pump();
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      running = true;
      setStatus("Listening", "ok");
    }catch(err){
      console.error(err);
      setStatus("Start failed: " + err.message, "err");
      setRunningUI(false);
      running = false;
      sessionId = null;
      await stopAudioPipeline();
    }
  };

  async function pump(){
    if (pushing) return;
    pushing = true;
    const chunkSamples = Math.round(TARGET_SR * (CHUNK_MS / 1000));

    try{
      while (running && buf.length >= chunkSamples){
        const chunk = buf.slice(0, chunkSamples);
        buf = buf.slice(chunkSamples);
        const payload = await apiPushChunk(chunk);
        renderPayload(payload);
        if (running) setStatus("Listening", "ok");
      }
    }catch(err){
      console.error(err);
      if (running) setStatus("Backend error: " + err.message, "err");
    }finally{
      pushing = false;
    }
  }

  btnStop.onclick = async () => {
    if (!running) return;

    running = false;
    setStatus("Finishing", "warn");
    setRunningUI(false);

    await stopAudioPipeline();

    try{
      if (sessionId){
        const payload = await apiFinish();
        renderPayload(payload);
      }
      setStatus("Stopped", "");
    }catch(err){
      console.error(err);
      setStatus("Finish failed: " + err.message, "err");
    }finally{
      sessionId = null;
      buf = new Float32Array(0);
      pushing = false;
    }
  };
})();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    """Serve the subtitle-oriented streaming page."""
    return Response(INDEX_HTML, mimetype="text/html; charset=utf-8")


@app.post("/api/start")
def api_start():
    """Create a fresh decoder state for one browser session."""
    session_id = uuid.uuid4().hex
    state = asr.init_streaming_state(
        unfixed_chunk_num=app.config["UNFIXED_CHUNK_NUM"],
        unfixed_token_num=app.config["UNFIXED_TOKEN_NUM"],
        chunk_size_sec=app.config["CHUNK_SIZE_SEC"],
    )
    now = time.time()
    SESSIONS[session_id] = Session(state=state, created_at=now, last_seen=now)
    payload = _session_payload(state)
    payload["session_id"] = session_id
    return jsonify(payload)


@app.post("/api/chunk")
def api_chunk():
    """Accept one float32 chunk and return the updated streaming hypothesis."""
    session_id = request.args.get("session_id", "")
    sess = _get_session(session_id)
    if not sess:
        return jsonify({"error": "invalid session_id"}), 400

    if request.mimetype != "application/octet-stream":
        return jsonify({"error": "expect application/octet-stream"}), 400

    raw = request.get_data(cache=False)
    if len(raw) % 4 != 0:
        return jsonify({"error": "float32 bytes length not multiple of 4"}), 400

    wav = np.frombuffer(raw, dtype=np.float32).reshape(-1)
    asr.streaming_transcribe(wav, sess.state)
    return jsonify(_session_payload(sess.state))


@app.post("/api/finish")
def api_finish():
    """Flush any buffered tail audio and return the final subtitle text."""
    session_id = request.args.get("session_id", "")
    sess = _get_session(session_id)
    if not sess:
        return jsonify({"error": "invalid session_id"}), 400

    asr.finish_streaming_transcribe(sess.state)
    payload = _session_payload(sess.state, final=True)
    SESSIONS.pop(session_id, None)
    return jsonify(payload)


def parse_args():
    """Parse server and vLLM runtime arguments."""
    parser = argparse.ArgumentParser("Qwen3-ASR Streaming Subtitles Server")
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
    """Load the vLLM backend once and start the Flask app."""
    args = parse_args()

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
        "[streaming-subtitles] model loaded "
        f"(gpu_memory_utilization={args.gpu_memory_utilization}, "
        f"max_model_len={args.max_model_len}, enforce_eager={args.enforce_eager})"
    )
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
