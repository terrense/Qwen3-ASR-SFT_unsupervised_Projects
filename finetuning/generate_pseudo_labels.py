"""Generate pseudo labels for Qwen3-ASR semi-supervised training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from asr_data_utils import (
    batched,
    collect_audio_files,
    ensure_prompt,
    format_asr_target,
    read_manifest_rows,
    resolve_audio_path,
    write_jsonl,
)
def parse_args():
    p = argparse.ArgumentParser("Generate pseudo labels for Qwen3-ASR")
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--input_manifest", default="", help="Optional manifest with an audio column")
    p.add_argument("--input_format", default="auto", choices=["auto", "csv", "tsv", "json", "jsonl"])
    p.add_argument("--audio_key", default="audio")
    p.add_argument("--prompt_key", default="prompt")
    p.add_argument("--audio_root", default="")
    p.add_argument("--audio_dir", default="", help="Alternative to input_manifest")
    p.add_argument("--recursive", type=int, default=1)
    p.add_argument("--backend", default="transformers", choices=["transformers", "vllm"])
    p.add_argument("--backend_kwargs_json", default="{}", help="Extra kwargs for backend init as JSON dict")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--decode_language", default="Chinese", help="Force decoding language, empty to disable")
    p.add_argument("--train_language", default="Chinese", help="Prefix language for output JSONL, use auto to reuse detected language")
    p.add_argument("--default_prompt", default="")
    p.add_argument("--loss_weight", type=float, default=0.3, help="Stored into output rows for later mixed training")
    p.add_argument("--skip_empty", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=0)
    return p.parse_args()


def choose_dtype():
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    return torch.float16


def build_input_rows(args) -> List[Dict[str, object]]:
    if bool(args.input_manifest) == bool(args.audio_dir):
        raise ValueError("Set exactly one of --input_manifest or --audio_dir")

    rows: List[Dict[str, object]] = []
    if args.input_manifest:
        manifest_rows = read_manifest_rows(args.input_manifest, fmt=args.input_format)
        for row in manifest_rows:
            if args.audio_key not in row:
                raise KeyError(f"Missing audio column: {args.audio_key}")
            audio = resolve_audio_path(
                str(row[args.audio_key]),
                manifest_path=args.input_manifest,
                audio_root=args.audio_root,
            )
            rows.append(
                {
                    "audio": audio,
                    "prompt": ensure_prompt(row.get(args.prompt_key), args.default_prompt),
                }
            )
    else:
        for audio in collect_audio_files(args.audio_dir, recursive=(args.recursive == 1)):
            rows.append({"audio": audio, "prompt": args.default_prompt})

    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    return rows


def load_model(args):
    from qwen_asr import Qwen3ASRModel

    backend_kwargs = json.loads(args.backend_kwargs_json)
    if args.backend == "transformers":
        return Qwen3ASRModel.from_pretrained(
            args.model_path,
            dtype=choose_dtype(),
            device_map=backend_kwargs.pop("device_map", "auto"),
            **backend_kwargs,
        )
    return Qwen3ASRModel.LLM(model=args.model_path, **backend_kwargs)


def choose_target_language(args, detected_language: str) -> str:
    mode = str(args.train_language).strip()
    if not mode:
        return "None"
    if mode.lower() == "auto":
        return detected_language or "None"
    return mode


def main():
    args = parse_args()
    rows = build_input_rows(args)
    asr = load_model(args)
    decode_language = args.decode_language.strip() or None
    outputs: List[Dict[str, object]] = []

    for batch in batched(rows, args.batch_size):
        audios = [item["audio"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        results = asr.transcribe(audio=audios, context=prompts, language=decode_language)

        for item, result in zip(batch, results):
            text = (result.text or "").strip()
            if args.skip_empty == 1 and not text:
                continue
            detected_language = (result.language or "").strip()
            target_language = choose_target_language(args, detected_language)
            out = {
                "audio": item["audio"],
                "text": format_asr_target(text, language=target_language),
                "loss_weight": float(args.loss_weight),
                "pseudo": True,
                "detected_language": detected_language,
                "raw_text": text,
            }
            if item["prompt"]:
                out["prompt"] = item["prompt"]
            outputs.append(out)

    count = write_jsonl(args.output_file, outputs)
    print(f"[pseudo] input rows:  {len(rows)}")
    print(f"[pseudo] output rows: {count}")
    print(f"[pseudo] wrote:       {args.output_file}")


if __name__ == "__main__":
    main()
