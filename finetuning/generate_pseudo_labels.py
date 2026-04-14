"""Generate pseudo labels for Qwen3-ASR semi-supervised training.

Conceptually this script turns unlabeled speech into weakly supervised training
data:

1. Gather audio either from a manifest or directly from a directory.
2. Run the released Qwen3-ASR checkpoint as a teacher model.
3. Store the decoded text back into the same JSONL schema used by supervised
   SFT, with an extra ``loss_weight`` that later training can down-weight.

This is not "true" self-supervised learning in the HuBERT/wav2vec sense. It is
teacher-generated pseudo labeling, which is often the most practical way to
exploit large untranscribed dialect corpora in an instruction-decoder ASR setup
like Qwen3-ASR.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import torch

# Adding the repository root to ``sys.path`` lets the script run directly from a
# source checkout without requiring the package to be installed into the active
# environment first.
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
    """Define CLI options for pseudo-label generation."""
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
    """Pick a reasonable inference dtype from the available CUDA capability."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    # FP16 is the fallback for older CUDA devices. CPU-only use is not the main
    # target for this script, but returning a torch dtype here keeps the calling
    # code simple.
    return torch.float16


def build_input_rows(args) -> List[Dict[str, object]]:
    """Collect audio inputs from either a manifest or a directory tree."""
    if bool(args.input_manifest) == bool(args.audio_dir):
        raise ValueError("Set exactly one of --input_manifest or --audio_dir")

    rows: List[Dict[str, object]] = []
    if args.input_manifest:
        # Manifest mode is useful when unlabeled audio still carries metadata such
        # as prompts or custom path columns.
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
                    # Preserving prompt here matters because pseudo labeling
                    # should ideally happen under the same conditioning context
                    # that later training rows will use.
                    "prompt": ensure_prompt(row.get(args.prompt_key), args.default_prompt),
                }
            )
    else:
        # Directory mode is the simplest path when you only have a folder tree of
        # unlabeled waveforms and no external manifest.
        for audio in collect_audio_files(args.audio_dir, recursive=(args.recursive == 1)):
            rows.append({"audio": audio, "prompt": args.default_prompt})

    if args.max_samples > 0:
        # A quick cap is handy when validating pipeline correctness before
        # generating pseudo labels for a large corpus.
        rows = rows[: args.max_samples]
    return rows


def load_model(args):
    """Load the teacher ASR model using either the Transformers or vLLM backend.

    Python note:
        The import is intentionally local so ``--help`` can work even in
        environments where model dependencies are not fully installed yet.
    """
    from qwen_asr import Qwen3ASRModel

    backend_kwargs = json.loads(args.backend_kwargs_json)
    if args.backend == "transformers":
        # The Transformers backend loads the full checkpoint locally and is the
        # most straightforward teacher path for moderate batch sizes.
        return Qwen3ASRModel.from_pretrained(
            args.model_path,
            dtype=choose_dtype(),
            device_map=backend_kwargs.pop("device_map", "auto"),
            **backend_kwargs,
        )
    # The vLLM backend keeps the same high-level API here, which lets the rest of
    # the script ignore backend-specific details.
    return Qwen3ASRModel.LLM(model=args.model_path, **backend_kwargs)


def choose_target_language(args, detected_language: str) -> str:
    """Choose which language prefix should be written into the training target."""
    mode = str(args.train_language).strip()
    if not mode:
        return "None"
    if mode.lower() == "auto":
        # ``auto`` means "trust the teacher's detected language and write that
        # back into the training target protocol".
        return detected_language or "None"
    # Any explicit language string forces a consistent prefix for every row.
    return mode


def main():
    """Run pseudo-label generation and persist the result as JSONL.

    The output schema intentionally mirrors supervised training rows so later
    mixed training can be implemented by simple dataset concatenation instead of
    format-specific branches.
    """
    args = parse_args()
    rows = build_input_rows(args)
    asr = load_model(args)
    # ``decode_language`` controls teacher decoding behavior. It is different
    # from ``train_language``, which controls the textual label protocol written
    # into the output JSONL.
    decode_language = args.decode_language.strip() or None
    outputs: List[Dict[str, object]] = []

    for batch in batched(rows, args.batch_size):
        # Batched transcription keeps GPU utilization reasonable while still
        # preserving the simple list-of-records data model at the script level.
        audios = [item["audio"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        results = asr.transcribe(audio=audios, context=prompts, language=decode_language)

        for item, result in zip(batch, results):
            text = (result.text or "").strip()
            if args.skip_empty == 1 and not text:
                # Empty teacher outputs are usually low-value training examples,
                # so dropping them keeps the pseudo set cleaner by default.
                continue
            detected_language = (result.language or "").strip()
            target_language = choose_target_language(args, detected_language)
            out = {
                "audio": item["audio"],
                # The pseudo dataset is written in the same target protocol as
                # supervised rows so later mixed training can concatenate the two
                # datasets with no format-specific branches.
                "text": format_asr_target(text, language=target_language),
                "loss_weight": float(args.loss_weight),
                "pseudo": True,
                "detected_language": detected_language,
                "raw_text": text,
            }
            if item["prompt"]:
                out["prompt"] = item["prompt"]
            outputs.append(out)

    # ``write_jsonl`` gives us a training-ready file that the semisupervised SFT
    # script can consume directly.
    count = write_jsonl(args.output_file, outputs)
    print(f"[pseudo] input rows:  {len(rows)}")
    print(f"[pseudo] output rows: {count}")
    print(f"[pseudo] wrote:       {args.output_file}")


if __name__ == "__main__":
    main()
