"""Prepare Qwen3-ASR fine-tuning manifests from labeled corpora.

This script is the normalization bridge between "real-world annotation exports"
and the much stricter JSONL schema expected by the fine-tuning scripts.

In practice it solves three recurring problems:

1. Annotation tools often export CSV/TSV, while training prefers JSONL.
2. Audio paths in datasets are frequently relative and need to be resolved
   reproducibly.
3. Human transcripts are plain text, but Qwen3-ASR SFT expects targets to use
   the decoder-side protocol ``language X<asr_text>...``.

The output JSONL is intentionally minimal so it can be consumed directly by
``finetuning/qwen3_asr_sft.py`` or the semi-supervised extension script.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

from asr_data_utils import (
    ensure_prompt,
    format_asr_target,
    read_manifest_rows,
    resolve_audio_path,
    split_rows,
    write_jsonl,
)


def parse_args():
    """Define the CLI contract for labeled-data normalization."""
    p = argparse.ArgumentParser("Prepare JSONL for Qwen3-ASR fine-tuning")
    p.add_argument("--input_manifest", required=True, help="csv/tsv/json/jsonl with at least audio and text columns")
    p.add_argument("--input_format", default="auto", choices=["auto", "csv", "tsv", "json", "jsonl"])
    p.add_argument("--audio_key", default="audio")
    p.add_argument("--text_key", default="text")
    p.add_argument("--prompt_key", default="prompt")
    p.add_argument("--audio_root", default="", help="Optional root dir for relative audio paths")
    p.add_argument("--default_prompt", default="", help="Used when prompt column is absent or empty")
    p.add_argument("--language", default="Chinese", help="Prefix language when text has no <asr_text> tag")
    p.add_argument("--output_train", required=True)
    p.add_argument("--output_eval", default="")
    p.add_argument("--eval_ratio", type=float, default=0.05)
    p.add_argument("--shuffle", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_missing_audio", type=int, default=0)
    return p.parse_args()


def normalize_row(row: Dict[str, object], args) -> Dict[str, object]:
    """Normalize one manifest row into the training JSONL schema.

    The function is intentionally strict about required columns. Failing early
    here is better than discovering malformed training data deep inside a GPU
    training run.
    """
    if args.audio_key not in row:
        raise KeyError(f"Missing audio column: {args.audio_key}")
    if args.text_key not in row:
        raise KeyError(f"Missing text column: {args.text_key}")

    # Path resolution is delegated to ``asr_data_utils`` so this script can stay
    # focused on manifest normalization rather than filesystem policy.
    audio = resolve_audio_path(
        str(row[args.audio_key]),
        manifest_path=args.input_manifest,
        audio_root=args.audio_root,
    )
    if not os.path.exists(audio):
        raise FileNotFoundError(audio)

    # The training scripts expect the target side to follow the Qwen3-ASR textual
    # protocol, not just raw transcript text. ``format_asr_target`` adds that
    # protocol wrapper only when it is missing, so preformatted rows remain valid.
    text = format_asr_target(str(row[args.text_key]), language=args.language)
    prompt = ensure_prompt(row.get(args.prompt_key), args.default_prompt)

    item = {
        "audio": audio,
        "text": text,
    }
    if prompt:
        # Prompt is optional. We only materialize it when non-empty so the output
        # JSONL stays compact and visually easy to inspect.
        item["prompt"] = prompt
    return item


def main():
    """Run end-to-end manifest loading, normalization, splitting and writing."""
    args = parse_args()
    # ``read_manifest_rows`` hides the source format differences so the rest of
    # this script can treat every input as ``List[Dict[str, object]]``.
    rows = read_manifest_rows(args.input_manifest, fmt=args.input_format)
    normalized: List[Dict[str, object]] = []
    skipped_missing = 0

    for idx, row in enumerate(rows, start=1):
        try:
            normalized.append(normalize_row(row, args))
        except FileNotFoundError:
            if args.skip_missing_audio == 1:
                # Some annotation exports contain stale paths. Allowing a skip
                # mode is convenient for quick dataset cleanup without rewriting
                # the source manifest first.
                skipped_missing += 1
                continue
            raise FileNotFoundError(f"Missing audio for row {idx}")

    # Splitting happens after normalization so both train and eval sets share the
    # same canonical schema and absolute-path policy.
    train_rows, eval_rows = split_rows(
        normalized,
        eval_ratio=args.eval_ratio if args.output_eval else 0.0,
        seed=args.seed,
        shuffle=(args.shuffle == 1),
    )

    train_count = write_jsonl(args.output_train, train_rows)
    print(f"[prepare] wrote train rows: {train_count} -> {args.output_train}")

    if args.output_eval:
        # ``output_eval`` being empty is the switch for "train only" mode.
        eval_count = write_jsonl(args.output_eval, eval_rows)
        print(f"[prepare] wrote eval rows:  {eval_count} -> {args.output_eval}")

    print(f"[prepare] input rows: {len(rows)}")
    print(f"[prepare] kept rows:  {len(normalized)}")
    print(f"[prepare] skipped missing audio: {skipped_missing}")


if __name__ == "__main__":
    main()
