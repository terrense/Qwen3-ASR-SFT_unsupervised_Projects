"""Utility helpers for ASR manifest preparation and pseudo-label pipelines."""

from __future__ import annotations

import csv
import json
import os
import random
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}


def infer_manifest_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return "jsonl"
    if ext == ".json":
        return "json"
    if ext == ".csv":
        return "csv"
    if ext == ".tsv":
        return "tsv"
    raise ValueError(f"Cannot infer manifest format from extension: {path}")


def read_manifest_rows(path: str, fmt: str = "auto") -> List[Dict[str, object]]:
    """Read rows from csv/tsv/json/jsonl manifest files."""
    fmt = infer_manifest_format(path) if fmt == "auto" else fmt.lower()
    if fmt == "jsonl":
        rows: List[Dict[str, object]] = []
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                item = json.loads(text)
                if not isinstance(item, dict):
                    raise ValueError(f"{path}:{lineno} is not a JSON object")
                rows.append(item)
        return rows

    if fmt == "json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list")
        rows = []
        for idx, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"{path}[{idx}] is not a JSON object")
            rows.append(item)
        return rows

    delimiter = "," if fmt == "csv" else "\t"
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return [dict(row) for row in reader]


def resolve_audio_path(
    audio_path: str,
    *,
    manifest_path: Optional[str] = None,
    audio_root: str = "",
) -> str:
    """Resolve relative audio paths against either audio_root or manifest dir."""
    audio_path = str(audio_path).strip()
    if not audio_path:
        raise ValueError("audio path is empty")

    candidates: List[str] = []
    if os.path.isabs(audio_path):
        candidates.append(audio_path)
    else:
        if audio_root:
            candidates.append(os.path.join(audio_root, audio_path))
        if manifest_path:
            candidates.append(os.path.join(os.path.dirname(os.path.abspath(manifest_path)), audio_path))
        candidates.append(os.path.abspath(audio_path))

    for cand in candidates:
        cand_abs = os.path.abspath(cand)
        if os.path.exists(cand_abs):
            return cand_abs

    return os.path.abspath(candidates[0])


def ensure_prompt(prompt: object, default_prompt: str = "") -> str:
    if prompt is None:
        return default_prompt
    text = str(prompt).strip()
    return text if text else default_prompt


def has_asr_prefix(text: str) -> bool:
    return "<asr_text>" in text


def format_asr_target(text: str, language: str = "Chinese") -> str:
    """Add the repo's expected language prefix when missing."""
    text = str(text).strip()
    if not text:
        raise ValueError("transcript text is empty")
    if has_asr_prefix(text):
        return text

    lang = str(language).strip() or "None"
    return f"language {lang}<asr_text>{text}"


def write_jsonl(path: str, rows: Iterable[Dict[str, object]]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def maybe_limit_rows(rows: Sequence[Dict[str, object]], max_rows: int) -> List[Dict[str, object]]:
    if max_rows <= 0:
        return list(rows)
    return list(rows[: max_rows])


def split_rows(
    rows: Sequence[Dict[str, object]],
    *,
    eval_ratio: float,
    seed: int,
    shuffle: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows = list(rows)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    if eval_ratio <= 0:
        return rows, []

    eval_count = int(round(len(rows) * eval_ratio))
    if len(rows) > 1:
        eval_count = max(1, min(eval_count, len(rows) - 1))
    else:
        eval_count = 0

    if eval_count == 0:
        return rows, []

    return rows[eval_count:], rows[:eval_count]


def collect_audio_files(audio_dir: str, recursive: bool = True) -> List[str]:
    if not os.path.isdir(audio_dir):
        raise ValueError(f"audio directory does not exist: {audio_dir}")

    collected: List[str] = []
    if recursive:
        for root, _, files in os.walk(audio_dir):
            for name in files:
                if os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
                    collected.append(os.path.abspath(os.path.join(root, name)))
    else:
        for name in os.listdir(audio_dir):
            path = os.path.join(audio_dir, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
                collected.append(os.path.abspath(path))

    collected.sort()
    return collected


def batched(items: Sequence[object], batch_size: int) -> Iterator[Sequence[object]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]
