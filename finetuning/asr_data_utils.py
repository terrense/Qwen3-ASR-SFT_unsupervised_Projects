"""Utility helpers for ASR manifest preparation and pseudo-label pipelines.

This file is intentionally lightweight and dependency-thin. The role of these
helpers is not to perform ASR itself, but to normalize the "messy outer world"
around training:

1. Manifest files may arrive as CSV/TSV/JSON/JSONL with slightly different
   field conventions.
2. Audio paths may be absolute, relative to the manifest, or relative to a
   separately mounted corpus root.
3. Text may be raw transcript text, while Qwen3-ASR training expects the
   protocol-like target format ``language X<asr_text>...``.

From a Python design perspective, keeping these helpers pure and mostly
stateless makes them reusable from multiple scripts without dragging model
dependencies into simple data-preparation workflows.
"""

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
    """Infer the manifest parser from the filename extension.

    Python note:
        Returning a small canonical string here lets the rest of the module
        dispatch with explicit ``if`` branches instead of relying on dynamic
        file sniffing, which keeps behavior predictable.
    """
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
    """Read rows from CSV/TSV/JSON/JSONL manifest files.

    The returned value is always ``List[Dict[str, object]]`` so downstream
    training scripts can treat all manifest sources uniformly.
    """
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
        # ``utf-8-sig`` quietly handles BOM-prefixed spreadsheet exports, which
        # are common when manifests were last touched by Excel-like tools.
        reader = csv.DictReader(f, delimiter=delimiter)
        return [dict(row) for row in reader]


def resolve_audio_path(
    audio_path: str,
    *,
    manifest_path: Optional[str] = None,
    audio_root: str = "",
) -> str:
    """Resolve relative audio paths against either ``audio_root`` or manifest dir.

    Search order is deliberate:

    1. If the path is already absolute, keep it.
    2. Try ``audio_root`` first when the caller provides an explicit corpus root.
    3. Fall back to the manifest directory, which is common for self-contained
       datasets.
    4. Finally interpret the value relative to the current working directory.

    Returning the first existing path keeps the caller logic simple while still
    supporting several dataset layouts.
    """
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
            # Returning the first existing candidate gives us a deterministic and
            # easy-to-explain search policy for relative paths.
            return cand_abs

    # If nothing exists yet, we still return the first candidate as a useful
    # "best effort" resolution for later error messages or downstream checks.
    return os.path.abspath(candidates[0])


def ensure_prompt(prompt: object, default_prompt: str = "") -> str:
    """Convert optional prompt-like values into a clean string.

    Python note:
        ``object`` is accepted instead of ``Optional[str]`` because values may
        come from CSV readers or JSON parsing with inconsistent types. The
        helper centralizes the normalization policy in one place.
    """
    if prompt is None:
        return default_prompt
    text = str(prompt).strip()
    # Empty strings collapse back to the caller's default so downstream code
    # never needs to distinguish between ``None``, whitespace, and missing keys.
    return text if text else default_prompt


def has_asr_prefix(text: str) -> bool:
    """Check whether a transcript already uses the Qwen3-ASR target protocol."""
    return "<asr_text>" in text


def format_asr_target(text: str, language: str = "Chinese") -> str:
    """Add the repo's expected language prefix when missing.

    Qwen3-ASR does not train against plain transcript text in this repo's SFT
    scripts. Instead, the target is a lightweight textual protocol such as
    ``language Chinese<asr_text>你好``. That format allows the same decoder to
    emit both language metadata and transcription content.
    """
    text = str(text).strip()
    if not text:
        raise ValueError("transcript text is empty")
    if has_asr_prefix(text):
        # Respect preformatted training targets so advanced users can supply
        # custom protocol strings without this helper rewriting them.
        return text

    # ``None`` is a valid training-time fallback when language metadata is
    # unavailable. The model will still learn ASR, just not an explicit language
    # classification signal from the prefix.
    lang = str(language).strip() or "None"
    return f"language {lang}<asr_text>{text}"


def write_jsonl(path: str, rows: Iterable[Dict[str, object]]) -> int:
    """Write an iterable of dictionaries to UTF-8 JSONL and return row count."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            # ``ensure_ascii=False`` preserves non-ASCII transcripts such as
            # Chinese text in a human-readable form.
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def maybe_limit_rows(rows: Sequence[Dict[str, object]], max_rows: int) -> List[Dict[str, object]]:
    """Optionally truncate rows for debugging or quick smoke tests."""
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
    """Split rows into train/eval partitions with optional deterministic shuffle.

    The implementation intentionally guarantees at least one train example when
    the input has more than one row. That protects quick experiments from
    accidentally producing an empty training set with a high ``eval_ratio``.
    """
    rows = list(rows)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    if eval_ratio <= 0:
        return rows, []

    eval_count = int(round(len(rows) * eval_ratio))
    if len(rows) > 1:
        # Clamp the eval size so quick experiments still keep at least one
        # training sample and one eval sample when possible.
        eval_count = max(1, min(eval_count, len(rows) - 1))
    else:
        eval_count = 0

    if eval_count == 0:
        return rows, []

    return rows[eval_count:], rows[:eval_count]


def collect_audio_files(audio_dir: str, recursive: bool = True) -> List[str]:
    """Collect audio files from a directory in stable sorted order.

    Stable ordering matters for pseudo-label generation because it makes runs
    easier to reproduce and compare.
    """
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

    # Stable sorting is especially useful for pseudo labeling because it makes it
    # easier to compare outputs across runs and to resume manual inspection.
    collected.sort()
    return collected


def batched(items: Sequence[object], batch_size: int) -> Iterator[Sequence[object]]:
    """Yield fixed-size slices from a sequence.

    Python note:
        This helper returns views via slicing rather than building a generator
        over iterators because the scripts using it already materialize lists and
        benefit from straightforward indexing semantics.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        # Slicing keeps the helper simple and works well because the calling
        # scripts already materialize the full list of items in memory.
        yield items[start:start + batch_size]
