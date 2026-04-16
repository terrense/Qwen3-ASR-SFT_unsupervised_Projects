# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility helpers shared by the high-level inference wrappers.

The functions in this file sit between "messy real-world inputs" and the model's
strict expectations. In practice that means:

1. Accepting multiple audio transport formats such as local paths, URLs,
   base64 payloads and in-memory numpy arrays.
2. Canonicalizing waveforms into mono 16 kHz float32 arrays, which is the audio
   space expected by the released Qwen3-ASR checkpoints.
3. Post-processing raw decoded strings into structured ASR outputs that are
   easier for downstream Python code to consume.

Several helpers look deceptively small, but they encode important assumptions
about audio tensor layout, batching semantics and prompt formatting.
"""

# 中文学习备注：
# 这个文件非常值得先读，因为它处在“用户真实输入”和“模型严格要求”之间。
# 你可以把它理解成推理系统的“清洗层 / 协议层”：
#
# 1. 清洗层：
#    把路径、URL、base64、numpy 波形这些乱七八糟的输入，
#    统一收敛成模型能接收的 `mono + 16kHz + float32 + [-1, 1]`。
# 2. 切块层：
#    把过长音频切成更安全的片段，并保留 offset 方便后面合并结果。
# 3. 协议层：
#    模型输出其实仍然是字符串，不是 JSON。
#    所以这里还负责把 `"language Chinese<asr_text>..."` 这类文本协议重新解析成结构化结果。
#
# 如果你在读 `qwen3_asr.py` 时不明白它为什么可以接受各种输入、为什么能自动分块、
# 为什么最后返回 `(language, text)`，答案大多都在本文件里。
import base64
import io
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf

# ``AudioLike`` mirrors the public API accepted by the inference wrappers.
# Using a type alias makes both static reading and future refactors easier than
# repeating the union in every function signature.
AudioLike = Union[
    str,                      # wav path / URL / base64
    Tuple[np.ndarray, int],   # (waveform, sr)
]
# `MaybeList` 是一个很常见的“外部接口友好、内部统一处理”技巧：
# 对外既允许传单个对象，也允许传 list；对内则尽早把它统一成 list。
MaybeList = Union[Any, List[Any]]

# The released checkpoints and feature extractor operate on 16 kHz audio. All
# user-provided audio is resampled into this canonical rate before inference.
SAMPLE_RATE = 16000
# 普通 ASR 最长允许的单段时长；超过后会先切块再识别。
MAX_ASR_INPUT_SECONDS = 1200
# 强制对齐的安全输入长度更短，因为对齐阶段通常更吃资源。
MAX_FORCE_ALIGN_INPUT_SECONDS = 180
# 太短的音频会让特征抽取和模型行为不稳定，所以切块后会按最小时长做零填充。
MIN_ASR_INPUT_SECONDS = 0.5
SUPPORTED_LANGUAGES: List[str] = [
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian"
]
_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


def normalize_language_name(language: str) -> str:
    """
    Normalize language name to the canonical format used by Qwen3-ASR:
    first letter uppercase, the rest lowercase (e.g., 'cHINese' -> 'Chinese').

    Args:
        language (str): Input language name.

    Returns:
        str: Normalized language name.

    Raises:
        ValueError: If language is empty.
    """
    # 这里做的是“规范化”，不是“翻译”。
    # 例如传入 `cHINese`、` chinese `，最终都会统一成 `Chinese`，
    # 这样后续语言校验和 prompt 构造才能用同一套标准名字。
    if language is None:
        raise ValueError("language is None")
    s = str(language).strip()
    if not s:
        raise ValueError("language is empty")
    return s[:1].upper() + s[1:].lower()


def validate_language(language: str) -> None:
    """
    Validate the language is supported.

    Args:
        language (str): Canonical language name.

    Raises:
        ValueError: If unsupported.
    """
    # 本函数不返回布尔值，而是直接在非法时抛错。
    # 这种 fail-fast 风格适合用户输入校验：一旦语言名不合法，就尽早终止，
    # 避免后面进入模型推理后才发现问题。
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}")


def ensure_list(x: MaybeList) -> List[Any]:
    """
    Normalize scalar-or-list inputs into a list.

    This is a common Python API pattern: the public method accepts both a single
    item and a batch, while the internal implementation only works on lists.
    """
    # 读仓库时看到很多函数签名写成 `Union[T, List[T]]`，通常都会配一层这种统一函数。
    # 这样做能显著减少后续 if/else 分支数量。
    return x if isinstance(x, list) else [x]


def is_url(s: str) -> bool:
    """Return True when ``s`` looks like an HTTP(S) audio location."""
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def is_probably_base64(s: str) -> bool:
    """
    Heuristically detect base64-encoded audio payloads.

    This is intentionally permissive: the goal is not perfect MIME detection,
    but routing obviously non-path, non-URL long strings into the base64 decode
    path before we try opening them as local files.
    """
    # 这是一个启发式判断，不保证 100% 准确。
    # 设计重点不是“严格识别 MIME”，而是尽量把明显不是路径的长字符串，
    # 在早期路由到 base64 分支。
    if s.startswith("data:audio"):
        return True
    if ("/" not in s and "\\" not in s) and len(s) > 256:
        return True
    return False


def decode_base64_bytes(b64: str) -> bytes:
    """Decode raw base64 or a ``data:...;base64,`` URL into bytes."""
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def load_audio_any(x: str) -> Tuple[np.ndarray, int]:
    """
    Load audio from a path-like string into ``(waveform, sample_rate)``.

    The implementation picks an IO backend based on transport:
    URL/base64 payloads are decoded with ``soundfile`` from memory, while local
    paths are delegated to ``librosa`` for broader file-format support.
    """
    # 这一步本质上在解决“同一个参数 `x`，可能是三种完全不同的传输介质”：
    # - URL：先下载字节流
    # - base64：先解码字节流
    # - 本地路径：直接从文件系统读取
    #
    # 但无论走哪条路径，最终都要统一成 `(waveform, sample_rate)`。
    if is_url(x):
        with urllib.request.urlopen(x) as resp:
            audio_bytes = resp.read()
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    elif is_probably_base64(x):
        audio_bytes = decode_base64_bytes(x)
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    else:
        audio, sr = librosa.load(x, sr=None, mono=False)

    audio = np.asarray(audio, dtype=np.float32)
    sr = int(sr)
    return audio, sr


def to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert waveform arrays into mono audio.

    AI note:
        Speech encoders in this repository are trained for a single waveform
        stream. Averaging channels is a simple, deterministic downmix strategy
        that preserves timing while discarding stereo spatial information that
        ASR generally does not require.

    Python/numpy note:
        The function accepts either ``(T,)`` or a two-dimensional array. Some
        decoders return ``(T, C)`` while other audio pipelines prefer ``(C, T)``,
        so a shape heuristic is used before reducing the channel axis.
    """
    # 从 ASR 角度看，多声道的空间信息通常不是第一优先级，
    # 时间内容才是关键，所以这里直接做均值 downmix。
    if audio.ndim == 1:
        return audio
    # soundfile can return shape (T, C); some pipelines use (C, T)
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        return np.mean(audio, axis=-1).astype(np.float32)
    raise ValueError(f"Unsupported audio ndim={audio.ndim}")


def float_range_normalize(audio: np.ndarray) -> np.ndarray:
    """
    Keep waveform amplitudes in the float audio range ``[-1, 1]``.

    This is a conservative normalization step. We only rescale when the decoded
    waveform clearly exceeds the usual floating-point audio range, which avoids
    unnecessarily changing already-normalized inputs.
    """
    audio = audio.astype(np.float32)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return audio
    # If decoded audio is int-like scaled or out-of-range, normalize conservatively.
    if peak > 1.0:
        audio = audio / peak
    audio = np.clip(audio, -1.0, 1.0)
    return audio


def normalize_audio_input(a: AudioLike) -> np.ndarray:
    """
    Normalize one audio input to mono 16k float32 waveform in [-1, 1].

    Supported inputs:
        - str: local file path / https URL / base64 audio string
        - (np.ndarray, sr): waveform and sampling rate

    Returns:
        np.ndarray:
            Mono 16k float32 waveform in [-1, 1].
    """
    # 这是“单条音频标准化”的总入口。你可以把它看成 4 个固定阶段：
    # 1. 读入：把路径/URL/base64/ndarray 统一成 `(audio, sr)`
    # 2. 单声道化：统一成 1D waveform
    # 3. 重采样：统一成 16kHz
    # 4. 振幅规范化：统一到 float32 和 [-1, 1]
    #
    # 只要经过这个函数，后续推理代码就可以假设输入已经满足模型契约。
    if isinstance(a, str):
        audio, sr = load_audio_any(a)
    elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
        audio, sr = a[0], int(a[1])
    else:
        raise TypeError(f"Unsupported audio input type: {type(a)}")

    audio = to_mono(np.asarray(audio))
    if sr != SAMPLE_RATE:
        # 重采样是为了和训练/processor 约定完全一致。
        # 即使用户给的是高采样率音频，模型最终看到的仍然是 16kHz。
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)
    audio = float_range_normalize(audio)
    return audio


def normalize_audios(audios: Union[AudioLike, List[AudioLike]]) -> List[np.ndarray]:
    """Batch wrapper around :func:`normalize_audio_input`."""
    # 外部接口允许单个音频或批量音频；内部始终转成 `List[np.ndarray]`。
    items = ensure_list(audios)
    return [normalize_audio_input(a) for a in items]


def chunk_list(xs: List[Any], chunk_size: int) -> Iterable[List[Any]]:
    """
    Yield chunks of a list.

    Args:
        xs (List[Any]): Input list.
        chunk_size (int): Chunk size.

    Yields:
        List[Any]: Slices of xs.
    """
    # `chunk_size <= 0` 被解释成“不分块”，这是工程上很常见的约定，
    # 能让上层把 `-1` 当作“尽量整批跑”的开关。
    if chunk_size <= 0:
        yield xs
        return
    for i in range(0, len(xs), chunk_size):
        yield xs[i : i + chunk_size]


@dataclass(frozen=True)
class AudioChunk:
    """
    One chunk cut from an original audio.

    Attributes:
        orig_index: Index of the original sample in the input batch.
        chunk_index: Index of this chunk within the original sample.
        wav: Mono float32 waveform.
        sr: Sampling rate.
        offset_sec: Start offset of this chunk in the original audio, in seconds.
    """
    # 这个小 dataclass 很关键，因为它保存了“切块后还能回到原始样本时间线”的最小信息：
    # - 属于原 batch 中哪一条样本
    # - 是该样本的第几个 chunk
    # - 这个 chunk 在原音频里从几秒开始
    orig_index: int
    chunk_index: int
    wav: np.ndarray
    sr: int
    offset_sec: float


def split_audio_into_chunks(
    wav: np.ndarray,
    sr: int,
    max_chunk_sec: float,
    search_expand_sec: float = 5.0,
    min_window_ms: float = 100.0,
) -> List[Tuple[np.ndarray, float]]:
    """
    Split a long audio into chunks close to max_chunk_sec, using a low-energy boundary.

    This implementation guarantees:
      - Concatenating all returned chunks reproduces the original audio exactly
        (total number of samples is identical, no overlaps, no gaps).

    Args:
        wav: Mono waveform float32.
        sr: Sampling rate.
        max_chunk_sec: Target max chunk duration in seconds.
        search_expand_sec: Boundary search half-window in seconds.
        min_window_ms: Sliding window in milliseconds for energy estimation.

    Returns:
        List[Tuple[np.ndarray, float]]: List of (chunk_wav, offset_sec).
    """
    # 这是长音频处理的关键函数。
    # 设计目标不是简单地“每 N 秒硬切一刀”，而是尽量在目标切点附近寻找低能量位置，
    # 避免把一个音节或词语从中间截断。
    #
    # 同时它还必须满足一个工程上很重要的性质：
    # 切块后再按顺序拼回去，样本总数必须和原音频完全一致。
    # 也就是：
    # - 没有重叠
    # - 没有空洞
    # - 不会丢采样点
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1).astype(np.float32)

    total_len = int(wav.shape[0])
    total_sec = total_len / float(sr)
    if total_sec <= max_chunk_sec:
        # 足够短时不做任何切块，直接原样返回，offset 从 0 开始。
        return [(wav, 0.0)]

    # ``max_len`` is the nominal cut point. We then look around that area for a
    # lower-energy boundary so the split is less likely to bisect a syllable.
    max_len = int(max_chunk_sec * sr)
    expand = int(search_expand_sec * sr)
    win = max(4, int((min_window_ms / 1000.0) * sr))

    chunks: List[Tuple[np.ndarray, float]] = []

    start = 0
    offset_sec = 0.0

    while (total_len - start) > max_len:
        cut = start + max_len

        left = max(start, cut - expand)
        right = min(total_len, cut + expand)

        if right - left <= win:
            # 搜索窗口过小时，说明附近没什么可选余地，只能回退到名义切点。
            boundary = cut
        else:
            seg = wav[left:right]
            seg_abs = np.abs(seg)

            # Sliding-window energy is a simple proxy for "speech activity".
            # The minimum-energy window near the target cut is a practical place
            # to split long utterances while keeping chunk concatenation exact.
            window_sums = np.convolve(seg_abs, np.ones(win, dtype=np.float32), mode="valid")

            min_pos = int(np.argmin(window_sums))

            wstart = min_pos
            wend = min_pos + win
            local = seg_abs[wstart:wend]
            inner = int(np.argmin(local))
            boundary = left + wstart + inner

        # 防御性约束：
        # - 至少往前推进 1 个 sample，避免死循环
        # - 不能越过整段音频末尾
        boundary = int(max(boundary, start + 1))
        boundary = int(min(boundary, total_len))

        chunk = wav[start:boundary]
        chunks.append((chunk, offset_sec))

        offset_sec += (boundary - start) / float(sr)
        start = boundary

    tail = wav[start:total_len]
    chunks.append((tail, offset_sec))

    # Very short clips are awkward for encoder feature extraction. Padding only
    # happens after chunk boundaries are finalized so timestamp offsets still
    # refer to the original unpadded audio.
    # 中文学习备注：
    # 这里先切边界，再补 pad，非常重要。
    # 因为 offset 的意义必须始终对应“原始真实音频时间线”，
    # 不能让后加的静音 padding 污染时间定位。
    min_len = int(MIN_ASR_INPUT_SECONDS * sr)
    padded: List[Tuple[np.ndarray, float]] = []
    for c, off in chunks:
        if c.shape[0] < min_len:
            pad = min_len - int(c.shape[0])
            c = np.pad(c, (0, pad), mode="constant", constant_values=0.0).astype(np.float32)
        padded.append((c, off))
    chunks = padded

    return chunks


def detect_and_fix_repetitions(text: str, threshold: int = 20) -> str:
    """
    Collapse pathological repetition artifacts from decoder output.

    Generative ASR models can occasionally loop when decoding unstable audio
    spans, producing outputs such as the same character or token pattern dozens
    of times. This function first removes obvious character runs and then looks
    for short repeated substrings.
    """
    # 这是生成式 ASR 常见的后处理兜底：
    # 当音频含糊、边界不稳定或模型解码发散时，可能出现“啊啊啊啊啊...”、
    # “hellohellohello...” 这种循环输出。
    #
    # 这里不是做语言学上的精确纠错，而是优先处理明显失真的病理重复。
    def fix_char_repeats(s, thresh):
        # 第一层：处理单字符连续爆炸重复，例如 "哈哈哈哈...."
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1

            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i:i+count])
                i += count
        return ''.join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        # 第二层：处理短模式重复，例如 "你好你好你好..." / "abcabcabc..."
        # `max_len` 限制了待尝试的重复单元长度，避免在长字符串上退化得太慢。
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s
            
        i = 0
        result = []
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                    
                pattern = s[i:i+k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx:start_idx+k] != pattern:
                        valid = False
                        break
                
                if valid:
                    total_rep = thresh
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index:end_index+k] == pattern:
                        total_rep += 1
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break
            
            if found:
                break
            else:
                result.append(s[i])
                i += 1

        if not found:
            result.append(s[i:])
        return ''.join(result)
    
    text_raw = text
    text = fix_char_repeats(text_raw, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text


def parse_asr_output(
    raw: str,
    user_language: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Parse Qwen3-ASR raw output into (language, text).

    Cases:
      - With tag: "language Chinese<asr_text>...."
      - With newlines: "language Chinese\\n...\\n<asr_text>...."
      - No tag: treat whole string as text.
      - "language None<asr_text>": treat as empty audio -> ("", "")

    If user_language is provided, language is forced to user_language and raw is treated as text-only
    (the model is expected to output plain transcription without metadata).

    Args:
        raw: Raw decoded string.
        user_language: Canonical language name if user forced language.

    Returns:
        Tuple[str, str]: (language, text)
    """
    # 这是整个推理协议里最关键的“文本 -> 结构化结果”桥接函数之一。
    #
    # 要点在于：模型本质上仍是语言模型，所以输出首先是一段字符串；
    # 但上层应用想要的是结构化结果 `(language, text)`。
    # 因此这里需要把一个轻量文本协议重新解释回来。
    #
    # 你可以把它理解成一个很小的 parser，而不是普通字符串清洗函数。
    if raw is None:
        return "", ""
    s = str(raw).strip()
    if not s:
        return "", ""

    # Raw generation is still "language model text", so we clean obvious decode
    # loops before interpreting the output format.
    s = detect_and_fix_repetitions(s)

    if user_language:
        # user explicitly forced language => model output is treated as pure text
        # 这里的含义是：既然用户已经指定语言，那么模型就不需要再“自报语言”，
        # 我们直接把生成串当成纯文本转写结果使用。
        return user_language, s

    meta_part = s
    text_part = ""
    has_tag = _ASR_TEXT_TAG in s
    if has_tag:
        meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)
    else:
        # no tag => pure text
        # 没看到 `<asr_text>` 时，就按“整串都是文本”处理。
        # 这是一个兼容策略，避免 parser 对轻微格式漂移过于脆弱。
        return "", s.strip()

    meta_lower = meta_part.lower()

    # empty audio heuristic
    if "language none" in meta_lower:
        t = text_part.strip()
        if not t:
            # 空音频或静音段的常见协议形式。
            return "", ""
        # if model still returned something, keep it but language unknown
        return "", t

    # The model uses a lightweight textual protocol instead of returning JSON.
    # We therefore recover structured fields with a small parser rather than a
    # tokenizer-dependent regex.
    lang = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(_LANG_PREFIX):
            val = line[len(_LANG_PREFIX):].strip()
            if val:
                lang = normalize_language_name(val)
            break

    return lang, text_part.strip()


def merge_languages(langs: List[str]) -> str:
    """
    Merge per-chunk languages into a compact comma-separated string,
    keeping order and removing consecutive duplicates and empty entries.

    Example:
      ["Chinese", "English", "English"] -> "Chinese,English"

    Args:
        langs: List of canonical language names.

    Returns:
        str: Merged language string.
    """
    # 多 chunk 长音频里，不同片段可能预测出不同语言。
    # 这里不做复杂投票，而是保留时间顺序，并去掉空值和连续重复项。
    # 因此它表达的更像“语言轨迹摘要”，而不是单一分类标签。
    out: List[str] = []
    prev = None
    for x in langs:
        x = (x or "").strip()
        if not x:
            continue
        if x == prev:
            continue
        out.append(x)
        prev = x
    return ",".join(out)
