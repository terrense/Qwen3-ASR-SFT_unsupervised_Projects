# coding=utf-8
from __future__ import annotations

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
High-level ASR wrapper for the released Qwen3-ASR checkpoints.

This file is the main "developer ergonomics" layer of the repository. The lower
backend implementations focus on tensor operations, while this wrapper focuses
on application concerns:

1. Accepting flexible Python audio inputs.
2. Broadcasting scalar arguments into batches.
3. Splitting long recordings into model-safe windows.
4. Dispatching to either Transformers or vLLM backends.
5. Re-assembling chunk outputs and optional forced-alignment timestamps into a
   user-friendly result structure.
"""

# 中文学习备注：
# 如果说 `modeling_qwen3_asr.py` 解决的是“张量怎么算”，
# 那本文件解决的是“用户到底怎么方便地用这个模型”。
#
# 它是整个仓库里最典型的“应用层封装”：
# - 上游接收用户的 Python 参数和多种音频输入形式
# - 中间把长音频切块、整理 prompt、分发到不同后端
# - 下游把模型输出重新拼成一个稳定、好用的 Python 结果对象
#
# 因此读这个文件时，最好不要把它当成“模型细节文件”，而是当成：
# “从用户调用到模型返回之间的总调度器”。
#
# 最推荐的阅读主线是：
# 1. `Qwen3ASRModel.from_pretrained()` / `Qwen3ASRModel.LLM()`
#    看对象是如何被初始化成 Transformers / vLLM 两种后端的。
# 2. `transcribe()`
#    看离线推理主流程：标准化 -> 广播 -> 切块 -> 推理 -> 解析 -> 合并。
# 3. `_infer_asr_transformers()` / `_infer_asr_vllm()`
#    看两种后端在请求格式上的差异。
# 4. `init_streaming_state()` / `streaming_transcribe()` / `finish_streaming_transcribe()`
#    看流式识别是如何通过“重复全上下文解码 + prefix 回退”实现的。
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from transformers import AutoConfig, AutoModel, AutoProcessor

# Registering custom config/model/processor classes makes the checkpoints work
# with Hugging Face auto classes just like first-party architectures.
# 中文学习备注：
# 这三行注册非常重要，它们在做“自定义架构接入 Hugging Face 生态”这件事。
# 注册后，外部只需要拿到 checkpoint + config 里的 `model_type=qwen3_asr`，
# 就能通过 `AutoConfig/AutoModel/AutoProcessor` 自动恢复正确类。
AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

# Forced aligner is optional. Keep ASR importable even when its language-
# specific dependency stack (e.g. `nagisa`) is not installed.
try:
    from .qwen3_forced_aligner import Qwen3ForcedAligner
    _FORCED_ALIGNER_IMPORT_ERROR = None
except Exception as _exc:
    Qwen3ForcedAligner = None  # type: ignore[assignment]
    _FORCED_ALIGNER_IMPORT_ERROR = _exc
from .utils import (
    MAX_ASR_INPUT_SECONDS,
    MAX_FORCE_ALIGN_INPUT_SECONDS,
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    AudioChunk,
    AudioLike,
    chunk_list,
    merge_languages,
    normalize_audios,
    normalize_language_name,
    parse_asr_output,
    split_audio_into_chunks,
    validate_language,
)

try:
    from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration
    from vllm import ModelRegistry
    # vLLM 也需要知道“某个模型名应该映射到哪个 Python 类”。
    # 这里的注册作用，和上面的 AutoModel 注册在思想上是相同的，
    # 只是服务对象从 HF 变成了 vLLM。
    ModelRegistry.register_model("Qwen3ASRForConditionalGeneration", Qwen3ASRForConditionalGeneration)
except:
    # vLLM is an optional dependency. Import failure is deferred until the user
    # explicitly asks for the vLLM backend.
    pass


@dataclass
class ASRTranscription:
    """
    One transcription result.

    Attributes:
        language (str):
            Merged language string for the sample, e.g. "Chinese" or "Chinese,English".
            Empty string if unknown or silent audio.
        text (str):
            Transcribed text.
        time_stamps (Optional[Any]):
            Forced aligner output (ForcedAlignResult).
            Present only when return_time_stamps=True.
    """
    # 这个 dataclass 是离线识别最终返回给调用者的“稳定结果壳”。
    # 相比直接返回 tuple/dict，它的优点是字段语义更明确，也更方便 IDE 补全。
    language: str
    text: str
    time_stamps: Optional[Any] = None


@dataclass
class ASRStreamingState:
    """
    Streaming ASR state for one audio stream (single utterance).

    Attributes:
        unfixed_chunk_num (int):
            For the first N chunks, do not use previous ASR result as prefix prompt (reset prefix to "").
        unfixed_token_num (int):
            When chunk_id >= unfixed_chunk_num, rollback the last K tokens from the accumulated text
            before using it as prefix prompt, to reduce boundary jitter.
        chunk_size_sec (float):
            Chunk size in seconds. Audio will be fed to the model in increments of this length.
        chunk_size_samples (int):
            Chunk size in samples at 16kHz (derived from chunk_size_sec).
        chunk_id (int):
            Current chunk index (0-based).
        buffer (np.ndarray):
            Buffered PCM samples that are not yet consumed into a full chunk.
        audio_accum (np.ndarray):
            Accumulated audio from the beginning of the stream up to current time (no padding).
        prompt_raw (str):
            Base prompt generated by chat template (with generation prompt), without appended prefix text.
        context (str):
            Context string.
        force_language (Optional[str]):
            If provided, force output to be text-only by appending "language X<asr_text>" in prompt_raw,
            consistent with non-streaming transcribe().
        language (str):
            Latest parsed language (updated after each chunk decode). Empty if unknown/silent.
        text (str):
            Latest parsed transcription text (updated after each chunk decode).
        _raw_decoded (str):
            Internal accumulated decoded raw text (before parse_asr_output normalization).
            Used for rollback/token trimming and as prefix for prompting.
    """
    # 流式识别没有一次性完成，所以需要一个可变状态对象把跨 chunk 的上下文记下来。
    # 可以把它理解成“单条音频流会话”的运行时上下文。
    unfixed_chunk_num: int
    unfixed_token_num: int
    chunk_size_sec: float
    chunk_size_samples: int

    chunk_id: int
    buffer: np.ndarray
    audio_accum: np.ndarray

    prompt_raw: str
    context: str
    force_language: Optional[str]

    language: str
    text: str
    _raw_decoded: str


class Qwen3ASRModel:
    """
    Unified inference wrapper for Qwen3-ASR with two backends:
      - Transformers backend 
      - vLLM backend

    It optionally supports time stamp output via Qwen3-ForcedAligner.

    Notes:
      - Each request uses a context text and exactly one audio.
      - If language is provided, the prompt will force the output to be text-only by appending
        "language {Language}<asr_text>" to the assistant prompt.
    """
    # 这是仓库里最重要的用户级封装类。
    # 它的核心设计不是继承某个底层模型，而是“组合”不同后端对象，
    # 再暴露出统一的 ASR 接口。
    #
    # 这样做的最大好处是：
    # - 用户只学一个 `Qwen3ASRModel`
    # - 底层到底是 Transformers 还是 vLLM，对调用接口影响很小
    # - 应用层逻辑（切块、合并、语言解析、时间戳对齐）可以复用

    def __init__(
        self,
        backend: str,
        model: Any,
        processor: Any,
        sampling_params: Optional[Any] = None,
        forced_aligner: Optional[Qwen3ForcedAligner] = None,
        max_inference_batch_size: int = -1,
        max_new_tokens: int = 512,
    ):
        """
        Store backend objects behind a uniform ASR-oriented interface.

        Python note:
            The wrapper does not subclass a backend-specific model class. It uses
            plain composition instead, which keeps the public API stable even
            though the underlying execution engines differ significantly.
        """
        # `backend` 是整个类后续分派逻辑的核心开关。
        # 很多方法名相同，但底层执行路径完全不同，因此这里先把关键对象都存下来。
        self.backend = backend  # "transformers" | "vllm"
        self.model = model
        self.processor = processor
        self.sampling_params = sampling_params
        self.forced_aligner = forced_aligner
        self.max_inference_batch_size = int(max_inference_batch_size)
        self.max_new_tokens = max_new_tokens

        if backend == "transformers":
            # Transformers 后端直接持有 torch model，因此这里顺手缓存 device/dtype，
            # 方便后面把 processor 输出迁移到同一设备和精度。
            self.device = getattr(model, "device", None)
            if self.device is None:
                try:
                    self.device = next(model.parameters()).device
                except StopIteration:
                    self.device = torch.device("cpu")
            self.dtype = getattr(model, "dtype", torch.float32)
        else:
            # vLLM 后端不通过这里手动 `.to(device)`，所以 device/dtype 对应用层意义不大。
            self.device = None
            self.dtype = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        forced_aligner: Optional[str] = None,
        forced_aligner_kwargs: Optional[Dict[str, Any]] = None,
        max_inference_batch_size: int = 32,
        max_new_tokens: Optional[int] = 512,
        **kwargs,
    ) -> "Qwen3ASRModel":
        """
        Initialize using Transformers backend.

        Args:
            pretrained_model_name_or_path:
                HuggingFace repo id or local directory.
            forced_aligner:
                Optional forced aligner model path/repo id.
            forced_aligner_kwargs:
                Optional kwargs forwarded to Qwen3ForcedAligner.from_pretrained(...).
            max_inference_batch_size:
                Batch size limit for inference. -1 means no chunking. Small values can avoid OOM.
            max_new_tokens:
                Maximum number of tokens to generate.
            **kwargs:
                Forwarded to AutoModel.from_pretrained(...).

        Returns:
            Qwen3ASRModel
        """
        # 这条路径对应 Hugging Face / PyTorch 推理。
        # 用户给一个 repo id 或本地模型目录，底层会恢复：
        # - model：自定义 Qwen3-ASR 模型
        # - processor：文本模板 + 音频特征提取器
        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)

        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True)

        forced_aligner_model = None
        if forced_aligner is not None:
            if Qwen3ForcedAligner is None:
                raise ImportError(
                    "Forced aligner dependencies are not available. "
                    "Install optional dependencies required by `qwen3_forced_aligner.py`."
                ) from _FORCED_ALIGNER_IMPORT_ERROR
            forced_aligner_model = Qwen3ForcedAligner.from_pretrained(
                forced_aligner, **(forced_aligner_kwargs or {})
            )

        # 注意这里返回的不是底层 HF model，而是统一包装后的 `Qwen3ASRModel`。
        return cls(
            backend="transformers",
            model=model,
            processor=processor,
            sampling_params=None,
            forced_aligner=forced_aligner_model,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )

    @classmethod
    def LLM(
        cls,
        model: str,
        forced_aligner: Optional[str] = None,
        forced_aligner_kwargs: Optional[Dict[str, Any]] = None,
        max_inference_batch_size: int = -1,
        max_new_tokens: Optional[int] = 4096,
        **kwargs,
    ) -> "Qwen3ASRModel":
        """
        Initialize using vLLM backend.

        Import is isolated to keep vLLM optional.

        Args:
            model:
                Model path/repo for vLLM.
            forced_aligner:
                Optional forced aligner model path/repo id.
            forced_aligner_kwargs:
                Optional kwargs forwarded to Qwen3ForcedAligner.from_pretrained(...).
            max_inference_batch_size:
                Batch size limit for inference. -1 means no chunking. Small values can avoid OOM.
            max_new_tokens:
                Maximum number of tokens to generate.
            **kwargs:
                Forwarded to vllm.LLM(...).

        Returns:
            Qwen3ASRModel

        Raises:
            ImportError: If vLLM is not installed.
        """
        # 这条路径对应 vLLM 推理，特点是更偏部署/吞吐场景。
        # 与 `from_pretrained()` 不同，这里底层拿到的是 vllm.LLM 引擎实例。
        try:
            from vllm import LLM as vLLM
            from vllm import SamplingParams
        except Exception as e:
            raise ImportError(
                "vLLM is not available. Install with: pip install qwen-asr[vllm]"
            ) from e

        llm = vLLM(model=model, **kwargs)

        processor = Qwen3ASRProcessor.from_pretrained(model, fix_mistral_regex=True)
        # vLLM 把采样参数单独包装成对象，和 Transformers 的 generate kwargs 不同。
        sampling_params = SamplingParams(**({"temperature": 0.0, "max_tokens": max_new_tokens}))

        forced_aligner_model = None
        if forced_aligner is not None:
            if Qwen3ForcedAligner is None:
                raise ImportError(
                    "Forced aligner dependencies are not available. "
                    "Install optional dependencies required by `qwen3_forced_aligner.py`."
                ) from _FORCED_ALIGNER_IMPORT_ERROR
            forced_aligner_model = Qwen3ForcedAligner.from_pretrained(
                forced_aligner, **(forced_aligner_kwargs or {})
            )

        return cls(
            backend="vllm",
            model=llm,
            processor=processor,
            sampling_params=sampling_params,
            forced_aligner=forced_aligner_model,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=None,
        )

    def get_supported_languages(self) -> List[str]:
        """
        Returns the supported language list.

        Returns:
            List[str]: Canonical language names.
        """
        # 返回副本而不是原列表，避免调用者意外修改模块级常量。
        return list(SUPPORTED_LANGUAGES)

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ) -> List[ASRTranscription]:
        """
        Transcribe audio with optional context and optional forced alignment timestamps.

        Args:
            audio:
                Audio input(s). Supported:
                  - str: local path / URL / base64 data url
                  - (np.ndarray, sr)
                  - list of above
            context:
                Context string(s). If scalar, it will be broadcast to batch size.
            language:
                Optional language(s). If provided, it must be in supported languages.
                If scalar, it will be broadcast to batch size.
                If provided, the prompt will force output to be transcription text only.
            return_time_stamps:
                If True, timestamps are produced via forced aligner and merged across chunks.
                This requires forced_aligner initialized.

        Returns:
            List[ASRTranscription]: One result per input audio.

        Raises:
            ValueError:
                - If return_time_stamps=True but forced_aligner is not provided.
                - If language is unsupported.
                - If batch sizes mismatch for context/language.
        """
        # 可以把 `transcribe()` 理解成 7 个连续阶段：
        # 1. 输入音频标准化
        # 2. 标量参数广播成 batch
        # 3. 长音频切块
        # 4. chunk 级模型推理
        # 5. 文本协议解析
        # 6. 可选时间戳对齐
        # 7. 按原样本重新合并
        if return_time_stamps and self.forced_aligner is None:
            raise ValueError("return_time_stamps=True requires `forced_aligner` to be provided at initialization.")

        # Step 1: canonicalize every input waveform into the feature space the
        # encoder expects: mono, 16 kHz, float32, no batch ambiguity.
        wavs = normalize_audios(audio)
        n = len(wavs)

        # Step 2: broadcast optional scalar metadata into per-sample lists so
        # the rest of the pipeline can stay batch-uniform.
        ctxs = context if isinstance(context, list) else [context]
        if len(ctxs) == 1 and n > 1:
            ctxs = ctxs * n
        if len(ctxs) != n:
            raise ValueError(f"Batch size mismatch: audio={n}, context={len(ctxs)}")

        langs_in: List[Optional[str]]
        if language is None:
            langs_in = [None] * n
        else:
            langs_in = language if isinstance(language, list) else [language]
            if len(langs_in) == 1 and n > 1:
                langs_in = langs_in * n
            if len(langs_in) != n:
                raise ValueError(f"Batch size mismatch: audio={n}, language={len(langs_in)}")

        langs_norm: List[Optional[str]] = []
        for l in langs_in:
            if l is None or str(l).strip() == "":
                langs_norm.append(None)
            else:
                # Validate early so user input errors fail fast before any costly
                # model or aligner work starts.
                ln = normalize_language_name(str(l))
                validate_language(ln)
                langs_norm.append(ln)

        # Forced alignment uses a shorter model-safe limit than plain ASR, so the
        # chunking policy depends on the requested output type.
        max_chunk_sec = MAX_FORCE_ALIGN_INPUT_SECONDS if return_time_stamps else MAX_ASR_INPUT_SECONDS

        # Step 3: long-form inference is implemented as "chunk, decode, merge".
        # ``AudioChunk`` keeps the bookkeeping needed to reconstruct the original
        # batch order and timeline later.
        chunks: List[AudioChunk] = []
        for i, wav in enumerate(wavs):
            parts = split_audio_into_chunks(
                wav=wav,
                sr=SAMPLE_RATE,
                max_chunk_sec=max_chunk_sec,
            )
            for j, (cwav, offset_sec) in enumerate(parts):
                # `orig_index + chunk_index + offset_sec` 这三项组合起来，
                # 就足以在后处理阶段恢复原 batch 顺序和时间线。
                chunks.append(AudioChunk(orig_index=i, chunk_index=j, wav=cwav, sr=SAMPLE_RATE, offset_sec=offset_sec))

        # Step 4: run chunk-level recognition in backend-sized batches.
        chunk_ctx: List[str] = [ctxs[c.orig_index] for c in chunks]
        chunk_lang: List[Optional[str]] = [langs_norm[c.orig_index] for c in chunks]
        chunk_wavs: List[np.ndarray] = [c.wav for c in chunks]
        raw_outputs = self._infer_asr(chunk_ctx, chunk_wavs, chunk_lang)

        # Step 5: parse the model's textual protocol back into structured data.
        per_chunk_lang: List[str] = []
        per_chunk_text: List[str] = []
        for out, forced_lang in zip(raw_outputs, chunk_lang):
            # 若用户强制指定语言，则 parser 直接把输出当纯文本看；
            # 否则 parser 会尝试从模型输出中提取 `language ... <asr_text> ...` 结构。
            lang, txt = parse_asr_output(out, user_language=forced_lang)
            per_chunk_lang.append(lang)
            per_chunk_text.append(txt)

        # Step 6: optionally recover timestamps by aligning each recognized chunk
        # against its waveform and then shifting the local offsets back into the
        # original audio timeline.
        per_chunk_align: List[Optional[Any]] = [None] * len(chunks)
        if return_time_stamps:
            to_align_audio = []
            to_align_text = []
            to_align_lang = []
            to_align_idx = []

            for idx, (c, txt, lang_pred) in enumerate(zip(chunks, per_chunk_text, per_chunk_lang)):
                if txt.strip() == "":
                    # 空文本没有必要送去对齐器，既节省时间，也避免无意义输出。
                    continue
                to_align_audio.append((c.wav, c.sr))
                to_align_text.append(txt)
                to_align_lang.append(lang_pred)
                to_align_idx.append(idx)

            # Reuse the same batching policy as ASR to reduce GPU spikes during
            # alignment on large request sets.
            aligned_results: List[Any] = []
            for a_chunk, t_chunk, l_chunk in zip(
                chunk_list(to_align_audio, self.max_inference_batch_size),
                chunk_list(to_align_text, self.max_inference_batch_size),
                chunk_list(to_align_lang, self.max_inference_batch_size),
            ):
                aligned_results.extend(
                    self.forced_aligner.align(audio=a_chunk, text=t_chunk, language=l_chunk)
                )

            # The aligner works in chunk-local time. We restore absolute times by
            # adding the chunk offset collected during audio splitting.
            for k, idx in enumerate(to_align_idx):
                c = chunks[idx]
                r = aligned_results[k]
                per_chunk_align[idx] = self._offset_align_result(r, c.offset_sec)

        # Step 7: merge chunk-level outputs back into per-audio results.
        out_langs: List[List[str]] = [[] for _ in range(n)]
        out_texts: List[List[str]] = [[] for _ in range(n)]
        out_aligns: List[List[Any]] = [[] for _ in range(n)]

        for c, lang, txt, al in zip(chunks, per_chunk_lang, per_chunk_text, per_chunk_align):
            out_langs[c.orig_index].append(lang)
            out_texts[c.orig_index].append(txt)
            if return_time_stamps and al is not None:
                out_aligns[c.orig_index].append(al)

        results: List[ASRTranscription] = []
        for i in range(n):
            # We concatenate chunk texts directly because the model already emits
            # language-appropriate spacing/punctuation. Injecting our own
            # separator would often damage Chinese text or duplicate spaces.
            merged_text = "".join([t for t in out_texts[i] if t is not None])
            merged_language = merge_languages(out_langs[i])
            merged_align = None
            if return_time_stamps:
                merged_align = self._merge_align_results(out_aligns[i])
            results.append(ASRTranscription(language=merged_language, text=merged_text, time_stamps=merged_align))

        return results

    def _build_messages(self, context: str, audio_payload: Any) -> List[Dict[str, Any]]:
        """Build the chat-style multimodal message list expected by the processor."""
        # Qwen3-ASR 继承了 chat-style multimodal prompt 习惯：
        # system 放上下文，user content 里放 audio。
        return [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
        ]

    def _build_text_prompt(self, context: str, force_language: Optional[str]) -> str:
        """
        Build the string prompt for one request.

        If force_language is provided, "language X<asr_text>" is appended after the generation prompt
        to request text-only output.
        """
        # 关键点：
        # 这里构造的是“文本 prompt 外壳”，不是实际音频张量输入。
        # 真正的 waveform 会在 processor/model 或 vLLM multi_modal_data 里单独提供。
        # The template renderer only needs the multimodal conversation shape to
        # produce the textual prompt. Actual waveform tensors are supplied via
        # the processor/model path later.
        msgs = self._build_messages(context=context, audio_payload="")
        base = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if force_language:
            # This suffix asks the model to skip language self-reporting and emit
            # plain ASR text in the user-specified language regime.
            base = base + f"language {force_language}{'<asr_text>'}"
        return base

    def _infer_asr(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
    ) -> List[str]:
        """
        Run backend inference for chunk-level items.

        Args:
            contexts: List of context strings.
            wavs: List of mono waveforms (np.ndarray).
            languages: List of forced languages or None.

        Returns:
            List[str]: Raw decoded strings (one per chunk).
        """
        # 这是一个薄分派层，真正目的只有一个：
        # 根据 `self.backend` 把统一的 chunk 输入路由到对应后端实现。
        if self.backend == "transformers":
            return self._infer_asr_transformers(contexts, wavs, languages)
        if self.backend == "vllm":
            return self._infer_asr_vllm(contexts, wavs, languages)
        raise RuntimeError(f"Unknown backend: {self.backend}")

    def _infer_asr_transformers(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
    ) -> List[str]:
        """
        Run offline generation with the Hugging Face backend.

        The processor creates both tokenized text inputs and audio features; the
        model then autoregressively generates new tokens after the prompt.
        """
        # Transformers 路径的关键数据流是：
        # prompt text + raw wav
        # -> processor
        # -> input_ids / input_features / masks
        # -> model.generate()
        # -> decode newly generated suffix
        outs: List[str] = []

        texts = [self._build_text_prompt(context=c, force_language=fl) for c, fl in zip(contexts, languages)]

        batch_size = self.max_inference_batch_size
        if batch_size is None or batch_size < 0:
            batch_size = len(texts)

        for i in range(0, len(texts), batch_size):
            sub_text = texts[i : i + batch_size]
            sub_wavs = wavs[i : i + batch_size]
            # processor 同时做两件事：
            # 1. 把文本 prompt token 化
            # 2. 把音频转成模型所需特征
            inputs = self.processor(text=sub_text, audio=sub_wavs, return_tensors="pt", padding=True)
            inputs = inputs.to(self.model.device).to(self.model.dtype)

            text_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

            # Only decode the newly generated suffix. The processor input prompt
            # already occupies the prefix of ``text_ids.sequences``.
            decoded = self.processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outs.extend(list(decoded))

        return outs

    def _infer_asr_vllm(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
    ) -> List[str]:
        """Run chunk-level ASR with vLLM's prompt-plus-multimodal-data API."""
        # vLLM 路径和 Transformers 最大差异在于请求格式：
        # 它不是把所有张量先在 Python 侧整理好再喂模型，
        # 而是把 prompt 和多模态数据按引擎约定打包成 request dict。
        inputs: List[Dict[str, Any]] = []
        for c, w, fl in zip(contexts, wavs, languages):
            prompt = self._build_text_prompt(context=c, force_language=fl)
            inputs.append({"prompt": prompt, "multi_modal_data": {"audio": [w]}})

        outs: List[str] = []
        for batch in chunk_list(inputs, self.max_inference_batch_size):
            # vLLM preserves request order within one generate call, so appending
            # outputs in sequence keeps them aligned with ``chunks``.
            outputs = self.model.generate(batch, sampling_params=self.sampling_params, use_tqdm=False)
            for o in outputs:
                outs.append(o.outputs[0].text)
        return outs

    def _offset_align_result(self, result: Any, offset_sec: float) -> Any:
        """
        Apply time offset to a ForcedAlignResult-like object.

        This function assumes:
          - result has attribute `.items` which is a list of items with start_time/end_time in seconds.
          - dataclasses are frozen in upstream implementation, so we reconstruct by type.

        Args:
            result: ForcedAlignResult
            offset_sec: Offset in seconds

        Returns:
            ForcedAlignResult: New object with shifted timestamps.
        """
        # 对齐器返回的是“chunk 内局部时间”，而用户想看到的是“原始整段音频绝对时间”。
        # 这里就是把 chunk 起点 `offset_sec` 加回去。
        if result is None:
            return None
        # Upstream dataclasses are frozen, so we reconstruct them instead of
        # mutating in place.
        items = []
        for it in result.items:
            items.append(type(it)(text=it.text, 
                                  start_time=round(it.start_time + offset_sec, 3), 
                                  end_time=round(it.end_time + offset_sec, 3)))
        return type(result)(items=items)

    def _merge_align_results(self, results: List[Any]) -> Optional[Any]:
        """
        Merge multiple ForcedAlignResult objects into a single one by concatenating items.

        Args:
            results: List of ForcedAlignResult

        Returns:
            ForcedAlignResult or None
        """
        # 因为 chunk 顺序本身就是按时间顺序生成的，所以这里不需要再排序，
        # 直接顺序拼接 items 即可得到完整时间线。
        if not results:
            return None
        all_items = []
        for r in results:
            if r is None:
                continue
            # ``results`` is already ordered by chunk chronology, so a plain
            # extend preserves the final timeline.
            all_items.extend(list(r.items))
        if not all_items:
            return None
        return type(results[0])(items=all_items)

    def init_streaming_state(
        self,
        context: str = "",
        language: Optional[str] = None,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 2.0,
    ) -> ASRStreamingState:
        """
        Initialize streaming ASR state for a single stream.

        Notes:
            - Streaming ASR is supported ONLY for vLLM backend.
            - Streaming ASR does NOT support timestamps (forced aligner is not used).
            - Batch inference is NOT supported.

        Args:
            context:
                Context string.
            language:
                Optional forced language. If provided, it must be in supported languages.
                Same behavior as transcribe(): forces text-only output via prompt suffix.
            unfixed_chunk_num:
                For the first N chunks, do not use previous output as prefix prompt (reset prefix to "").
            unfixed_token_num:
                Roll back the last K tokens from accumulated output when using it as prefix prompt
                after unfixed_chunk_num.
            chunk_size_sec:
                Chunk size in seconds (audio is 16k PCM). The function will internally convert it
                to sample count at 16kHz.

        Returns:
            ASRStreamingState: Mutable state object to be passed to streaming_transcribe() and
            finish_streaming_transcribe().

        Raises:
            ValueError:
                - If backend is not "vllm".
                - If chunk_size_sec <= 0.
                - If forced language is invalid (same validation rules as transcribe()).
        """
        # 流式识别只支持 vLLM，因为这里的实现依赖它更适合在线反复调用的推理接口。
        if self.backend != "vllm":
            raise ValueError("Streaming ASR is supported only for vLLM backend (backend='vllm').")
        if chunk_size_sec is None or float(chunk_size_sec) <= 0:
            raise ValueError(f"chunk_size_sec must be > 0, got: {chunk_size_sec}")

        force_language = None
        if language is not None and str(language).strip() != "":
            ln = normalize_language_name(str(language))
            validate_language(ln)
            force_language = ln

        chunk_size_samples = int(round(float(chunk_size_sec) * SAMPLE_RATE))
        chunk_size_samples = max(1, chunk_size_samples)

        # ``prompt_raw`` is cached because the system/user prompt framing is
        # invariant across incremental streaming updates.
        prompt_raw = self._build_text_prompt(context=context, force_language=force_language)

        # 返回的是一个“初始化好的可变会话状态”，而不是立即开始识别。
        return ASRStreamingState(
            unfixed_chunk_num=int(unfixed_chunk_num),
            unfixed_token_num=int(unfixed_token_num),
            chunk_size_sec=float(chunk_size_sec),
            chunk_size_samples=int(chunk_size_samples),
            chunk_id=0,
            buffer=np.zeros((0,), dtype=np.float32),
            audio_accum=np.zeros((0,), dtype=np.float32),
            prompt_raw=prompt_raw,
            context=context or "",
            force_language=force_language,
            language="",
            text="",
            _raw_decoded="",
        )

    def streaming_transcribe(self, pcm16k: np.ndarray, state: ASRStreamingState) -> ASRStreamingState:
        """
        Streaming ASR decode step.

        This function accepts an arbitrary-length 16k PCM float numpy array (mono).
        It buffers incoming samples, and whenever enough samples are accumulated to form one
        full chunk (chunk_size_sec), it runs one incremental decode step and updates:

          - state.language
          - state.text

        The caller only needs to keep passing audio to this function and read state.language/state.text.

        Implementation details:
            - Each time a new chunk is ready, we append it to audio_accum and re-feed *all* audio seen
              so far to the model (no padding).
            - We update the prompt as: state.prompt_raw + prefix_text
            - Prefix rollback strategy:
                * If chunk_id < unfixed_chunk_num: prefix_text = ""
                * Else: rollback last unfixed_token_num tokens from previously accumulated decoded text.

        Notes:
            - vLLM backend only.
            - No timestamps.
            - Single stream only (no batching).

        Args:
            pcm16k:
                16kHz mono PCM waveform (np.ndarray). Length can be any non-negative integer.
                dtype can be float32/float64/int16; it will be converted to float32.
            state:
                Streaming state returned by init_streaming_state().

        Returns:
            ASRStreamingState: The same state object (mutated) for convenience.

        Raises:
            ValueError:
                If backend is not "vllm" or state is invalid.
        """
        # 这一版 streaming 并不是传统的“真正增量声学状态缓存”实现。
        # 它的思想更接近：
        # - 音频按固定 chunk 收集
        # - 每来一个新 chunk，就把“截至当前的全部音频”重新送去解码
        # - 用 prefix 回退策略减轻 chunk 边界抖动
        #
        # 这样做计算上更重，但实现简单，而且离线/在线两条路径更一致。
        if self.backend != "vllm":
            raise ValueError("streaming_transcribe() is supported only for vLLM backend (backend='vllm').")
        if state is None:
            raise ValueError("state must not be None. Call init_streaming_state() first.")
        if pcm16k is None:
            raise ValueError("pcm16k must not be None.")

        # Ensure 1D mono
        x = np.asarray(pcm16k)
        if x.ndim != 1:
            x = x.reshape(-1)

        # Convert to float32 PCM in [-1, 1] if int16 provided
        if x.dtype == np.int16:
            x = (x.astype(np.float32) / 32768.0)
        else:
            x = x.astype(np.float32, copy=False)

        # Append to buffer
        if x.shape[0] > 0:
            state.buffer = np.concatenate([state.buffer, x], axis=0)

        # Streaming is implemented as repeated full-context decoding over all
        # accumulated audio seen so far. This is computationally heavier than a
        # truly stateful speech decoder, but it is simple and keeps the model
        # output format identical to offline ASR.
        #
        # The prefix rollback heuristic prevents unstable trailing tokens from
        # being copied verbatim into the next prompt.
        # Consume full chunks
        while state.buffer.shape[0] >= state.chunk_size_samples:
            chunk = state.buffer[: state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples :]

            # Accumulate audio (re-feed from start, no padding)
            if state.audio_accum.shape[0] == 0:
                state.audio_accum = chunk
            else:
                state.audio_accum = np.concatenate([state.audio_accum, chunk], axis=0)

            # Build prefix with rollback strategy
            prefix = ""
            if state.chunk_id < state.unfixed_chunk_num:
                # 初期 chunk 通常不稳定，直接不继承旧文本，避免错误前缀放大。
                prefix = ""
            else:
                cur_ids = self.processor.tokenizer.encode(state._raw_decoded)
                k = int(state.unfixed_token_num)
                while True:
                    end_idx = max(0, len(cur_ids) - k)
                    prefix = self.processor.tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
                    # ``\ufffd`` is the Unicode replacement character. If it
                    # appears, trimming likely cut through a token/byte boundary,
                    # so we back off further until the prefix decodes cleanly.
                    if '\ufffd' not in prefix:
                        break
                    else:
                        if end_idx == 0:
                            prefix = ""
                            break
                        k += 1

            prompt = state.prompt_raw + prefix

            # vLLM expects a list of requests even for a single utterance.
            inp = {"prompt": prompt, "multi_modal_data": {"audio": [state.audio_accum]}}

            outputs = self.model.generate([inp], sampling_params=self.sampling_params, use_tqdm=False)
            gen_text = outputs[0].outputs[0].text

            # Keep the exact model text before structured parsing because future
            # rollback operates in tokenizer space over this raw string.
            # 这里保存 raw decoded 非常重要：
            # parser 可能会做清洗或格式解释，但下一轮 prefix 回退需要的是“原始模型文本”。
            state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text

            lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
            state.language = lang
            state.text = txt

            state.chunk_id += 1

        return state

    def finish_streaming_transcribe(self, state: ASRStreamingState) -> ASRStreamingState:
        """
        Finish streaming ASR.

        This function flushes the remaining buffered audio in state.buffer (tail audio).
        It sends the remaining samples to the model even if shorter than chunk_size_sec,
        without padding. Then it updates state.language/state.text one last time.

        Notes:
            - vLLM backend only.
            - No timestamps.
            - Single stream only.

        Args:
            state:
                Streaming state.

        Returns:
            ASRStreamingState: Updated state (mutated).

        Raises:
            ValueError:
                If backend is not "vllm" or state is invalid.
        """
        # 这个函数专门处理“最后一截不满 chunk 的尾音频”。
        # 如果不 flush，它就会永远留在 buffer 里，最终结果会少最后一小段内容。
        if self.backend != "vllm":
            raise ValueError("finish_streaming_transcribe() is supported only for vLLM backend (backend='vllm').")
        if state is None:
            raise ValueError("state must not be None.")

        # If no remaining buffer, still return state as-is.
        if state.buffer is None or state.buffer.shape[0] == 0:
            return state

        tail = state.buffer
        state.buffer = np.zeros((0,), dtype=np.float32)

        # Append tail to accumulated audio
        if state.audio_accum.shape[0] == 0:
            state.audio_accum = tail
        else:
            state.audio_accum = np.concatenate([state.audio_accum, tail], axis=0)

        # Reuse the same prompt-prefix strategy as the incremental path so the
        # final tail decode stays behaviorally aligned with previous chunks.
        prefix = ""
        if state.chunk_id < state.unfixed_chunk_num:
            prefix = ""
        else:
            cur_ids = self.processor.tokenizer.encode(state._raw_decoded)
            # Keep at least one token when possible; an entirely empty carry-over
            # tends to make the final tail decode less stable.
            # 中文学习备注：
            # 这里和中间 chunk 略有不同：尾段 flush 时尽量保留至少一个 token，
            # 否则最后一次解码有时会因为上下文过空而变得不稳定。
            end_idx = max(1, len(cur_ids) - int(state.unfixed_token_num))
            prefix = self.processor.tokenizer.decode(cur_ids[:end_idx])

        prompt = state.prompt_raw + prefix
        inp = {"prompt": prompt, "multi_modal_data": {"audio": [state.audio_accum]}}

        outputs = self.model.generate([inp], sampling_params=self.sampling_params, use_tqdm=False)
        gen_text = outputs[0].outputs[0].text

        state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text
        lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
        state.language = lang
        state.text = txt

        state.chunk_id += 1
        return state
