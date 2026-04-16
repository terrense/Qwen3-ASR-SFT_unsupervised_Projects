# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
Processor glue code for Qwen3-ASR.

The processor is where text tokens and audio features are synchronized into one
multimodal prompt. Conceptually it performs two jobs:

1. Use the Whisper feature extractor to turn waveforms into frame features.
2. Expand each audio placeholder token in the text prompt so the language model
   sees exactly as many multimodal slots as the audio encoder will output.
"""

# 中文学习备注：
# `processor` 在 Hugging Face 体系里经常容易被低估，但对多模态模型来说它非常关键。
# 这里它不是单纯的“前处理工具”，而是在维护一个严格的长度契约：
#
# - 音频侧会产出多少个连续 feature 向量
# - 文本 prompt 里就必须预留多少个占位 token 位置
#
# 如果这两边数量不一致，后面模型里的 `masked_scatter` 就无法把 audio embeddings
# 填进文本 embedding 序列。因此本文件的本质任务是：
# “让文本侧 placeholder 计数，和音频侧 encoder 输出长度完全同步”。
from __future__ import annotations

from collections import deque
from collections.abc import Iterator
import re

import numpy as np

from transformers.audio_utils import AudioInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import TextInput

# 中文学习备注：这份文件的职责非常具体，可以概括为两件事：
# 1. 把原始音频 waveform 交给 WhisperFeatureExtractor，变成 `input_features`
# 2. 把文本 prompt 里的“一个音频占位符”扩成“和 audio encoder 输出长度完全一致的一串占位符”
#
# 只有这两件事都做对了，后面 `modeling_qwen3_asr.py` 里的 `masked_scatter`
# 才能把连续 audio embeddings 精确塞进文本序列。


class Qwen3ASRProcessorKwargs(ProcessingKwargs, total=False):
    """Default tokenizer/feature-extractor kwargs used by the processor."""
    # 中文学习备注：
    # 这是一个 TypedDict 风格的 kwargs 配置壳。
    # `total=False` 表示这些键不要求全部出现；缺失时会用 `_defaults` 里的默认值。
    # 有些 IDE / 静态检查器会把 TypedDict 类体里的普通赋值误判成“非法字段定义”，
    # 所以把 `_defaults` 挪到类外赋值，避免整块区域发红。
    pass


setattr(
    Qwen3ASRProcessorKwargs,
    "_defaults",
    {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "return_attention_mask": True,
        },
    },
)


def _get_feat_extract_output_lengths(input_lengths):
    """
    Compute encoder sequence lengths after the audio downsampling stack.

    The released checkpoints downsample along time before feeding the Transformer
    encoder. Matching this length exactly is important because the number of
    audio placeholder tokens inserted into the text prompt must equal the number
    of produced audio feature vectors.
    """
    # 中文学习备注：
    # 这里的 `input_lengths` 仍然是特征帧长度，不是原始波形采样点长度。
    # 输出 `output_lengths` 表示：
    #   一段音频经过 3 次 stride=2 下采样和长度对齐规则后，
    #   最终会产出多少个 audio feature 向量。
    #
    # 这和模型里的同名函数必须严格保持一致，否则：
    # - processor 以为要插 N 个 audio placeholder
    # - audio_tower 实际产出 M 个 feature
    # - 后面的 embedding 注入会直接 shape 对不上
    #
    # 所以虽然它只是一个长度公式，但它实际上是 processor 和 model 之间的“暗合同”。

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


class Qwen3ASRProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen3ASR processor.
    [`Qwen3ASRProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`], and [`Qwen2TokenizerFast`]. See the
    [`~Qwen3ASRProcessor.__call__`] and [`~Qwen3ASRProcessor.decode`] for more information.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The audio feature extractor.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The text tokenizer.
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided, the default chat template is used.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    # 中文学习备注：
    # `ProcessorMixin` 是 HF 的“组合处理器”基类。
    # 它允许把 feature_extractor 和 tokenizer 绑成一个统一入口，
    # 这样用户只需要 `processor(text=..., audio=...)` 就能同时处理文本和音频。
    #
    # 对 Qwen3-ASR 来说，这种组合尤其重要，因为文本和音频不是彼此独立预处理，
    # 而是要在“占位符长度”这个点上严格对齐。

    def __init__(
        self, feature_extractor=None, tokenizer=None, chat_template=None
    ):
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)
        # These special tokens are defined by the tokenizer config and are later
        # used to mark where audio embeddings should be injected.
        # 中文学习备注：
        # 这三个 token 的语义是：
        # - `audio_token`：真正会被重复展开、并最终被 audio embeddings 覆盖的核心占位符
        # - `audio_bos_token` / `audio_eos_token`：音频片段边界标记
        self.audio_token = self.tokenizer.audio_token
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_eos_token = self.tokenizer.audio_eos_token

    def __call__(
        self,
        text: TextInput | list[TextInput] | None = None,
        audio: AudioInput | None = None,
        **kwargs: Unpack[Qwen3ASRProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audio` and `kwargs` arguments to
        WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] if `audio` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audio (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audio to be prepared. Each audio can be a NumPy array.
        """
        # 中文学习备注：
        # 这个 `__call__` 是整个 processor 最重要的入口。
        # 最终返回的 BatchFeature 里，既有 tokenizer 产出的：
        # - input_ids
        # - attention_mask
        # 也有 feature extractor 产出的：
        # - input_features
        # - feature_attention_mask
        #
        # 从调用链上看，这个函数的输出会直接进入：
        # `Qwen3ASRForConditionalGeneration.generate(**inputs)`。

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        output_kwargs = self._merge_kwargs(
            Qwen3ASRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) or any(not isinstance(sample, str) for sample in text):
            raise ValueError("Invalid input text. Please provide a string or a list of strings.")

        if audio is not None:
            output_kwargs["audio_kwargs"]["padding"] = True
            output_kwargs["audio_kwargs"]["truncation"] = False
            # 中文学习备注：
            # WhisperFeatureExtractor 这里会把原始 waveform -> log-mel features。
            # 对 batch 输入时，输出主形状一般是：
            # input_features:         (B, mel_bins, T_mel)
            # attention_mask:         (B, T_mel)
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            audio_inputs["feature_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )  # rename feature_attention_mask to prevent conflicts later on
            audio_inputs["input_features"] = audio_inputs.pop(
                "input_features"
            )  # rename input_features to prevent conflicts later on
            # ``audio_lengths`` is consumed when replacing each audio placeholder
            # with the correct number of repeated audio tokens.
            # 中文学习备注：
            # 这里得到的不是 mel 帧长度，而是“audio encoder 输出长度”。
            # 后面文本占位符扩写必须按这个长度来。
            audio_length_values = np.asarray(
                _get_feat_extract_output_lengths(audio_inputs["feature_attention_mask"].sum(-1))
            ).reshape(-1)
            # 这里把长度转成 deque，而不是普通 list，是因为后面的文本替换过程
            # 会按照 prompt 中 placeholder 出现的先后顺序逐个消费这些长度值。
            audio_lengths = deque(int(length) for length in audio_length_values.tolist())
        else:
            audio_inputs = {}
            audio_lengths = deque()
            # 中文学习备注：即便没传音频，也统一走同一条代码路径，只是长度迭代器为空。

        text = self.replace_multimodal_special_tokens(text, audio_lengths)
        # 中文学习备注：
        # 这一步非常关键：它会把 prompt 里“逻辑上的一个音频 token”
        # 替换成“重复 N 次的 audio_token”，其中 N = audio encoder 输出长度。
        #
        # 从“抽象 API”角度看，调用者只写了一个音频占位符；
        # 从“底层张量对齐”角度看，模型真正需要的是一整段连续占位位置。

        if audio_lengths:
            raise ValueError("Received more audio inputs than audio placeholder tokens in `text`.")

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        # tokenizer 输出后，文本序列长度已经包含了展开后的音频占位符位置。

        return BatchFeature(
            data={**texts_inputs, **audio_inputs},
            tensor_type=return_tensors,
        )
        # 语法点：
        # `{**a, **b}` 是 Python 字典解包合并写法。
        # 如果 key 重名，后面的字典会覆盖前面的值。

    def replace_multimodal_special_tokens(
        self,
        text: list[str],
        audio_lengths: deque[int],
    ) -> list[str]:
        """
        Expand each logical audio token into the exact number of encoder slots.

        AI note:
            The language model does not consume raw waveforms directly. Instead,
            the audio encoder produces a sequence of embeddings, and the prompt
            must reserve one token position for each of those embeddings.

        Python note:
            ``audio_lengths`` is an iterator, not a list, because the replacement
            lengths are consumed in prompt order as we walk the samples.
        """
        # 中文学习备注：
        # 输入：
        # - text:          List[str]
        # - audio_lengths: iterator，每次 `next(audio_lengths)` 取到一条音频该展开的长度
        #
        # 输出：
        # - processed_text: List[str]
        #   其中每个 `<audio_token>` 都已经被替换成重复 N 次的 audio_token 串
        #
        # 这个函数其实就是在做“逻辑音频片段 -> 真实序列槽位”的展开。

        processed_text: list[str] = []
        for sample in text:
            positions: list[tuple[int, str]] = []
            special_tokens = [re.escape(tok) for tok in [self.audio_token]]
            pattern = "|".join(special_tokens)
            positions = sorted([(match.start(), match.group()) for match in re.finditer(pattern, sample)])
            positions.sort(key=lambda x: x[0])
            # 中文学习备注：
            # 这里先用正则把所有音频特殊 token 的位置找出来。
            # `re.escape(tok)` 的作用是防止 token 本身包含会被正则解释的特殊字符。
            #
            # 语法点：
            # `[(match.start(), match.group()) for match in re.finditer(...)]`
            # 是列表推导式，会生成 `(位置, 匹配到的字符串)` 列表。

            for _, special_token in positions:
                if special_token == self.audio_token:
                    # 中文学习备注：
                    # 这里不是简单替换成一个别的 token，而是替换成：
                    # `<|audio_placeholder|>` * N
                    # 也就是把一个逻辑音频占位符扩成 N 个真实序列位置。
                    if not audio_lengths:
                        raise ValueError("Found more audio placeholder tokens in `text` than provided audio inputs.")
                    sample = sample.replace(self.audio_token, "<|audio_placeholder|>" * audio_lengths.popleft(), 1)
                    # 语法点：
                    # - `audio_lengths.popleft()` 从队列左侧取出“下一条音频该扩成多少个位置”
                    # - `replace(..., 1)` 只替换第一个命中的 token，避免一次性全替换

            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            # 中文学习备注：
            # 这里先用临时字符串占位，再统一替回真正的 audio_token，
            # 是为了避免在重复替换时相互影响。
            #
            # 否则一边 replace 一边又产生新的 `self.audio_token`，很容易把刚替换出来的内容再次命中。
            processed_text.append(sample)
        return processed_text

    def get_chunked_index(self, token_indices: np.ndarray, tokens_per_chunk: int) -> list[tuple[int, int]]:
        """
        Splits token index list into chunks based on token value ranges.

        Given a list of token indices, returns a list of (start, end) index tuples representing
        slices of the list where the token values fall within successive ranges of `t_ntoken_per_chunk`.

        For example, if `t_ntoken_per_chunk` is 1000, the function will create chunks such that:
        - the first chunk contains token values < 1000,
        - the second chunk contains values >= 1000 and < 2000, and so on.

        Parameters:
            token_indices (`np.ndarray`): A monotonically increasing list of token index values.
            t_ntoken_per_chunk (`int`): Number of tokens per chunk (used as the chunk size threshold).

        Returns:
            `list[tuple[int, int]]`: A list of tuples, each representing the start (inclusive)
                                and end (exclusive) indices of a chunk in `token_indices`.
        """
        # 中文学习备注：
        # 这是一个工具函数，把一串单调递增的 token 位置索引按固定块大小切段。
        # 它更偏“后处理/辅助切块逻辑”，不是 audio encoder 主数学的一部分。

        def _iter() -> Iterator[tuple[int, int]]:
            i, start_idx = 0, 0  # skip bos token
            current_chunk = 1
            while i < len(token_indices):  # skip eos token
                if token_indices[i] >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    # `yield` 说明这里是 generator，而不是一次性构造完整列表。
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(token_indices))

        return list(_iter())

    def apply_chat_template(self, conversations, chat_template: str | None = None, **kwargs):
        # 中文学习备注：这里没有自定义额外逻辑，直接复用父类实现。
        return super().apply_chat_template(conversations, chat_template, **kwargs)

    @property
    def model_input_names(self) -> list[str]:
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        # 中文学习备注：
        # 这里把 tokenizer 和 feature extractor 两边的输入字段合并起来，
        # 再额外补上 `feature_attention_mask`。
        # 这决定了上游 pipeline / Trainer / generate 调用在收集模型输入时，
        # 会把哪些字段当作这个 processor/model 组合的标准输入集合。
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + ["feature_attention_mask"]
            )
        )
        # 语法点：
        # `dict.fromkeys(seq)` 可以借助 dict 的“保序且去重”特性做去重，
        # 再转回 list，得到一个按出现顺序去重后的字段名列表。


__all__ = ["Qwen3ASRProcessor"]
