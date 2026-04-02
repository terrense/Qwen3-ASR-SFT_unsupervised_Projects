# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
Inference-only vLLM implementation of Qwen3-ASR.

This file adapts the same conceptual architecture used in the Transformers
backend to vLLM's execution model:

1. The audio tower is reimplemented with tensor-parallel vLLM layers.
2. Multimodal registration hooks teach vLLM how to parse, batch and inject
   speech features.
3. Prompt helpers expose the model to vLLM's speech-to-text serving interface.

Compared with the Transformers file, more code here is about runtime plumbing
than about model math, because vLLM requires explicit multimodal metadata and
weight-loading mappings.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config import MultiModalConfig, ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen2_5OmniAudioFeatureInputs,
    Qwen3OmniMoeThinkerMultiModalProcessor,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)
from vllm.model_executor.models.whisper import ISO639_1_SUPPORTED_LANGS
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    AudioItem,
    ModalityData,
    MultiModalDataDict,
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.model_executor.models.vision import (
    get_vit_attn_backend,
)
from ..transformers_backend.configuration_qwen3_asr import (
    Qwen3ASRConfig,
    Qwen3ASRThinkerConfig,
    Qwen3ASRAudioEncoderConfig
)
from ..transformers_backend.processing_qwen3_asr import (
    Qwen3ASRProcessor,
)

try:
    from vllm.multimodal.profiling import BaseDummyInputsBuilder
except:
    from vllm.multimodal.processing import BaseDummyInputsBuilder

logger = init_logger(__name__)

# 中文学习备注：这份文件建议按下面顺序阅读。
# 1. `_get_feat_extract_output_lengths`：长度公式要和 HF 版完全一致
# 2. `Qwen3ASRAudioEncoder`：看 vLLM 版音频塔如何复刻
# 3. `Qwen3ASRMultiModalProcessor`：看 prompt 占位符如何扩写
# 4. `Qwen3ASRForConditionalGeneration.embed_multimodal/embed_input_ids`
#    ：看音频 embedding 如何并进文本序列


def _get_feat_extract_output_lengths(input_lengths: torch.Tensor):
    """Mirror the Hugging Face processor's audio downsampling length formula."""
    # 中文学习备注：
    # vLLM 路径必须和 Transformers 路径共用同一套长度公式，
    # 否则 prompt 里展开的音频占位符数量会和 audio_tower 实际产出的 feature 数对不上。
    # 这里的 `input_lengths` 同样是 mel 特征长度，而不是原始 waveform 采样点数。
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    )
    return output_lengths


# ============= Audio Encoder Components =============


class SinusoidsPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for audio encoder."""
    # 中文学习备注：vLLM 版音频塔沿用同样的正余弦位置编码。

    def __init__(self, length: int, channels: int, max_timescale: int = 10000):
        super().__init__()
        self.length = length
        self.channels = channels
        self.max_timescale = max_timescale

        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")

        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(channels // 2).float()
        )
        scaled_time = (
            torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        )
        positional_embedding = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=1
        )
        self.register_buffer(
            "positional_embedding", positional_embedding, persistent=False
        )
        # 中文学习备注：这里同样使用 register_buffer，
        # 表示这是一块“跟着模型走、但不训练”的常量张量。

    def forward(self, seqlen: int) -> torch.Tensor:
        return self.positional_embedding[:seqlen, :]


class Qwen3ASRAudioAttention(nn.Module):
    """Multi-headed attention for Qwen3-Omni Audio Encoder using MMEncoderAttention."""
    # 中文学习备注：这里和 HF 版的差异主要在“执行器”：
    # HF 版用 ALL_ATTENTION_FUNCTIONS 分发，vLLM 版直接走 MMEncoderAttention。

    def __init__(
        self,
        config: Qwen3ASRAudioEncoderConfig,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        tp_size = get_tensor_model_parallel_world_size()
        self.num_local_heads = self.num_heads // tp_size

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scaling = self.head_dim**-0.5

        self.qkv = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_heads,
            bias=True,
            prefix=f"{prefix}.qkv",
        )
        # 中文学习备注：vLLM 把 q/k/v 三个线性层合并成一个并行层，
        # 这样更适合张量并行切分和高性能 kernel。

        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            bias=True,
            prefix=f"{prefix}.out_proj",
        )

        self.attn = MMEncoderAttention(
            num_heads=self.num_local_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            multimodal_config=multimodal_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor:
        seq_length, _ = hidden_states.size()
        qkv, _ = self.qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        # 形状说明：
        # hidden_states: (S_total, D_model)
        # qkv:           (S_total, 3 * D_model)
        # q/k/v:         (S_total, D_model)
        # vLLM's attention kernels expect (batch, seq, heads, head_dim). We use
        # a synthetic batch dimension of 1 because the sequences are already
        # packed with ``cu_seqlens``.
        # 中文学习备注：这里和 Transformers 版的本质完全一样，
        # 只是 attention 内核换成了 vLLM 的 MMEncoderAttention。
        q = q.view(1, seq_length, -1, self.head_dim)
        k = k.view(1, seq_length, -1, self.head_dim)
        v = v.view(1, seq_length, -1, self.head_dim)

        attn_output = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        attn_output = attn_output.view(seq_length, -1)
        output, _ = self.out_proj(attn_output)
        return output


class Qwen3ASRAudioEncoderLayer(nn.Module):
    """Transformer encoder layer for Qwen3-Omni Audio Encoder."""
    # 中文学习备注：一层音频 encoder block，结构仍然是 Attn + MLP + 残差。

    def __init__(
        self,
        config: Qwen3ASRAudioEncoderConfig,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen3ASRAudioAttention(
            config, multimodal_config=multimodal_config, prefix=f"{prefix}.self_attn"
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = _ACTIVATION_REGISTRY[config.activation_function]
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.encoder_ffn_dim,
            bias=True,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            config.encoder_ffn_dim,
            self.embed_dim,
            bias=True,
            prefix=f"{prefix}.fc2",
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor of shape (seq_len, hidden_size)
            cu_seqlens: Cumulative sequence lengths
            max_seqlen: Maximum sequence length in the batch
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 形状始终是 (S_total, D_model)，只是语义上从原始音频表示逐层变成更抽象的语义表示。
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        # ColumnParallelLinear / RowParallelLinear 是 vLLM 为张量并行准备的线性层，
        # 调用时通常会返回 `(output, bias_or_aux)`，所以这里会写成 `hidden_states, _ = ...`。
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        # Clamp for numerical stability with fp16
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        return hidden_states


class Qwen3ASRAudioEncoder(nn.Module):
    """vLLM-native Qwen3-ASR Audio Encoder."""

    def __init__(
        self,
        config: Qwen3ASRAudioEncoderConfig,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize
        # 中文学习备注：
        # 这一版是“vLLM 运行时重写版音频塔”，不是另一套新架构。
        # 如果你已经读过 Transformers 版，这里主要对照看：卷积、packed 序列、cu_seqlens、输出投影。

        # Position embedding
        self.positional_embedding = SinusoidsPositionEmbedding(
            self.max_source_positions, embed_dim
        )

        # Convolutional layers for mel-spectrogram processing
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            2,
            padding=1,
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            2,
            padding=1,
        )

        conv_out_dim = config.downsample_hidden_size * (
            (((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        )
        # 中文学习备注：这个公式和 HF 版同义，
        # 表示频率维经过 3 次 stride=2 后的剩余长度，再乘通道数得到 flatten 长度。
        self.conv_out = nn.Linear(conv_out_dim, config.d_model, bias=False)

        # Transformer encoder layers
        self.layers = nn.ModuleList(
            [
                Qwen3ASRAudioEncoderLayer(
                    config,
                    multimodal_config=multimodal_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(config.encoder_layers)
            ]
        )

        # Output layers
        self.ln_post = nn.LayerNorm(config.d_model)
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.act = _ACTIVATION_REGISTRY[config.activation_function]
        self.proj2 = nn.Linear(config.d_model, config.output_dim)

        # Get attention backend
        attn_backend_override = (
            multimodal_config.mm_encoder_attn_backend
            if multimodal_config is not None
            else None
        )
        self.attn_backend = get_vit_attn_backend(
            head_size=config.d_model // config.encoder_attention_heads,
            dtype=torch.get_default_dtype(),
            attn_backend_override=attn_backend_override,
        )

    def compute_attn_mask_seqlen(self, cu_seqlens: torch.Tensor) -> torch.Tensor | None:
        """Compute max_seqlen only for flash attention backends."""
        # 中文学习备注：某些 FA backend 需要显式给出当前 batch 里最大的局部段长。
        # 如果不是这些 backend，就返回 None，让下游忽略这个参数。
        max_seqlen = None
        if self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
        }:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        return max_seqlen

    @property
    def dtype(self) -> torch.dtype:
        # 代理属性：直接以第一层卷积权重的 dtype 作为整座音频塔的 dtype。
        return self.conv2d1.weight.dtype

    @property
    def device(self) -> torch.device:
        # 代理属性：默认认为整座塔都在 conv2d1 所在设备上。
        return self.conv2d1.weight.device

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
        aftercnn_lens: torch.Tensor,
    ):
        # 中文学习备注：这里复现的就是“长音频先切 chunk，再打包做局部 attention”的主流程。
        # 输入/输出主形状：
        # input_features:  (mel_bins, T_mel) 或与之兼容的单音频特征张量
        # output:          (S_total_after_cnn, output_dim)
        # Compute chunk information
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        # Split input features into chunks and pad
        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(
            chunk_list, batch_first=True
        ).transpose(1, 2)
        # 语法点：这里和 HF 版一样，是“先转置到时间优先，再按每块长度列表切块，再 pad 回 batch”。

        # Compute feature lengths after CNN
        feature_lens_after_cnn = self._get_cnn_output_lengths(chunk_lengths)
        # Vectorized mask creation: avoid creating many small tensors
        max_len_after_cnn = feature_lens_after_cnn.max().item()
        indices = torch.arange(max_len_after_cnn, device=padded_feature.device)
        padded_mask_after_cnn = indices.unsqueeze(0) < feature_lens_after_cnn.unsqueeze(
            1
        )

        # Add channel dimension for conv2d
        padded_feature = padded_feature.unsqueeze(1)
        # 形状从 (N_chunk, mel_bins, T) 变成 (N_chunk, 1, mel_bins, T)，以适配 Conv2d。

        # Apply convolutional layers (chunk if needed to avoid OOM)
        if padded_feature.size(0) <= self.conv_chunksize:
            # Fast path: no chunking needed
            padded_embed = F.gelu(self.conv2d1(padded_feature))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
        else:
            # Chunked processing to avoid OOM
            padded_embeds = []
            for chunk in padded_feature.split(self.conv_chunksize, dim=0):
                padded_embed = F.gelu(self.conv2d1(chunk))
                padded_embed = F.gelu(self.conv2d2(padded_embed))
                padded_embed = F.gelu(self.conv2d3(padded_embed))
                padded_embeds.append(padded_embed)
            padded_embed = torch.cat(padded_embeds, dim=0)

        # (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = padded_embed.size()
        # 中文学习备注：这里的 `permute(...).contiguous().view(...)` 是典型 PyTorch 形状重排套路：
        # - `permute` 只改维度顺序
        # - `contiguous` 确保内存连续
        # - `view` 才能安全 reshape
        padded_embed = self.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        )

        # Add positional embedding
        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding

        # Extract valid hidden states and compute cu_seqlens
        hidden_states = padded_embed[padded_mask_after_cnn]
        # 这一步会把 (N_chunk, T', D_model) 压缩成 (所有有效帧总数, D_model)。

        # Compute cumulative sequence lengths for chunked attention. These mark
        # the ragged boundaries after convolutional downsampling.
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (
            self.n_window_infer // (self.n_window * 2)
        )
        # Use tolist() for efficient batch conversion from tensor to Python
        for cnn_len in aftercnn_lens.tolist():
            num_full_chunks = cnn_len // window_aftercnn
            remainder = cnn_len % window_aftercnn
            cu_chunk_lens.extend([window_aftercnn] * num_full_chunks)
            if remainder:
                cu_chunk_lens.append(remainder)
        # 中文学习备注：`cu_seqlens` 和 Transformers 版同义，
        # 都是在 packed hidden states 上标记 ragged chunk 的边界。
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(
            -1, dtype=torch.int32
        )

        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)

        # Apply transformer layers
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens,
                max_seqlen,
            )

        # Apply output layers
        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)

        return hidden_states

    def _get_cnn_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute output lengths after the three conv2d layers."""
        # 中文学习备注：这是长度公式的“朴素循环版”，和上面那个闭式公式本质等价。
        lengths = input_lengths
        for _ in range(3):
            lengths = (lengths - 1) // 2 + 1
        return lengths

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with mapping from HuggingFace format."""
        # 中文学习备注：因为 vLLM 版把 q/k/v 合成了一个 qkv 参数，
        # 所以加载 HF 权重时需要做一次名字映射和分片装配。
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("self_attn.qkv.", "self_attn.q_proj.", "q"),
            ("self_attn.qkv.", "self_attn.k_proj.", "k"),
            ("self_attn.qkv.", "self_attn.v_proj.", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # `for ... else` 语法点：
                # 只有当上面的 for 循环没有被 `break` 命中时，才会进入这里。
                param = params_dict.get(name)
                if param is not None:
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3ASRProcessingInfo(BaseProcessingInfo):
    """Bridge vLLM's multimodal registry with the Hugging Face processor objects."""
    # 中文学习备注：这是 vLLM 多模态注册体系里的“信息提供者”。
    # 它告诉 vLLM：这个模型对应哪个 HF config、哪个 processor、支持什么模态。

    def get_hf_config(self):
        # 直接把最外层 Qwen3ASRConfig 里的 thinker_config 拿出来给 vLLM 用。
        return self.ctx.get_hf_config(Qwen3ASRConfig).thinker_config

    def get_hf_processor(self, **kwargs: object) -> Qwen3ASRProcessor:
        processor = self.ctx.get_hf_processor(
            Qwen3ASRProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )
        if not hasattr(processor, "audio_token"):
            # 某些 processor 版本里可能没有显式挂这个字段，所以这里做兜底。
            processor.audio_token = "<|audio_pad|>"
        return processor

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor
        # `assert isinstance(...)` 在这里既是运行时检查，也相当于给后续类型推断一个提示。
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}


class Qwen3ASRDummyInputsBuilder(BaseDummyInputsBuilder[Qwen3ASRProcessingInfo]):
    """Generate synthetic multimodal requests for profiling and shape inference."""
    # 中文学习备注：profiling / shape 推断时，vLLM 会先造假输入。
    # 这里决定“假的音频长什么样”。

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        hf_processor = self.info.get_hf_processor()
        audio_token = hf_processor.audio_token

        return audio_token * num_audios
        # 中文学习备注：这里故意返回重复的音频占位符字符串，
        # 让 profiling 阶段也能走到多模态 prompt 替换逻辑。

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)

        feature_extractor = self.info.get_feature_extractor()

        target_audio_length = (
            min(
                feature_extractor.chunk_length,
                30,
            )
            * feature_extractor.sampling_rate
        )
        # 中文学习备注：profiling 时不一定需要超长音频，取一个“够代表性但不太大”的长度即可。

        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=target_audio_length,
                num_audios=num_audios,
                overrides=audio_overrides,
            ),
        }


def _qwen3asr_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    """Describe how batched audio tensors are laid out inside vLLM multimodal inputs."""
    # 中文学习备注：这一步是在告诉 vLLM：
    # batched audio 特征、长度、mask 在内存里怎么排布。
    audio_feature_lengths = hf_inputs.get("audio_feature_lengths", torch.empty((0,)))
    return dict(
        input_audio_features=MultiModalFieldConfig.flat_from_sizes(
            "audio", audio_feature_lengths, dim=1
        ),
        feature_attention_mask=MultiModalFieldConfig.batched("audio"),
        audio_feature_lengths=MultiModalFieldConfig.batched("audio"),
    )


class Qwen3ASRMultiModalDataParser(MultiModalDataParser):
    """Accept either raw audio items or precomputed audio-feature dictionaries."""
    # 中文学习备注：既支持原始音频，也支持已经提取好的 audio features 直接喂进来。

    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[AudioItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            # 中文学习备注：如果上游已经给了现成的特征字典，就直接包装，不再重新提特征。
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"input_audio_features", "audio_feature_lengths"},
                fields_factory=_qwen3asr_field_config,
            )

        return super()._parse_audio_data(data)


class Qwen3ASRMultiModalProcessor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
):
    """Customize prompt replacement and feature bookkeeping for audio-only ASR."""
    # 中文学习备注：
    # 这个类是 vLLM 路径里最值得学的胶水层。
    # 因为“音频怎么从一个占位符变成 N 个 placeholder token”就是它决定的。

    def _get_data_parser(self) -> MultiModalDataParser:
        # 中文学习备注：把输入音频解析器固定成 Qwen3-ASR 自己这套。
        feature_extractor = self.info.get_feature_extractor()
        return Qwen3ASRMultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # 中文学习备注：把 HF processor 产出的字段布局翻译成 vLLM 能理解的 field config。
        return _qwen3asr_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        # 中文学习备注：
        # vLLM 路径里，“一个音频占位符到底要扩成多少个序列位置”就是在这里显式决定的。
        # 数量必须和 audio_tower 产出的 feature 数一模一样，否则后面 embedding 合并会报 shape 错。
        # 这一步非常关键，因为 vLLM 的文本 token 规划和 multimodal embedding 注入是分开的。
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        audio_token = processor.audio_token
        audio_token_id = vocab[audio_token]

        out_mm_data = out_mm_kwargs.get_data()
        audio_feature_lengths = out_mm_data.get("audio_feature_lengths")
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        if audio_feature_lengths is None and feature_attention_mask is None:
            audio_output_lengths = []
        elif audio_feature_lengths is not None:
            audio_output_lens = _get_feat_extract_output_lengths(audio_feature_lengths)
            audio_output_lengths = audio_output_lens.tolist()
        elif feature_attention_mask is not None:
            assert isinstance(feature_attention_mask, torch.Tensor)
            audio_output_lens = _get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )
            audio_output_lengths = audio_output_lens.tolist()
        # 中文学习备注：最终得到的是一个 Python list，
        # 每个元素表示“第 i 条音频要展开成多少个 placeholder token”。

        def get_replacement_qwen2_audio(item_idx: int):
            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio = audios.get(item_idx)
                raise ValueError(
                    f"The audio {audio} (len={len(audio)}) is too short "
                    "to be represented inside the model"
                )

            # The prompt token stream must reserve one placeholder token per
            # encoded audio feature vector so embedding replacement is shape-safe.
            # 中文学习备注：这里返回的不是最终识别结果，而是“给音频 embedding 预留坑位”的 token 序列。
            return [audio_token_id] * num_features

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_qwen2_audio,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3ASRMultiModalProcessor,
    info=Qwen3ASRProcessingInfo,
    dummy_inputs=Qwen3ASRDummyInputsBuilder,
)
class Qwen3ASRForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
    SupportsTranscription,
):
    supported_languages = ISO639_1_SUPPORTED_LANGS
    # 中文学习备注：
    # 这是 vLLM 世界里的顶层模型类。
    # 你可以把它理解成“HF 版 ThinkerForConditionalGeneration 的 vLLM 适配壳”。
    # 上面的 `@MULTIMODAL_REGISTRY.register_processor(...)` 是注册 decorator：
    # 它会把这个模型类和对应的 processor/info/dummy builder 绑定进 vLLM 的多模态注册表。

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "thinker.lm_head.": "language_model.lm_head.",
            "thinker.model.": "language_model.model.",
            "thinker.": "",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|audio_start|><|audio_pad|><|audio_end|>"
        # 中文学习备注：这里返回的是“一个逻辑音频块”的外层占位符模板，
        # 真正扩成多少个 `<|audio_pad|>` 位置要到 processor 那边才知道。

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # vLLM passes a prebuilt config bundle that already contains the Hugging
        # Face config, quantization settings and multimodal runtime options.
        self.vllm_config = vllm_config  # needed for torch compile forward context
        thinker_config: Qwen3ASRThinkerConfig = (
            vllm_config.model_config.hf_config.thinker_config
        )
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = thinker_config
        self.multimodal_config = multimodal_config

        # 中文学习备注：这里不是另起一套模型，而是把“音频塔 + Qwen3 文本模型”
        # 接进 vLLM 的多模态运行时。
        self.audio_tower = Qwen3ASRAudioEncoder(
            thinker_config.audio_config,
            multimodal_config=multimodal_config,
            prefix=maybe_prefix(prefix, "audio_tower"),
        )
        self.quant_config = quant_config

        self.language_model = Qwen3ForCausalLM(
            vllm_config=vllm_config.with_hf_config(
                thinker_config.text_config, architectures=["Qwen3ForCausalLM"]
            ),
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Qwen2_5OmniAudioFeatureInputs | None:
        """Normalize raw multimodal kwargs into vLLM's typed audio-input object."""
        # 中文学习备注：先把松散 kwargs 规整成 vLLM 内部标准的 audio 输入对象。
        input_audio_features = kwargs.pop("input_audio_features", None)
        audio_feature_lengths = kwargs.pop("audio_feature_lengths", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        if input_audio_features is None:
            return None

        return Qwen2_5OmniAudioFeatureInputs(
            type="audio_features",
            input_features=input_audio_features,
            audio_feature_lengths=audio_feature_lengths,
            feature_attention_mask=feature_attention_mask,
        )
        # 语法点：这里虽然写起来像 dict，但返回的是一个带类型约束的结构化对象。

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        # 中文学习备注：当前模型只关心 audio 模态，但仍然遵守 vLLM 的统一多模态入口格式。
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("input_audio_features")
                and "audio" not in mm_input_by_modality
            ):
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(
                    **kwargs
                )
        return mm_input_by_modality

    def _process_audio_input(
        self,
        audio_input: Qwen2_5OmniAudioFeatureInputs,
        audio_hashes: list[str] | None = None,
        cached_audio_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_features = audio_input["input_features"]
        audio_feature_lengths = audio_input["audio_feature_lengths"]

        # 中文学习备注：先算每条音频最终会产出多少个 audio features，
        # 再跑 audio_tower，最后按长度切回“每条音频一段 embedding 序列”。
        audio_output_lengths = _get_feat_extract_output_lengths(audio_feature_lengths)

        audio_features = self.audio_tower(
            input_features.to(self.audio_tower.dtype),
            feature_lens=audio_feature_lengths,
            aftercnn_lens=audio_output_lengths,
        )
        return audio_features.split(audio_output_lengths.tolist())
        # 中文学习备注：`.split(lengths)` 会把一条总 embedding 序列切回“每条音频自己的那一段”。

    def get_language_model(self) -> torch.nn.Module:
        # 中文学习备注：vLLM 上层有时只关心“文本模型部分”。
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        """
        Encode audio inputs into embedding sequences consumable by the language model.

        vLLM keeps text embedding and multimodal embedding as separate concepts
        until late in the pipeline, which allows more flexible batching.
        """
        # 中文学习备注：vLLM 的思路是“先单独把音频编码成 embeddings，
        # 再在更晚的阶段和文本 embedding 合并”。
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()
        # 语法点：这里故意用 tuple 累积，是为了贴合 vLLM 对 MultiModalEmbeddings 的接口约定。

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "audio":
                audio_embeddings = self._process_audio_input(multimodal_input)
                multimodal_embeddings += tuple(audio_embeddings)
        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        # 中文学习备注：如果当前请求没有多模态输入，这里就退化成纯文本 embedding 路径。

        # Merge audio embeddings into the placeholder token positions computed by
        # vLLM's multimodal planner.
        # 中文学习备注：这一步和 Transformers 版的 `masked_scatter` 是同一件事，
        # 只是这里换成了 vLLM 统一的多模态合并接口。
        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # 中文学习备注：真正的前向大多已经在 embedding 合并阶段准备好了；
        # 这里主要是把合成后的输入交给 Qwen3 文本模型。
        if intermediate_tensors is not None:
            # 中文学习备注：pipeline parallel / 分段执行时，中间张量优先级高于外部传入的 inputs_embeds。
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # 中文学习备注：标准语言模型头，把 hidden_states 投到词表 logits。
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # 中文学习备注：复用 AutoWeightsLoader，把 HF checkpoint 权重映射进 vLLM 模块树。
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["talker.", "code2wav."],
        )
        loaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        return loaded_weights

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        """
        Build multimodal RoPE positions for a prompt containing audio placeholders.

        The text decoder still expects a monotonic position sequence. Audio spans
        therefore occupy position ranges just like normal tokens, but their
        lengths come from the encoded audio feature counts rather than from the
        original placeholder token count.
        """
        seq_len = len(input_tokens)

        if not mm_features:
            # No audio features, just return linear positions
            llm_positions = (
                torch.arange(seq_len, dtype=torch.long).view(1, -1).expand(3, -1)
            )
            return llm_positions.clone(), 0
        # 中文学习备注：没有音频时，多模态位置就退化成普通从 0 到 S-1 的线性位置。

        llm_pos_ids_list: list[torch.Tensor] = []
        st = 0

        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            offset = mm_feature.mm_position.offset

            # Get audio feature length from mm_feature data
            audio_feature_length = mm_feature.data["audio_feature_lengths"].data
            if isinstance(audio_feature_length, torch.Tensor):
                audio_feature_length = audio_feature_length.item()
            audio_len = _get_feat_extract_output_lengths(
                torch.tensor(audio_feature_length)
            ).item()
            # 中文学习备注：虽然原始 prompt 里只是一个音频占位符片段，
            # 但位置编码必须按“展开后的 audio_len 个位置”来编号。

            # Text segment before audio (includes audio_start token)
            text_len = offset - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_positions = (
                torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1)
                + st_idx
            )
            # 中文学习备注：这里的 `st_idx` 保证位置编号在整个混合序列上单调递增。
            llm_pos_ids_list.append(text_positions)
            st_idx = st_idx + text_len

            # Audio token segment
            audio_positions = (
                torch.arange(audio_len, dtype=torch.long).view(1, -1).expand(3, -1)
                + st_idx
            )
            llm_pos_ids_list.append(audio_positions)

            st = offset + audio_len

        # Handle remaining text (includes audio_end and any trailing text)
        if st < seq_len:
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = seq_len - st
            final_text_positions = (
                torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1)
                + st_idx
            )
            llm_pos_ids_list.append(final_text_positions)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        if llm_positions.shape[1] != seq_len:
            raise RuntimeError("Position ids length mismatch with input ids length")

        mrope_position_delta = (llm_positions.max() + 1 - seq_len).item()
        # 这个 delta 表示“多模态展开后的位置上界”和“原始 token 长度”之间的偏差。
        return llm_positions, mrope_position_delta

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        # 中文学习备注：告诉 vLLM 哪部分是 language model，哪部分是 multimodal tower。
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            tower_model=["audio_tower."],
        )

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        # 中文学习备注：从 processor 里把采样率、最大音频长度这些服务层配置提出来。
        processor = cached_processor_from_config(model_config)
        feature_extractor: WhisperFeatureExtractor = processor.feature_extractor
        return SpeechToTextConfig(
            max_audio_clip_s=feature_extractor.chunk_length,
            sample_rate=feature_extractor.sampling_rate,
        )

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        model_config: ModelConfig,
        stt_config: SpeechToTextConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        """Get the generation prompt to be used for transcription requests."""
        tokenizer = cached_tokenizer_from_config(model_config)
        audio_placeholder = cls.get_placeholder_str("audio", 0)
        # 中文学习备注：
        # 服务侧 prompt 很薄，只负责搭起 user/audio/assistant 这层外壳；
        # 真正把音频展开成多个 placeholder，并替换成 embeddings，
        # 是前面的 multimodal processor 和 model 在做。

        if task_type not in ("transcribe", "translate"):
            raise ValueError(
                f"Unsupported task_type '{task_type}'. "
                "Supported task types are 'transcribe' and 'translate'."
            )
        full_lang_name_to = cls.supported_languages.get(to_language, to_language)
        # The prompt is intentionally minimal: user message with audio followed
        # by the assistant prefix. When ``to_language`` is set, the model is
        # nudged into the "text-only with known language" output format used by
        # the higher-level wrapper.
        if to_language is None:
            prompt = (
                f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            prompt = (
                f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
                f"<|im_start|>assistant\nlanguage {full_lang_name_to}<asr_text>"
            )

        prompt_token_ids = tokenizer.encode(prompt)
        prompt_dict = {
            "prompt_token_ids": prompt_token_ids,
            "multi_modal_data": {"audio": audio},
        }
        # 中文学习备注：返回值不是纯 token 列表，而是一个 prompt dict：
        # 文本 token 和多模态原始数据会被一起交给 vLLM 的 planner。
        return cast(PromptType, prompt_dict)
