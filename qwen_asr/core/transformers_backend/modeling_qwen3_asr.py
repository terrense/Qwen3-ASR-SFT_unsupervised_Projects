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
Transformers implementation of the Qwen3-ASR architecture.

At a high level the model is a multimodal sequence transducer:

1. A convolutional + Transformer audio encoder compresses log-mel frames into a
   shorter sequence of speech embeddings.
2. Audio placeholder tokens inside the prompt are replaced with those speech
   embeddings.
3. A causal text decoder ("thinker") autoregressively emits the recognition
   result and optional metadata such as language tags.

The file is large because it contains both the audio tower and the text-side
generation stack, plus the glue logic that merges them for Hugging Face
generation APIs.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    MoeCausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import TransformersKwargs, check_model_inputs

from .configuration_qwen3_asr import (
    Qwen3ASRAudioEncoderConfig,
    Qwen3ASRConfig,
    Qwen3ASRThinkerConfig,
)

# 中文学习备注：这份文件可以粗分成四段来读。
# 1. 通用文本侧积木：RMSNorm / attention / MLP / decoder layer
# 2. 通用多模态辅助函数：长度公式、RoPE 位置、causal mask 等
# 3. 音频塔：Qwen3ASRAudioEncoder 及其 attention/layer
# 4. Thinker 与顶层 wrapper：把 audio features 注入文本序列并做生成
#
# 如果你第一次学，建议按下面顺序：
# `_get_feat_extract_output_lengths`
# -> `Qwen3ASRAudioEncoder.forward`
# -> `Qwen3ASRThinkerForConditionalGeneration.get_audio_features`
# -> `get_placeholder_mask`
# -> `masked_scatter`


@use_kernel_forward_from_hub("RMSNorm")
class Qwen3ASRTextRMSNorm(nn.Module):
    # 中文学习备注：这是文本侧用的 RMSNorm。
    # 它和 LayerNorm 的区别是只按方差归一化，不做均值中心化。
    # `@use_kernel_forward_from_hub("RMSNorm")` 是 Hugging Face 的一个优化钩子：
    # 如果运行环境里有更快的同名 kernel，可以在不改调用代码的前提下替换 forward 实现。
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3ASRTextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 形状说明：
        # hidden_states: (..., hidden_size)
        # variance:      (..., 1)
        # return:        (..., hidden_size)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # RMSNorm 核心公式：
        # y = x / sqrt(mean(x^2) + eps) * weight
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # 中文学习备注：RoPE 在实数张量里可以理解成“半边互换 + 一侧取负”。
    # RoPE can be written as a complex-number rotation. In real tensor form that
    # becomes "swap the two half-vectors and negate one side".
    # 形状说明：
    # x: (..., head_dim)
    # x1/x2: (..., head_dim/2)
    # return: (..., head_dim)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # ``expand`` avoids materializing copies immediately; the reshape then gives
    # the logical view expected by grouped-query attention.
    # 中文学习备注：GQA/MQA 下，K/V 头数可能小于 Q 头数；
    # 这里负责把较少的 K/V 头“逻辑上复制”成 attention 所需的头数。
    # 语法点：
    # `[:, :, None, :, :]` 会在第 3 维插入一个长度为 1 的新轴；
    # `expand` 不会真的复制数据，而是返回一个带 broadcast 语义的 view。
    # 形状变化：
    # (B, H_kv, S, D)
    # -> (B, H_kv, 1, S, D)
    # -> (B, H_kv, n_rep, S, D)
    # -> (B, H_kv * n_rep, S, D)
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    """
    Plain PyTorch attention path used when no fused backend is selected.

    This function mirrors the tensor contract expected by Hugging Face's
    attention dispatch utilities, which allows the same module code to work with
    eager attention, SDPA and FlashAttention backends.
    """
    # 中文学习备注：这是最朴素的 PyTorch attention 实现。
    # 它的价值主要不是快，而是给所有后端提供一个统一的“保底语义”。
    # 输入形状约定：
    # query: (B, H_q, S_q, D)
    # key:   (B, H_kv, S_k, D)
    # value: (B, H_kv, S_k, D)
    # 输出：
    # attn_output:  (B, S_q, H_q, D) 在 transpose 后再交给上层 reshape
    # attn_weights: (B, H_q, S_q, S_k)
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # 注意力核心公式：
    # softmax(Q K^T / sqrt(D) + mask) V
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # 中文学习备注：RoPE 只作用在 Q/K 上，不作用在 V 上。
    # 形状说明：
    # q, k:   (B, H, S, D) 或其他与 `unsqueeze_dim` 匹配的排列
    # cos/sin 原始通常是 (B, S, D) 或 (S, D)
    # 通过 unsqueeze 后变成可 broadcast 到 q/k 的形状
    #
    # 数学上相当于把每两个维度看成一个复数平面，然后按位置相关的角度做旋转。
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3ASRTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # 中文学习备注：这是文本 decoder 侧的 attention，不是音频 encoder 侧的 attention。
    # 文本侧是 causal attention，要保证生成时只能看见当前位置之前的 token。

    def __init__(self, config: Qwen3ASRConfig, layer_idx: int):
        super().__init__()
        # 语法点：`getattr(config, "head_dim", fallback)` 表示如果 config 里没有 head_dim，
        # 就退回到 hidden_size // num_attention_heads 这个默认计算值。
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3ASRTextRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3ASRTextRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 语法点：
        # `tuple[torch.Tensor, torch.Tensor]` 是 Python 3.9+ 原生泛型写法；
        # `Optional[Cache]` 等价于 `Cache | None`。
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Shape flow:
        #   hidden_states              -> (batch, seq, hidden)
        #   linear + view             -> (batch, seq, heads, head_dim)
        #   transpose for attention   -> (batch, heads, seq, head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # 中文学习备注：这里完成的是
        # hidden -> Q/K/V -> 加 RoPE -> 走 attention backend -> 再映射回 hidden。
        # 细一点写就是：
        # (B, S, H_model)
        # -> 线性投影到 (B, S, H_heads * D_head)
        # -> view 成 (B, S, H_heads, D_head)
        # -> transpose 成 (B, H_heads, S, D_head)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            # 中文学习备注：生成阶段不会每一步都重新算全部历史 K/V；
            # 这里把本步的 K/V 追加进缓存，后面 attention 就可以直接复用。
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        # 语法点：
        # 这里把函数当对象来传递/调用。`attention_interface(...)` 可能指向 eager、SDPA 或 FlashAttention。

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # Collapse the head dimension back into the model hidden size.
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3ASRTextMLP(nn.Module):
    # 中文学习备注：标准的 gated MLP。
    # 可以把它看成 decoder block 里的“前馈子层”。
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 形状说明：
        # x:         (B, S, H)
        # gate/up:   (B, S, I)
        # act(gate) * up: (B, S, I)
        # down_proj: (B, S, H)
        # 数学上是 gated MLP：Down(Act(Gate(x)) * Up(x))
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3ASRThinkerTextDecoderLayer(GradientCheckpointingLayer):
    # 中文学习备注：一个标准 decoder layer = RMSNorm + Self-Attn + 残差 + RMSNorm + MLP + 残差。
    # `GradientCheckpointingLayer` 的含义是：训练时可以按层重算前向，换显存省用量。
    def __init__(self, config: Qwen3ASRConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3ASRTextAttention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3ASRTextMLP(config)
        self.input_layernorm = Qwen3ASRTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3ASRTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        # 中文学习备注：残差连接保证 attention 子层和 MLP 子层都只做“增量修正”。
        # 这也是深层 Transformer 能稳定训练的核心结构之一。
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Qwen3ASRPreTrainedModel(PreTrainedModel):
    # 中文学习备注：所有 HF 版模块的共同父类。
    # 主要作用是把 config、attention backend 支持、gradient checkpointing 等通用能力收口。
    # 语法点：类属性如 `_supports_flash_attn = True` 会被 HF 框架读取，
    # 不是普通业务字段。
    config: Qwen3ASRConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "attentions": Qwen3ASRTextAttention,
    }


@dataclass
class Qwen3ASRThinkerCausalLMOutputWithPast(MoeCausalLMOutputWithPast):
    r"""
    Args:
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    rope_deltas: Optional[torch.LongTensor] = None
    # 中文学习备注：`@dataclass` 会自动生成初始化器和字段管理逻辑，
    # 这里相当于是在 HF 的标准输出对象上再额外挂一个 `rope_deltas` 字段。


def _get_feat_extract_output_lengths(input_lengths):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """
    # 中文学习备注：
    # `input_lengths` 是 mel 特征长度，不是原始采样点数。
    # 这个公式等价于 3 层 stride=2 卷积之后还剩多少个时间步，
    # 也就决定了一段音频最终会占多少个 audio placeholder 位置。
    # 你可以近似把它记成：
    # 长度经过 3 次 `L -> floor((L - 1) / 2) + 1`
    # 但这里特意拆成 `% 100` 和 `// 100` 两部分，是为了和前处理/切窗的长度约定严格对齐。

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


class Qwen3ASRPreTrainedModelForConditionalGeneration(Qwen3ASRPreTrainedModel):
    def _prepare_4d_causal_attention_mask_with_cache_position(
        self,
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        min_dtype: float,
        cache_position: torch.Tensor,
        batch_size: int,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            min_dtype (`float`):
                The minimum value representable with the dtype `dtype`.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        # 中文学习备注：当后端需要显式 4D causal mask 时，这个函数负责构造它。
        # 如果用的是 FlashAttention/SDPA，一部分逻辑会被更底层的 kernel 接管。
        # 形状流：
        # 2D attention_mask: (B, K)
        # -> 先构造 (Q, K_target) 的上三角 causal mask
        # -> 再扩成 (B, 1, Q, K_target)
        # -> 再把 padding 区域也置成最小值
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


    def get_chunked_index(
        self, token_indices: torch.Tensor, tokens_per_chunk: int, remove_index: int
    ) -> list[tuple[int, int]]:
        """
        Splits token index list into chunks based on token value ranges.

        Given a list of token indices, returns a list of (start, end) index tuples representing
        slices of the list where the token values fall within successive ranges of `t_ntoken_per_chunk`.

        For example, if `t_ntoken_per_chunk` is 1000, the function will create chunks such that:
        - the first chunk contains token values < 1000,
        - the second chunk contains values >= 1000 and < 2000, and so on.

        Parameters:
            token_indices (`torch.Tensor` of shape `(seq_len, )`): A monotonically increasing list of
                                token index values.
            t_ntoken_per_chunk (`int`): Number of tokens per chunk (used as the chunk size threshold).
            remove_index (`int`) An index id to subtract from `token_indices` before chunking

        Returns:
            `list[tuple[int, int]]`: A list of tuples, each representing the start (inclusive)
                                and end (exclusive) indices of a chunk in `token_indices`.
        """
        # 中文学习备注：这个函数把一串单调递增的位置索引按固定大小切块。
        # 它更像一个通用工具，不是音频塔数学本身。

        def _iter():
            i, start_idx = 0, 0  # skip bos token
            current_chunk = 1
            while i < len(token_indices):  # skip eos token
                if token_indices[i] - remove_index >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    # 中文学习备注：`yield` 说明这是一个 generator。
                    # `list(_iter())` 会把它一次性消费成列表。
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(token_indices))

        return list(_iter())

    def get_rope_index(
        self,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the rope index in LLM.

        Explanation:
            Each embedding sequence contains text embedding.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            audio_seqlens (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        mrope_position_deltas = []
        # 中文学习备注：Qwen3-ASR 的位置编码逻辑不是“单纯从 0 到 N-1”。
        # 这里会结合 attention_mask 计算真正有效 token 的位置，并为多模态 RoPE 预留偏移量。
        # 形状说明：
        # attention_mask:        (B, S)
        # cumsum 后 position:   (B, S)
        # expand 后 position:   (3, B, S)
        # mrope_position_delta:  (B, 1)

        position_ids = attention_mask.float().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

        return position_ids, mrope_position_deltas


class Qwen3ASRAudioAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # 中文学习备注：这是音频 encoder 侧 attention。
    # 和文本侧不同，它不是 causal 的，而是块内双向注意力。

    def __init__(self, config):
        super().__init__()
        # 音频 encoder 这里 `num_key_value_groups = 1`，
        # 因为它本质上是普通多头注意力，不需要文本侧那种 GQA 复用 K/V。
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.config = config

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_decoder = False
        self.is_causal = False
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # 中文学习备注：
        # 这里的 `hidden_states` 已经不是传统 `(batch, seq, hidden)`，
        # 而是把所有有效 chunk 打包后的 `(total_valid_frames, hidden)`。
        # `cu_seqlens` 负责告诉注意力：每个 chunk 在这条长序列里的边界在哪里。

        seq_length, _ = hidden_states.size()
        # 这里的 `-1` 最终会被推断成 `head_dim`。
        # 因为 embed_dim = num_heads * head_dim。

        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        # 形状变化：
        # (S_total, H, D) -> (H, S_total, D) -> (1, H, S_total, D)
        # 前面的 `1` 是伪 batch 维，因为真正的 ragged 批边界靠 cu_seqlens 表示。
        # FlashAttention 的 varlen 路径除了要知道 ragged 边界，还要知道最大的局部段长。
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            cu_seq_lens_q=cu_seqlens,  # pass cu seq lens for FA2
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **kwargs,
        )

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output


class Qwen3ASRAudioEncoderLayer(GradientCheckpointingLayer):
    # 中文学习备注：音频塔里的一个 encoder block。
    # 结构上更像标准 Transformer encoder layer，而不是 decoder layer。
    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen3ASRAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 形状始终保持：(S_total, D_model)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        return outputs


class SinusoidsPositionEmbedding(nn.Module):
    # 中文学习备注：音频塔这里用的是经典正余弦位置编码，不是文本侧那套 RoPE 实现。
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        # 中文学习备注：`register_buffer` 注册的是“随模型一起搬设备/存 checkpoint，
        # 但不参与训练”的张量，适合位置编码、均值方差统计这类常量。
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        # 返回形状：(seqlen, channels)
        return self.positional_embedding[:seqlen, :]


@auto_docstring(
    custom_intro="""
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`Qwen3ASRAudioEncoderLayer`].
    """
)
class Qwen3ASRAudioEncoder(Qwen3ASRPreTrainedModel):
    config: Qwen3ASRAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["Qwen3ASRAudioEncoderLayer"]
    _supports_sdpa = True

    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        # 中文学习备注：
        # 这就是音频塔的主骨架：
        # 3 层 Conv2d 先做时间下采样，再用 Transformer 编码，最后投影成可注入文本模型的 audio features。
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.layers = nn.ModuleList([Qwen3ASRAudioEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
        )
        # 中文学习备注：
        # 这里这个长公式本质上是在算“频率维经过 3 次 stride=2 后还剩多少格”，
        # 然后乘上通道数 downsample_hidden_size，得到每个时间步 flatten 后的向量长度。
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(config.d_model, config.output_dim)
        self.n_window_infer = self.config.n_window_infer
        self.conv_chunksize = self.config.conv_chunksize
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        # 中文学习备注：把整座音频塔冻结，常用于只想训文本侧或上层适配器时。
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        # 中文学习备注：
        # 音频塔没有词表 embedding，这里按 HF 约定返回“最靠前的输入映射层”。
        # 对本模型来说，就是第一层卷积 conv2d1。
        return self.conv2d1

    def set_input_embeddings(self, value: nn.Module):
        # 中文学习备注：与 `get_input_embeddings` 对应，外部如果要替换输入映射层，
        # 实际替换的是第一层卷积。
        self.conv2d1 = value

    def _prepare_attention_mask(self, inputs_tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        # Flash Attention 2 doesn't need a 4D mask and relies on `cu_seqlens/max_seqlen`
        # NOTE: the created attention masl only approximates the ragged FA2 attention by
        # allowing bidirectional attention within `cu_seqlens` blocks, and not attending between
        # blocks. Though it will not be a 100% match for FA2's `varlen` path
        # 中文学习备注：非 FA2 后端没有原生 ragged attention，只能用块对角 mask 去近似它。
        if self.config._attn_implementation == "flash_attention_2":
            return None

        seq_length = inputs_tensor.shape[0]
        # 最终 mask 形状：(1, 1, S_total, S_total)
        attention_mask = torch.full(
            [1, 1, seq_length, seq_length],
            torch.finfo(inputs_tensor.dtype).min,
            device=inputs_tensor.device,
            dtype=inputs_tensor.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        return attention_mask

    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: Optional[torch.LongTensor] = None,
        aftercnn_lens: Optional[torch.LongTensor] = None,
    ) -> BaseModelOutput:
        r"""
        feature_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length
        aftercnn_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length after cnn
        """
        # 中文学习备注：
        # 同一个模型既能吃短音频也能吃长音频，关键不在“换一套网络”，
        # 而在这里先把 mel 特征按固定窗口切块，再把卷积后的有效帧 packed 到一条长序列里。
        # 输入/输出主形状：
        # input_features:  (mel_bins, T_mel) 对单条样本来说是二维
        # chunk_list:      若干个 (chunk_T, mel_bins) 小块
        # padded_feature:  (N_chunk, mel_bins, max_chunk_T)
        # conv 输出:       (N_chunk, C, F', T')
        # packed hidden:   (S_total_after_cnn, D_model)
        # return.last_hidden_state: (S_total_after_cnn, output_dim)
        # The encoder processes long audio as several fixed-size windows. This
        # keeps convolution and attention memory bounded while still letting us
        # concatenate the resulting hidden states into one long sequence.
        if feature_lens is None:
            raise ValueError("`feature_lens` must be provided for the audio encoder forward pass.")
        if aftercnn_lens is None:
            aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        # 每条音频会被拆成多少个基础 chunk。
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        # ``chunk_lengths`` describes how the long mel sequence is partitioned
        # into convolution-friendly windows. Every original sample may contribute
        # several windows plus one shorter tail.
        # 中文学习备注：除尾块外，绝大多数 chunk 的 mel 长度都是固定的 `n_window * 2`。
        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        # 中文学习备注：
        # `tail_chunk_index` 的作用是定位“每条音频最后一个 chunk”在全局 chunk 列表中的位置。
        # 这样就能把尾块长度改成真实余数，而不是默认满块长度。
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2
        # 如果刚好整除，余数会是 0，这里要把它恢复成满块长度。

        # ``input_features`` arrives as (mel_bins, time). Splitting along the
        # transposed time axis yields a Python list of variable-length windows.
        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        # 语法点：
        # `input_features.T` 是转置，把时间维放到前面，便于按时间切块；
        # `.split(chunk_lengths.tolist(), dim=0)` 会按“每块长度列表”切分，而不是固定等长切分。
        padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        # 中文学习备注：这个 mask 用来从 padding 后的 batch 中挑出卷积后的真实时间步，
        # 后面会把它们重新打包成 packed hidden states。
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)
        # 形状此时变成 (N_chunk, 1, mel_bins, max_chunk_T)，
        # 可以直接喂给 Conv2d，其中 1 是输入通道数。
        # Convolution is the main activation-memory hotspot for long speech.
        # Running it chunk-by-chunk avoids OOM without changing the resulting
        # sequence because windows were already split above.
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            # 中文学习备注：
            # 这里的 `.split(self.conv_chunksize, dim=0)` 不是时间切块，
            # 而是把“chunk batch”再按 batch 维分成若干小批，纯粹为了省显存。
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        # After the 2D conv stack we treat each time step's (channel, freq)
        # patch as one vector and project it into the Transformer width.
        # 形状变化：
        # (B_chunk, C, F', T')
        # -> permute 成 (B_chunk, T', C, F')
        # -> view 成 (B_chunk, T', C*F')
        # -> Linear 到 (B_chunk, T', D_model)
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        # 中文学习备注：这里的位置编码是“每个 chunk 内独立按时间位置加”的。
        # 因为后面真正控制 chunk 边界语义的是 packed 序列和 cu_seqlens。
        # ``padded_embed`` is batch-major, but varlen attention prefers a packed
        # representation plus cumulative sequence lengths.
        hidden_states = padded_embed[padded_mask_after_cnn]
        # 中文学习备注：
        # 这一步相当于把“padding 后的 chunk batch”重新压成一条 packed 序列。
        # 也就是从 (N_chunk, T', D_model) -> (所有有效时间步拼接后的总长, D_model)
        cu_chunk_lens = [0]
        # 中文学习备注：
        # `window_aftercnn` 表示一个推理 attention 窗口在卷积后对应多少个 token。
        # 默认配置下，`n_window_infer` 会把若干基础 chunk 组合成更大的局部注意力窗口。
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        # 中文学习备注：`cu_seqlens = [0, len1, len1+len2, ...]`，
        # 它是 ragged / varlen attention 的核心索引表。
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

        # ``cu_seqlens`` tells varlen attention where each ragged sequence starts
        # and ends inside the packed ``hidden_states`` tensor.
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )

            hidden_states = layer_outputs[0]
        # 中文学习备注：每层都在同一条 packed 序列上做块内双向注意力。
        # 从数学上说，每一层都保持：
        # hidden_states shape = (S_total_after_cnn, D_model)
        # 但语义表示逐层更抽象。

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        # 中文学习备注：这里输出的是连续 audio features，不是离散 token id。
        # 后面的 thinker 会把它们直接塞进 prompt 的音频占位符位置。
        return BaseModelOutput(last_hidden_state=hidden_states)

    def padded_and_mask_function(self, tensor_list, tensor_len, padding_value=0, padding_side="right"):
        """
        Pads a sequence of tensors to their maximum length on indicated `padding_side`.
        Then prepares a mask so that pad tokens are not attended to.
        """
        # 中文学习备注：
        # 这是一个通用 padding 工具：
        # 输入若干个长度不一的 2D tensor，输出统一长度的 batch tensor 和对应 mask。
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=self.dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        # 返回三件东西：
        # 1. padded_tensor:         (B, dim, max_len)
        # 2. batch_mask:           (B, 1, max_len)
        # 3. batch_mask_after_cnn: (B, max_len_after_cnn)
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )


class Qwen3ASRThinkerTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`
    # 中文学习备注：
    # 这是 thinker 文本侧的 RoPE 生成器。
    # 和普通 LLM 不同，这里还要适配多模态位置，所以会看到 3 维 position ids 和 mRoPE 逻辑。

    def __init__(self, config: Qwen3ASRConfig, device=None):
        super().__init__()
        # `rope_type` 允许同一套代码支持多种 RoPE 变体。
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", "default")
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # 中文学习备注：`persistent=False` 表示它不会被保存到 state_dict，
        # 因为它可以由 config 重新计算出来。
        self.original_inv_freq = self.inv_freq

        self.mrope_section = (config.rope_scaling or {}).get("mrope_section", [24, 20, 20])

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        # 中文学习备注：这里在做多路位置频率的交织重排，
        # 目的是把多模态位置布局对齐到 thinker 期望的格式。
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen3ASRThinker has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        # 中文学习备注：最终返回的是给 attention 用的 cos/sin，不是直接返回 position_ids。
        # 输入形状：
        # x:            (B, S, H_model) 或与之兼容的隐藏状态
        # position_ids: (3, B, S) 或 (B, S)
        # 输出：
        # cos/sin:      (B, S, head_dim) 广播后可作用于 Q/K
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            # 语法点：`with torch.autocast(..., enabled=False)` 是上下文管理器，
            # 明确关闭混合精度，避免 RoPE 三角函数在低精度下累积误差。
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3ASRThinkerTextMLP(nn.Module):
    # 中文学习备注：Thinker 文本模型自己的 MLP 实现。
    # 作用和前面的 `Qwen3ASRTextMLP` 类似，都是 decoder block 的前馈部分。
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 和前面的 TextMLP 一样，都是 (B, S, H) -> (B, S, H)。
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


@use_kernel_forward_from_hub("RMSNorm")
class Qwen3ASRThinkerTextRMSNorm(nn.Module):
    # 中文学习备注：Thinker 文本侧 RMSNorm。
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3ASRThinkerTextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 形状保持不变：(..., hidden_size) -> (..., hidden_size)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3ASRThinkerTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # 中文学习备注：
    # 这是 Thinker 真正用到的文本 attention 实现。
    # 它支持缓存 past_key_values，用于自回归生成。

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3ASRThinkerTextRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3ASRThinkerTextRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape
        self.sliding_window = None
        # 中文学习备注：`sliding_window = None` 表示这里默认不开局部滑窗文本注意力。

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # 中文学习备注：这里的张量流和普通 LLM attention 基本一致，
        # 只是位置编码来自前面专门算出的多模态 RoPE。
        # 输入 hidden_states: (B, S, H_model)
        # Q/K/V after transpose: (B, H_head, S, D_head)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@auto_docstring(
    custom_intro=(
        "Text part of Qwen3ASRThinker, "
    )
)
class Qwen3ASRThinkerTextModel(Qwen3ASRPreTrainedModel):
    # 中文学习备注：这是“只看文本序列”的 decoder 主体。
    # 但这里的“文本序列”已经不一定是纯文本，因为前面可能已经把 audio embeddings 注进来了。
    config: Qwen3ASRConfig
    _no_split_modules = ["Qwen3ASRThinkerTextDecoderLayer"]
    config_class = Qwen3ASRConfig
    _can_record_outputs = {
        "hidden_states": Qwen3ASRThinkerTextDecoderLayer,
        "attentions": Qwen3ASRThinkerTextAttention,
    }

    def __init__(self, config: Qwen3ASRConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3ASRThinkerTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3ASRTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3ASRThinkerTextRotaryEmbedding(config)
        self.gradient_checkpointing = False
        # 中文学习备注：这里自己不关心音频。
        # 它只接受一段已经组装好的 embedding 序列并当作普通 decoder 输入来处理。

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        # `@check_model_inputs` 是 HF 的输入校验 decorator；
        # 它会在 forward 前帮你检查 mutually exclusive 的输入组合等问题。
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)
        # 中文学习备注：DynamicCache 是 HF 的 KV cache 容器。
        # 如果正在 tracing，则避免把这类复杂对象放进图里。

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # 中文学习备注：如果调用方已经提前把 audio embeddings 合进来了，
        # 这里就会直接走 `inputs_embeds` 分支，而不再重新查纯文本 embedding。

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        # 中文学习备注：这里的 3 维 position_ids 是多模态 RoPE 约定的一部分。
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]
        # 中文学习备注：这里兼容两种位置格式：
        # 一种只给 3 路多模态位置，一种额外还带一份文本位置索引。

        # The thinker is still a causal decoder, but it uses the multimodal RoPE
        # indexing scheme produced above.
        # 中文学习备注：虽然有音频参与，但 decoder 依然是严格的 causal decoder。
        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        # 中文学习备注：先统一算一份 cos/sin，再在每层 attention 里复用。
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring(
    custom_intro="""
    The Qwen3ASRThinker model which consists of a audio backbone and a language model.
    """
)
class Qwen3ASRThinkerForConditionalGeneration(Qwen3ASRPreTrainedModelForConditionalGeneration, GenerationMixin):
    # 中文学习备注：
    # 这是 ASR 主体里最重要的类之一。
    # 它做三件事：
    # 1. 调 audio_tower 把音频转成连续向量
    # 2. 把这些向量替换进 prompt 的音频占位符
    # 3. 调文本 decoder 自回归生成识别结果
    config: Qwen3ASRThinkerConfig
    base_model_prefix = "thinker"
    _tied_weights_keys = ["model.embed_tokens.weight", "lm_head.weight"]
    _no_split_modules = [
        "Qwen3ASRAudioEncoderLayer",
        "Qwen3ASRThinkerTextDecoderLayer",
    ]
    _can_record_outputs = {
        "hidden_states": Qwen3ASRThinkerTextDecoderLayer,
        "attentions": Qwen3ASRThinkerTextAttention,
    }

    def __init__(self, config):
        super().__init__(config)
        # 中文学习备注：audio_tower 负责“听”，model/lm_head 负责“写”。
        self.audio_tower = Qwen3ASRAudioEncoder._from_config(config.audio_config)
        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen3ASRThinkerTextModel._from_config(config.text_config)
        if "forced_aligner" in config.model_type:
            self.lm_head = nn.Linear(config.text_config.hidden_size, config.classify_num, bias=False)
        else:
            self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.rope_deltas = None
        self.post_init()

    def get_input_embeddings(self):
        # 代理方法：直接复用内部文本模型的 embedding 表。
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        # 代理方法：允许外部替换 embedding 表，例如做词表扩展或权重共享实验。
        self.model.set_input_embeddings(value)

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
    ):
        """
        Encodes audios into continuous embeddings that can be forwarded to the language model.

        Args:
            input_features (`torch.FloatTensor`):
                The tensors corresponding to the input audios.
            feature_attention_mask (`torch.LongTensor`, *optional*):
                Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
            audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
        """
        if feature_attention_mask is not None:
            feature_lens = torch.sum(feature_attention_mask, dim=1)
        elif audio_feature_lengths is not None:
            feature_lens = audio_feature_lengths
        else:
            raise ValueError("Either `feature_attention_mask` or `audio_feature_lengths` must be provided.")
        # 中文学习备注：
        # 这里有一个接口层兼容写法：既支持显式给长度，也支持从 mask 里现算长度。
    
        # The audio tower is applied sample-by-sample. This is slower than a
        # padded batch, but it avoids subtle precision and masking mismatches for
        # very uneven audio lengths.
        # 中文学习备注：这里按样本逐条跑 audio_tower，
        # 目的是保证裁剪后的真实长度与 placeholder 数严格对齐。
        audio_features = []
        for input_feature, feature_len in zip(input_features, feature_lens):
            # Slice away feature padding before entering the audio encoder so the
            # packed output length matches the placeholder count exactly.
            # 单条样本形状：
            # input_feature[:, :feature_len] -> (mel_bins, real_T)
            # audio_output.last_hidden_state -> (audio_token_len, output_dim)
            audio_output = self.audio_tower(
                input_feature[:, :feature_len],
                feature_lens=feature_len.unsqueeze(0),
            )
            audio_feature = audio_output.last_hidden_state
            audio_features.append(audio_feature)
        audio_features = torch.cat(audio_features, dim=0)

        return audio_features

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_audio_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            ).all(-1)
        else:
            special_audio_mask = input_ids == self.config.audio_token_id
        # 中文学习备注：如果没有 input_ids，只能退化成“比 embedding 向量本身”来找占位符，
        # 这也是为什么这里用了 `.all(-1)`。

        # ``masked_scatter`` expects a boolean mask with the same final shape as
        # the target tensor, so we expand the token-level mask across hidden
        # dimensions.
        # 中文学习备注：这张 mask 回答的是“哪些文本位置其实应该被音频向量替换”。
        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return special_audio_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, Qwen3ASRThinkerCausalLMOutputWithPast]:
        r"""
        feature_attention_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
            The length of feature shape of each audio in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must provide `input_ids` when `inputs_embeds` is not supplied.")
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)
            # 中文学习备注：此时还是“纯文本占位符 embedding”视角。

        # 2. Merge text, audios
        if input_features is not None:
            if input_ids is None:
                raise ValueError("`input_ids` are required when injecting audio features into placeholder tokens.")
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            # Replace placeholder token embeddings with encoded speech features.
            # 中文学习备注：这是整套架构最关键的一步桥接：
            # prompt 里先有一串音频占位符，然后用连续 audio features 覆盖这些位置的 embedding。
            # 形状匹配要求：
            # audio_mask 展开后为 (B, S, H_model)
            # audio_features 需要能按这个 mask 展平后的元素数刚好填满对应位置
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
            # 中文学习备注：
            # 从这一步开始，后面的文本 decoder 已经“不知道自己在看音频占位符”了，
            # 它只看到一段普通的 embedding 序列，其中部分位置来自 audio encoder。

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                # ``delta0`` compensates for left padding so multimodal RoPE
                # indices line up with the true, unpadded token positions.
                # 中文学习备注：第一次解码时要完整计算多模态位置偏移；
                # 后续增量生成时则复用缓存的 rope_deltas。
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    attention_mask,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                # During autoregressive generation we only need the positions for
                # the newly appended token slice, shifted by the cached delta.
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        # 中文学习备注：
        # 这里不再传 `input_ids`，而是直接传 `inputs_embeds`。
        # 原因很简单：音频已经被注入 embedding 级别，无法再用离散 token id 表达。

        hidden_states = outputs[0]
        # 中文学习备注：到这里为止，audio 信息已经完全混进 hidden_states 里了，
        # 后面的 lm_head 并不区分“这是文本来的”还是“这是音频注入来的”上下文。
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 中文学习备注：训练时这里走标准 next-token loss；
            # 推理时 labels=None，loss 分支不会执行。
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
            )

        return Qwen3ASRThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        input_features: Optional[torch.FloatTensor] = None,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> dict[str, object]:
        # 中文学习备注：生成阶段的关键优化在这里：
        # 第一步需要音频特征；后续 token 续写只靠文本侧 KV cache，不必重复编码整段音频。
        # 这也是“重用同一份音频上下文”的关键，否则 streaming/offline 每步都会重复跑 audio_tower。
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            **kwargs,
        )

        # Position ids are recomputed internally because they depend on cached
        # multimodal offsets rather than only on the current token slice.
        model_inputs["position_ids"] = None

        if cache_position is not None and cache_position[0] != 0:
            # Audio features are only needed on the first decoding step. Later
            # steps read from the text-side KV cache.
            model_inputs["input_features"] = None

        return model_inputs


@auto_docstring
class Qwen3ASRThinkerTextPreTrainedModel(PreTrainedModel):
    # 中文学习备注：这个类更像“只暴露文本模型能力”的预训练父类壳。
    config = Qwen3ASRConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3ASRThinkerTextDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3ASRThinkerTextDecoderLayer,
        "attentions": Qwen3ASRThinkerTextAttention,
    }
    config_class = Qwen3ASRConfig


class Qwen3ASRForConditionalGeneration(Qwen3ASRPreTrainedModel, GenerationMixin):
    # 中文学习备注：最外层 API wrapper。
    # 用户通常拿到的是这个类；它内部再把工作转交给 thinker。
    config_class = Qwen3ASRConfig

    def __init__(self, config: Qwen3ASRConfig):
        super().__init__(config)
        self.config = config

        # 中文学习备注：真正的识别逻辑几乎都在 thinker 里。
        self.thinker = Qwen3ASRThinkerForConditionalGeneration._from_config(config.thinker_config)
        self.post_init()
    
    def get_support_languages(self):
        # 简单透传配置里的语言列表。
        return self.config.support_languages

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 4096,
        eos_token_id: int | list[int] = [151645, 151643],
        **kwargs,
    ):
        shared_kwargs = {}
        thinker_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": eos_token_id,
        }
        # 中文学习备注：顶层 wrapper 自己不重新实现生成，
        # 它只是把参数整理后转发给真正的 thinker 子模块。
        # 语法点：这里通过拆 kwargs 的方式把“顶层通用参数”和“thinker 需要的参数”重新归类。

        # Route kwargs to the thinker submodule while preserving the familiar
        # ``model.generate(...)`` surface on the wrapper class.
        for key, value in kwargs.items():
            # Process special input values
            if key == "feature_attention_mask":
                thinker_kwargs[key] = value
            elif key in ("input_features", "attention_mask"):
                thinker_kwargs[key] = value
            # Put other key to shared kwargs
            else:
                shared_kwargs[key] = value

        # Merge kwargs
        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value

        thinker_result = self.thinker.generate(input_ids=input_ids, return_dict_in_generate=True, **thinker_kwargs)

        return thinker_result


__all__ = [
    "Qwen3ASRForConditionalGeneration",
    "Qwen3ASRThinkerTextModel",
    "Qwen3ASRThinkerForConditionalGeneration",
    "Qwen3ASRPreTrainedModel",
    "Qwen3ASRPreTrainedModelForConditionalGeneration",
    "Qwen3ASRThinkerTextPreTrainedModel",
]
