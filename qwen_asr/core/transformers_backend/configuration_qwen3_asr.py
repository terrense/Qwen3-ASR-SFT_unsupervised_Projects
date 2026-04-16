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
Configuration objects for the Qwen3-ASR family.

The project uses nested Hugging Face ``PretrainedConfig`` objects to describe
the composite architecture:

1. An audio encoder that compresses log-mel features into dense speech tokens.
2. A text/"thinker" decoder that performs autoregressive reasoning and output.
3. A top-level wrapper that binds the sub-configs into one checkpoint schema.

Keeping these concerns in separate config classes mirrors the actual model
topology and makes it easier to reuse the text or audio pieces independently.
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


# 中文学习备注：
# 这份文件不是简单的“默认参数表”，而是在定义整个 Qwen3-ASR 模型的
# “可序列化结构说明书”。训练、推理、保存 checkpoint、重新加载 checkpoint、
# 对接 vLLM/Transformers，都会依赖这里的配置对象。
#
# 可以把本文件理解成回答下面 4 个问题：
# 1. 音频输入先进入哪种 audio encoder，它的宽度、层数、窗口策略是什么？
# 2. 编码后的音频特征最终要接到哪个文本 decoder 上，文本侧隐藏维和注意力结构是什么？
# 3. 音频塔和文本塔如何被组合成一个“可生成文本”的多模态 ASR 模型？
# 4. 当外部框架只拿到 config.json 时，如何仅凭配置就把同一套模型结构重建出来？
#
# 从“代码阅读”角度，建议分 4 层来读：
# 1. `Qwen3ASRAudioEncoderConfig`
#    只关心“听”的部分：mel 特征如何被压缩成更短、更稠密的音频表示。
# 2. `Qwen3ASRTextConfig`
#    只关心“写”的部分：文本 decoder 的隐藏维、头数、RoPE、KV cache 等。
# 3. `Qwen3ASRThinkerConfig`
#    第一次把 audio/text 两个子配置装进同一个组合配置对象里。
# 4. `Qwen3ASRConfig`
#    最外层 wrapper，负责把整个模型包装成一个可被 AutoConfig/AutoModel 识别的顶层架构。
#
# 从“和 modeling 文件联动”的角度，也可以这样对应：
# - 这里的 `AudioEncoderConfig` -> `modeling_qwen3_asr.py` 里的 `audio_tower`
# - 这里的 `TextConfig` -> `modeling_qwen3_asr.py` 里的文本 decoder / lm 部分
# - 这里的 `ThinkerConfig` -> `Qwen3ASRThinkerForConditionalGeneration`
# - 这里的 `Qwen3ASRConfig` -> 最外层 `Qwen3ASRForConditionalGeneration`
class Qwen3ASRAudioEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3ASRAudioEncoder`]. It is used to instantiate a
    Qwen3-ASR audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the Qwen2-Audio
    architecture.

    e.g. [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel features used per input features. Should correspond to the value used in the
            `Qwen3ASRProcessor` class.
        encoder_layers (`int`, *optional*, defaults to 32):
            Number of encoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
        d_model (`int`, *optional*, defaults to 1280):
            Dimensionality of the layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(d_model).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
        n_window (`int`, *optional*, defaults to 100):
            The chunk for conv and flash attn in AudioEncoder.
        output_dim (`int`, *optional*, defaults to 3584):
            The output dimension of AudioEncoder.

    Example:

    ```python
    >>> from transformers import Qwen3ASRAudioEncoderConfig, Qwen3ASRAudioEncoder

    >>> # Initializing a Qwen3ASRAudioEncoderConfig
    >>> configuration = Qwen3ASRAudioEncoderConfig()

    >>> # Initializing a Qwen3ASRAudioEncoder (with random weights)
    >>> model = Qwen3ASRAudioEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_asr_audio_encoder"
    # 中文学习备注：
    # 这个类只描述“音频塔”本身，不包含文本 decoder，也不关心最终词表输出。
    # 你可以把它理解成一条纯语音编码流水线：
    #
    #   原始波形
    #   -> Fbank / log-mel 特征
    #   -> 3 层 stride=2 Conv2d 做时间下采样
    #   -> Audio Transformer 做时序建模
    #   -> 输出为 `output_dim` 维连续音频特征
    #
    # 这些特征后面不会直接拿来分类，而是会被送去覆盖文本 prompt 中的音频占位符，
    # 最终交给文本 decoder 继续做自回归生成。
    #
    # 所以读这个配置类时，重点不要只看“数值大小”，而要看这些字段如何决定：
    # 1. 音频序列被压缩多少倍
    # 2. 编码后的特征宽度是多少
    # 3. 推理时是否会按窗口/分块做局部计算来节省显存

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        d_model=1280,
        dropout=0,
        attention_dropout=0,
        activation_function="gelu",
        activation_dropout=0,
        scale_embedding=False,
        initializer_range=0.02,
        max_source_positions=1500,
        n_window=100,
        output_dim=3584,
        n_window_infer=400,
        conv_chunksize=500,
        downsample_hidden_size=480,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # 中文学习备注：
        # 这里看起来只是“把参数存起来”，但配置类的作用远不止普通 dataclass：
        # 1. 会被 Hugging Face 自动序列化进 `config.json`
        # 2. `modeling_qwen3_asr.py` 里会通过 `_from_config(...)` 用这些字段实例化真实模块
        # 3. vLLM 之类的下游框架也会读取同一份配置并重建等价结构
        # 4. 因此这里的每个字段，本质上都在参与定义“checkpoint 的结构契约”
        #
        # 读字段时推荐按 4 组理解，而不是零散地背参数名：
        # - 输入特征组：`num_mel_bins`
        # - 主干网络组：`d_model` / `encoder_layers` / `encoder_attention_heads` / `encoder_ffn_dim`
        # - 正则化与初始化组：`dropout` / `attention_dropout` / `activation_dropout` / `initializer_range`
        # - 长音频推理组：`n_window` / `n_window_infer` / `conv_chunksize`

        # 输入侧：一帧音频特征的通道数。它必须和 processor 提取出的 mel 维度一致，
        # 否则模型第一层就会发生 shape 不匹配。
        self.num_mel_bins = num_mel_bins
        # `d_model` 是 audio tower 内部的主隐藏维，决定每层表示的宽度。
        self.d_model = d_model
        # Transformer 主干深度，决定音频侧堆多少层时序建模模块。
        self.encoder_layers = encoder_layers
        # 音频 Transformer 的注意力头数。
        self.encoder_attention_heads = encoder_attention_heads
        # 音频侧 FFN 的中间层宽度，通常显著大于 `d_model`。
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        # Hugging Face utilities often look for ``num_hidden_layers`` regardless
        # of whether the module is an encoder or decoder, so we mirror the value.
        # 中文学习备注：
        # 这行是“框架兼容字段”。虽然这里是 audio encoder，不是 decoder，
        # 但很多 HF 工具只认通用名字 `num_hidden_layers`，所以这里做一份镜像。
        # 这样做可以减少不同框架/工具在读配置时的特判逻辑。
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        # 如果为 True，embedding 会按 sqrt(d_model) 做缩放。
        # 这是标准 Transformer 里常见的稳定训练技巧之一。
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        # 这是音频特征最大长度上限，主要用于位置编码和输入约束。
        self.max_source_positions = max_source_positions
        # 中文学习备注：
        # `n_window` 决定训练/常规编码时的局部窗口语义。
        # 你可以把它理解成 audio encoder 在处理很长 mel 序列时的“局部视野粒度”。
        # 编码时会先按 `n_window * 2` 的粒度切 mel 帧，再做卷积和局部 attention。
        self.n_window = n_window
        # 音频塔最终产出的连续特征维度。后续这些特征会被注入到文本侧 embedding 序列里。
        self.output_dim = output_dim
        # 中文学习备注：
        # `n_window_infer` 是推理阶段的窗口大小，通常会和训练窗口区分开，
        # 因为推理时更关注显存占用和吞吐。
        #
        # `conv_chunksize` 则更偏工程实现：它只是把卷积阶段切成更小分段来防 OOM，
        # 本意不是改变模型语义，而是让大输入在有限显存上也能跑起来。
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        # 中文学习备注：
        # 这是前面 3 层 stride=2 Conv2d 的中间通道宽度。
        # 它影响下采样前端的容量，也会影响卷积投影阶段的计算量。
        self.downsample_hidden_size = downsample_hidden_size


class Qwen3ASRTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3ASRTextModel`]. It is used to instantiate a
    Qwen3-ASR model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen3-ASR-1.7B [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen3ASR model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen3ASRModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
        head_dim (`int`, *optional*, defaults to 128):
            The dimension of the head. If not specified, will default to `hidden_size // num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 128000):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 5000000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Qwen3ASRTextModel, Qwen3ASRTextConfig

    >>> # Initializing a Qwen3ASR style configuration
    >>> configuration = Qwen3ASRTextConfig()

    >>> # Initializing a model from the Qwen3-VL-7B style configuration
    >>> model = Qwen3ASRTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_asr_text"
    base_config_key = "text_config"
    # 中文学习备注：
    # 这里定义的是文本 decoder 的超参数，整体上就是一个 Qwen 风格的 causal decoder。
    # 但在 ASR 场景下，它的输入并不是“纯文本 token 序列”这么简单：
    #
    # 1. prompt 里先会放入一串音频占位 token
    # 2. 在 `modeling_qwen3_asr.py` 里，这些占位位置会被 audio tower 输出的连续向量替换
    # 3. 文本 decoder 随后把“文本 token + 音频向量”看成同一条 embedding 序列继续解码
    #
    # 所以这个配置类虽然名字叫 TextConfig，但它控制的不只是“写文字”的能力，
    # 还隐含决定了音频特征最终要注入到多宽的 hidden space 中。
    #
    # 这些字段和张量形状直接相关，是阅读模型代码时最常见的锚点：
    # - `hidden_size`：每个位置的隐藏表示宽度
    # - `num_attention_heads`：Query 头数
    # - `num_key_value_heads`：Key/Value 头数，决定是 MHA / GQA / MQA
    # - `head_dim`：单个头的宽度，通常满足 `hidden_size = num_attention_heads * head_dim`
    #
    # 也可以把本类参数分成 5 组来记：
    # - 词表与输出：`vocab_size`
    # - 主干规模：`hidden_size` / `intermediate_size` / `num_hidden_layers`
    # - 注意力形状：`num_attention_heads` / `num_key_value_heads` / `head_dim`
    # - 长上下文机制：`max_position_embeddings` / `rope_theta` / `rope_scaling`
    # - 推理效率：`use_cache`

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=128000,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=5000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        # 中文学习备注：
        # 这里没有先 `super().__init__()` 再赋值，而是先把核心字段写好，再调父类。
        # 这是 HF 配置类里很常见的写法，因为父类初始化流程、序列化逻辑或者某些工具方法，
        # 在构造阶段就可能读取这些字段。
        #
        # 直觉上看它像“普通赋值顺序”，本质上却是在保证：
        # “一旦父类开始接管这个配置对象，它已经具备足够完整的结构信息”。
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # Older checkpoints may omit ``num_key_value_heads``. Falling back to
        # full multi-head attention keeps those configs loadable.
        if num_key_value_heads is None:
            # 中文学习备注：这里等价于“退化成普通多头注意力 MHA”。
            # 也就是让每个 Query 头都拥有自己独立的 Key/Value 头。
            # 这么做的主要目的是 checkpoint 兼容，而不是新的结构设计。
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            # 中文学习备注：这是做老 checkpoint 向后兼容。
            # 旧配置里有些会写成 `type`，新逻辑期望字段名是 `rope_type`。
            # 这里不是在发明新参数，而是在统一不同历史版本的命名。
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        # `tie_word_embeddings` 交给父类统一处理，因为它属于 HF 通用配置语义：
        # 是否复用输入 embedding 和输出 lm_head 的权重。
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen3ASRThinkerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3ASRThinker`]. It is used to instantiate a
    Qwen3-ASR-Thinker model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the thinker component of the Qwen3-Omni
    architecture.

    e.g. [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`dict`, *optional*):
            The config dictionary of the audio backbone.
        text_config (`dict`, *optional*):
            The config dictionary of the text backbone.
        audio_token_id (`int`, *optional*, defaults to 151646):
            The audio token id to encode the audio prompt.
        audio_start_token_id (`int`, *optional*, defaults to 151647):
            The audio start token id to encode the audio prompt.
        user_token_id (`int`, *optional*, defaults to 872):
            The user token id to encode the user token.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import Qwen3ASRThinkerModel, Qwen3ASRThinkerConfig

    >>> # Initializing a default Qwen3ASRThinkerConfig
    >>> configuration = Qwen3ASRThinkerConfig()

    >>> # Initializing a model (with random weights) from the default configuration
    >>> model = Qwen3ASRThinkerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_asr_thinker"
    # 中文学习备注：
    # Thinker 这一层是“组合配置”的核心，因为它第一次把 `audio_config` 和 `text_config`
    # 放进同一个对象里。后面模型初始化时，audio_tower 和 text model 都是从这里拆出来的。
    #
    # 你可以把它理解成“多模态拼装说明书”：
    # - AudioEncoderConfig 只知道怎样把音频编码成连续向量
    # - TextConfig 只知道怎样把一串 embedding 当作上下文继续生成文本
    # - ThinkerConfig 则定义：这两个部件如何被放进同一个可生成模型中
    #
    # 如果说 Audio/Text 两个 config 是“零件参数表”，
    # 那 ThinkerConfig 更像“装配图”。

    attribute_map = {}
    sub_configs = {
        "audio_config": Qwen3ASRAudioEncoderConfig,
        "text_config": Qwen3ASRTextConfig,
    }

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id=151646,
        audio_start_token_id=151647,
        user_token_id=872,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # 中文学习备注：Thinker 是真正把“音频塔 + 文本 decoder”捆在一起的那一层。
        # 后面在 modeling_qwen3_asr.py 里看到的 audio_tower / model，就是从这里分出来的。
        # 你可以把它理解成“一个组合配置对象”，而不是单一子网络配置。
        #
        # 这里最重要的不是几个 token id 本身，而是“组合关系”：
        # - `audio_config` 决定怎样把音频编码成连续特征
        # - `text_config` 决定这些特征最终要注入到怎样的 decoder hidden space
        # - `audio_token_id` / `audio_start_token_id` 决定 prompt 里音频片段如何被占位和标记
        self.user_token_id = user_token_id
        self.audio_start_token_id = audio_start_token_id
        self.initializer_range = initializer_range

        # Nested config dicts are eagerly materialized into typed config objects
        # so downstream code can rely on attribute access rather than dictionary
        # lookups.
        # 中文学习备注：
        # checkpoint 反序列化回来时，子配置经常最初只是原始 dict。
        # 这里会把 dict 立即“提升”为强类型配置对象，原因有三层：
        # 1. 后续模型代码可以直接用 `config.audio_config.xxx`，而不是手动查字典
        # 2. 类型边界更清晰，谁是 audio 子配置、谁是 text 子配置一目了然
        # 3. 下游框架需要递归读取子配置时，也更容易统一处理
        if isinstance(audio_config, dict):
            # 中文学习备注：允许从原始 dict 直接恢复成强类型配置对象。
            audio_config = Qwen3ASRAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            # 没显式传入时，就按默认音频塔结构构造一份配置。
            audio_config = Qwen3ASRAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = Qwen3ASRTextConfig(**text_config)
        elif text_config is None:
            # 同理，文本塔也支持“没传就用默认结构”。
            text_config = Qwen3ASRTextConfig()
        self.text_config = text_config
        # 中文学习备注：prompt 里的音频占位符最终会落到这个 token id 上，
        # 后续再由 audio features 覆盖这些位置的 embedding。
        # 这也是为什么 modeling 代码里会先根据 `input_ids == audio_token_id`
        # 找到占位位置，再把 audio encoder 输出塞进去。
        self.audio_token_id = audio_token_id


class Qwen3ASRConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Qwen3ASRForConditionalGeneration`]. It is used to instantiate a Qwen3ASR
    model according to the specified sub-models configurations, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        thinker_config (`dict`, *optional*): Configuration of the underlying thinker sub-model.
        support_languages (`List[str]`, *optional*): The languages supported by the model.

    Example:

    ```python
    >>> from transformers import (
    ...     Qwen3ASRThinkerConfig,
    ...     Qwen3ASRForConditionalGeneration,
    ...     Qwen3ASRConfig,
    ... )

    >>> # Initializing a Qwen3ASR style configuration
    >>> configuration = Qwen3ASRConfig()

    >>> # Initializing a model from the configuration
    >>> model = Qwen3ASRForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_asr"
    # 中文学习备注：
    # 这是最外层 wrapper config，也是外部框架最先看到的顶层配置类型。
    # 它不自己展开大量结构细节，而是把主体继续委托给 `thinker_config`。
    #
    # 为什么还需要这一层，而不是直接拿 ThinkerConfig 当顶层？
    # 因为从 Hugging Face 生态角度看，需要一个“整个模型”的统一入口：
    # - `AutoConfig` 需要知道这个 checkpoint 的 `model_type` 是什么
    # - `AutoModel` 需要知道顶层该实例化哪个模型类
    # - 顶层还需要放一些整模型级别的元信息，例如 `support_languages`
    #
    # 所以这层更像“仓库对外声明的标准包装壳”。
    sub_configs = {
        "thinker_config": Qwen3ASRThinkerConfig,
    }

    def __init__(
        self,
        thinker_config=None,
        support_languages=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if thinker_config is None:
            thinker_config = {}
        # 中文学习备注：顶层配置允许直接传一个 dict 进来，
        # 内部再把它包装成 Qwen3ASRThinkerConfig。
        # 这意味着顶层 checkpoint 不需要先手工构造好 `Qwen3ASRThinkerConfig` 实例，
        # 只要提供结构兼容的字典即可恢复出完整组合配置。

        # The top-level config mostly delegates architectural detail to the
        # nested thinker config, while keeping model-wide metadata such as the
        # supported language list.
        # 中文学习备注：顶层 config 本身并不展开音频塔和文本塔的全部细节，
        # 而是把绝大多数结构信息继续下放给 thinker_config。
        # 因此你可以把 `self.thinker_config` 看成“真正的结构主体”，
        # 而 `Qwen3ASRConfig` 自己更像“壳 + 元信息 + 框架入口”。
        self.thinker_config = Qwen3ASRThinkerConfig(**thinker_config)
        # 这个字段不是网络结构参数，而是模型能力元信息：
        # 它告诉上层调用者该 checkpoint 声称支持哪些语言。
        self.support_languages = support_languages

    def get_text_config(self, decoder=False) -> "PretrainedConfig":
        """
        Returns the config that is meant to be used with text IO. On most models, it is the original config instance
        itself. On specific composite models, it is under a set of valid names.

        Args:
            decoder (`Optional[bool]`, *optional*, defaults to `False`):
                If set to `True`, then only search for decoder config names.
        """
        # Overridden for deeply nested config like Qwen2.5-Omni. We don't have any omni model
        # except for Qwen yet. This has to be generalized if more deeply nested configs are
        # added. NOTE: currently method used only by vLLM
        # 中文学习备注：vLLM 常常需要“给我文本 decoder 的配置”而不是整个 ASR wrapper，
        # 所以这里提供一个统一入口把它取出来。
        # 这也是“组合模型配置”和“单塔配置”之间最常见的桥接方法之一。
        #
        # 注意这里虽然参数里有 `decoder=False`，但当前实现没有基于它做复杂分支，
        # 因为这个模型的文本 IO 主体非常明确，就是 thinker 里的 text_config。
        # 这个接口名更像是在对齐 HF/vLLM 的通用调用约定。
        return self.thinker_config.get_text_config()


__all__ = ["Qwen3ASRConfig", "Qwen3ASRThinkerConfig", "Qwen3ASRAudioEncoderConfig"]
