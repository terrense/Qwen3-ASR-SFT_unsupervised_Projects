# Qwen3-ASR 流式实现技术报告

作者：Codex  
范围：基于当前仓库源码，对 Qwen3-ASR 的“流式识别”实现做分层分析，并解释它与论文描述的关系。  
目标：既讲清“模型为什么能做流式”，也讲清“这个仓库到底是怎么把它包装成流式接口的”。

---

## 1. 先给结论

如果只用一句话概括：

> 这个仓库里的流式识别，是“支持流式/离线统一的音频编码器架构”加上“应用层的分块输入与累计重解码策略”共同实现的。

更具体地说：

1. 模型层面，Qwen3-ASR 的音频塔确实具备论文所说的统一 streaming/offline 能力。
2. 工程层面，这个仓库当前暴露的 streaming API 并不是严格意义上的“完全状态式增量解码”。
3. 它的实际做法是：浏览器持续送音频块，后端缓存这些音频；每当攒够一个 chunk，就把“截至当前的全部音频”重新送给模型解码一次。
4. 为了避免每次重解码都把不稳定尾巴原样带过去，代码会回滚末尾若干 token，再继续生成，所以你会看到字幕尾部被修订。

因此：

- **论文说“模型支持流式”**，这是真的。
- **仓库里的 demo 采用“累计音频 + 重解码”的 streaming 封装**，这也是真的。
- 这两件事并不矛盾。

---

## 2. 读这份报告时，先分清三层

理解这个项目最容易卡住的地方，是把三层逻辑混在一起了。

### 2.1 模型结构层

这一层回答的是：

- 音频是怎么变成语音 token 的？
- 为什么同一个模型既能处理短 chunk，也能处理长音频？
- 为什么论文里说它可以统一 streaming/offline？
- 尽量能够手撕代码

核心源码：

- [qwen_asr/core/transformers_backend/modeling_qwen3_asr.py](./qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)
- [qwen_asr/core/vllm_backend/qwen3_asr.py](./qwen_asr/core/vllm_backend/qwen3_asr.py)
- [qwen_asr/core/transformers_backend/configuration_qwen3_asr.py](./qwen_asr/core/transformers_backend/configuration_qwen3_asr.py)


### 2.1.1 先建立总图 [A+B]

```text
configuration_qwen3_asr.py
  -> 定义模型拓扑和关键超参
  -> 重点看：audio_config / text_config / thinker_config / audio_token_id

modeling_qwen3_asr.py
  -> Transformers 版主实现
  -> 重点看：Qwen3ASRAudioEncoder.forward()
           Qwen3ASRThinkerForConditionalGeneration.get_audio_features()
           get_placeholder_mask()
           forward() 里的 masked_scatter

qwen3_asr.py (vLLM backend)
  -> vLLM 版实现 + 多模态接线层
  -> 重点看：Qwen3ASRAudioEncoder.forward()
           Qwen3ASRMultiModalProcessor._get_prompt_updates()
           embed_multimodal()
           embed_input_ids()
```

把它们串成一句话就是：

> `configuration_qwen3_asr.py` 定义“模型长什么样”，`modeling_qwen3_asr.py` 定义“Transformers 下怎么真正算”，`qwen3_asr.py` 定义“vLLM 下怎么把同一套音频塔和文本解码器接进服务框架”。  

### 2.1.2 先看 configuration_qwen3_asr.py：这不是样板文件，而是结构合同 [A]

这个文件最容易被低估，因为它看起来像“默认参数收纳处”，但其实它定义了整个 Qwen3-ASR 的三层拓扑：

1. `Qwen3ASRAudioEncoderConfig`
2. `Qwen3ASRTextConfig`
3. `Qwen3ASRThinkerConfig`
4. `Qwen3ASRConfig`

建议你读这个文件时，带着下面四个问题看：

- 音频塔输入是什么维度？  
  `num_mel_bins=128`，这直接对应论文里的 128 维 Fbank。
- 音频塔中间怎么跑？  
  `encoder_layers`、`encoder_attention_heads`、`encoder_ffn_dim`、`d_model` 定义了音频 encoder 的 Transformer 主体。
- 长音频靠什么被切开处理？  
  `n_window`、`n_window_infer`、`conv_chunksize` 这三个参数最关键，它们不是随便的工程参数，而是长音频分块、卷积分段、防 OOM、attention window 划分的直接控制柄。
- 音频最后怎么接到文本模型里？  
  `audio_token_id`、`audio_start_token_id` 定义了 prompt 里音频占位符与真正音频特征的桥梁。

这里最值得记住的一点是：

> Qwen3-ASR 不是“一个音频模型 + 一个文本模型松散拼一下”，而是通过 `Qwen3ASRThinkerConfig` 把 `audio_config` 和 `text_config` 绑定成一个真正的多模态生成模型。  

也就是说，后面你在 `modeling_qwen3_asr.py` 里看到的“先算 audio tower，再把 audio feature 塞进文本 embedding 序列”，在配置层其实已经埋好了接口。

### 2.1.3 再看 modeling_qwen3_asr.py：这是结构细节最密的主战场 [A+B]

这份文件很长，但如果是为了学 2.1，你不要平均用力。建议先抓四个锚点。

#### 锚点 1：`_get_feat_extract_output_lengths()` [A]

这是整个“音频会变成多少个语音位置”的长度公式入口。

它做的事情很朴素：

- 输入：原始 mel 特征长度
- 输出：经过 3 层 stride=2 的卷积后，剩下多少个时间步

这件事非常关键，因为后面无论是 Transformers 还是 vLLM，都必须提前知道：

- 一段音频最终会产生多少个 audio feature
- prompt 里需要预留多少个 audio placeholder

所以你可以把这个函数理解为：

> “音频长度”到“音频占位长度”的官方换算器。  

#### 锚点 2：`Qwen3ASRAudioEncoder` [A+B]

这是“音频怎么变成语音 token”的真正答案所在。

更准确地说，源码里产生的不是离散 token id，而是连续的 `audio_features / audio_embeddings`。论文或文档里口语化地叫它“audio token”，但从实现上看，它本质是会去替换文本占位 embedding 的连续向量。

这部分要重点看 `__init__()` 和 `forward()`。

`__init__()` 里先把主干结构搭出来：

- `conv2d1`
- `conv2d2`
- `conv2d3`
- `layers`
- `proj1`
- `proj2`

你可以把这条链路记成：

```text
Fbank
  -> 3层 Conv2d(stride=2)
  -> 压缩后的时序表示
  -> Audio Transformer Encoder
  -> proj1 + 激活 + proj2
  -> audio features
```

`forward()` 里真正体现了“同一个模型既能吃短 chunk，也能吃长音频”：

1. 先按 `self.n_window * 2` 把 mel 特征切成多个 chunk  
2. 每个 chunk 先过 3 层卷积，下采样 8 倍  
3. 卷积输出被整理成 `(time, hidden)` 形式，再加位置编码  
4. 用 `padded_mask_after_cnn` 把有效帧抽出来，形成 packed 的 `hidden_states`  
5. 用 `cu_seqlens` 标记每个 ragged chunk 在 packed 序列中的边界  
6. 每层 `Qwen3ASRAudioEncoderLayer` 都按这些边界做 attention  
7. 最后过 `ln_post + proj1 + act + proj2`，得到可注入文本模型的音频特征

这里有两个学习重点。

第一个重点是：长音频并不是靠“整段做一次超大 dense attention”处理的。

源码实际上做的是：

- 卷积前先切块
- 卷积后再把有效时间步打包
- attention 时靠 `cu_seqlens` 指出每个局部块的边界

所以它本质上是在做一种带块边界的 varlen/ragged attention。

第二个重点是：

> “统一 streaming/offline”在模型结构层上的含义，不是有两套不同网络，而是同一个 `Qwen3ASRAudioEncoder`，面对短输入时块少一点，面对长输入时块多一点，权重完全一样。  

也就是说，流式和离线首先在这里共享的是同一套音频塔，而不是两套模型。

#### 锚点 3：`get_audio_features()` [A]

这个函数在 `Qwen3ASRThinkerForConditionalGeneration` 里，是音频塔和文本解码器之间的桥。

它做的事情是：

- 先根据 `feature_attention_mask` 算出每段音频真正的特征长度
- 对 batch 里的每段音频逐条调用 `self.audio_tower(...)`
- 把各段输出拼接成一个总的 `audio_features`

这里源码专门选择了“逐条算音频塔”，注释里说得很明确，理由是避免长度很不齐时的 padding/masking 误差。

这说明作者在这里优先保证的是：

- 输出长度和 placeholder 数严格对齐
- 多条长度差异很大的音频不会互相干扰

#### 锚点 4：`get_placeholder_mask()` 和 `forward()` 里的 `masked_scatter` [A]

这是“音频怎么接进文本模型”的关键一步。

逻辑非常重要：

1. 先把文本 prompt 变成 `inputs_embeds`
2. prompt 里原本包含若干个 `audio_token_id`
3. `get_placeholder_mask()` 找到这些音频占位位置
4. `inputs_embeds.masked_scatter(audio_mask, audio_features)` 把这些位置上的文本占位 embedding，替换成真正的音频特征

这一步之后，文本 decoder 看到的就不再是“一个抽象的 `<audio>` token”，而是一长串已经编码好的音频向量。

这里再补一个很关键的阅读提醒：

- 在 Transformers 路径里，这些 `audio_token_id` 通常是在进入 `modeling_qwen3_asr.py` 之前，由 processor 侧先准备好的
- 在 vLLM 路径里，这个“把一个音频占位符扩成 N 个位置”的动作，则是直接写在 `qwen3_asr.py` 的 `_get_prompt_updates()` 里

所以从实现视角看：

> 音频不是先被离散化成某套 speech token id 再喂给 decoder，而是先被 audio tower 编成连续向量，然后直接覆盖 prompt 里的音频占位 embedding。  

这是理解整个架构时最容易被说糊涂、但源码里其实最清楚的一点。

### 2.1.4 最后看 qwen3_asr.py：它证明 vLLM 没有“换模型”，只是“换运行时” [A+B]

这份文件如果你第一次看，很容易被大量 vLLM 接口名吓住。正确读法是先分层：

1. 音频塔复刻
2. 多模态处理器
3. vLLM 顶层模型封装

先看音频塔部分，你会发现它和 Transformers 版在结构上是严格同构的：

- 同样的 `_get_feat_extract_output_lengths()`
- 同样的 3 层 stride=2 卷积
- 同样的 `n_window / n_window_infer / conv_chunksize`
- 同样的 packed hidden states + `cu_seqlens`
- 同样的输出投影 `proj1 + act + proj2`

这说明一件事：

> `qwen3_asr.py` 不是重新发明了一套 ASR 架构，而是在 vLLM 的张量并行、多模态注册、服务接口约束下，把同一套架构重写了一遍。  

区别主要在运行时细节上：

- attention 实现换成了 vLLM 的 `MMEncoderAttention`
- 线性层换成了 `QKVParallelLinear / RowParallelLinear / ColumnParallelLinear`
- 多了 processor / parser / registry 这一整层“怎么让 vLLM 知道音频该怎么进模型”的胶水代码

其中最值得重点看的，是 `Qwen3ASRMultiModalProcessor._get_prompt_updates()`。

这个函数几乎把“音频占位符扩成多少个位置”这件事摊开给你看了：

1. 先根据输入音频长度，调用 `_get_feat_extract_output_lengths()` 算出 `audio_output_lengths`
2. 再把 prompt 里的一个音频占位符，替换成 `num_features` 个 `audio_token_id`

也就是说，vLLM 路径里这件事是显式写出来的：

> 一段音频最终会占多少个序列位置，不是拍脑袋决定，而是由音频塔下采样公式提前算出来。  

然后在顶层 `Qwen3ASRForConditionalGeneration` 里，流程变成：

```text
embed_multimodal()
  -> _process_audio_input()
  -> audio_tower()
  -> 得到每段音频的 embeddings

embed_input_ids()
  -> 先做文本 embedding
  -> 再用 _merge_multimodal_embeddings()
  -> 把音频 embeddings 合并进 placeholder 位置
```

这和 Transformers 版的 `masked_scatter` 在本质上是一回事，只是接入点换成了 vLLM 的多模态 planner。

### 2.1.5 三份文件合起来，正好回答你最关心的三个问题 [A+B+C]

#### 问题 1：音频是怎么变成语音 token 的？

更精确的源码表述应该是：

```text
音频波形
  -> feature extractor 变成 128 维 Fbank
  -> audio encoder 做 8 倍时间下采样
  -> transformer encoder 输出连续 audio features
  -> 这些 features 替换 prompt 里的 audio placeholder embeddings
  -> 文本 decoder 在“混合序列”上继续自回归生成
```

所以如果你看到“语音 token”这个说法，最好在脑子里自动翻译成：

> “占据文本序列位置的音频特征向量”，而不是“离散 speech token id”。  

#### 问题 2：为什么同一个模型既能处理短 chunk，也能处理长音频？

因为长度变化不是通过换模型解决的，而是通过同一音频塔里的分块与 packed attention 解决的。

短音频时：

- `chunk_num` 小
- `cu_seqlens` 段数少
- attention 只覆盖少量块

长音频时：

- `chunk_num` 大
- `cu_seqlens` 段数多
- 仍然是同一套卷积层、同一套 encoder 层、同一套投影层

因此“短 chunk”和“长音频”的差别，更多是输入切分方式不同，而不是模型结构不同。

#### 问题 3：为什么论文里说它可以统一 streaming/offline？

从这三份文件能支持的最稳妥表述是：

1. 音频塔结构本身就是按局部窗口和分块 attention 设计的  
2. 同一套权重既能编码短前缀，也能编码长序列  
3. vLLM 和 Transformers 后端都沿用了这套长度换算与占位替换逻辑  

所以论文所说的“统一 streaming/offline”，在模型结构层上是成立的。

但要注意一个非常重要的边界：

> “模型结构支持统一”不等于“这个仓库暴露出的 streaming API 已经做成严格的增量状态式解码”。  

前者是 2.1 这三份文件告诉你的事。  
后者要去 2.2 的推理包装层和 2.3 的 demo 层看。

### 2.1.6 建议你的阅读顺序 [实战版]

如果你现在就要开始读代码，我建议按这个顺序：

1. 先读 `configuration_qwen3_asr.py`  
   只做一件事：把 `audio_config / text_config / thinker_config / audio_token_id` 这几个名词记牢。
2. 再读 `modeling_qwen3_asr.py`  
   只盯住四个锚点：`_get_feat_extract_output_lengths()`、`Qwen3ASRAudioEncoder.forward()`、`get_audio_features()`、`masked_scatter`。
3. 最后读 `qwen3_asr.py`  
   重点确认三件事：音频塔是否同构、placeholder 是怎么扩写的、audio embeddings 是怎么并进 vLLM 的。

如果你按这个顺序读，你会明显感觉到：

- 第一个文件是在“定义世界”
- 第二个文件是在“真正计算世界”
- 第三个文件是在“把这个世界接进服务系统”

这时候再回头看论文里的“统一 streaming/offline”，就不会再停留在一句概念话，而会变成一条你能在代码里逐段对上的实现链。

### 2.2 推理包装层

这一层回答的是：

- 用户给一段音频，仓库怎么调模型？
- “流式状态”到底存了什么？
- 为什么 streaming 只支持 vLLM？

核心源码：

- [qwen_asr/inference/qwen3_asr.py](./qwen_asr/inference/qwen3_asr.py)

### 2.3 演示网页层

这一层回答的是：

- 浏览器麦克风音频怎么送到后端？
- 多久发一次？
- 为什么页面字幕会更新、回退、修订？

核心源码：

- [qwen_asr/cli/demo_streaming.py](./qwen_asr/cli/demo_streaming.py)

---

## 3. 证据等级说明

为了避免“把论文里的描述硬套进代码”或者“把我自己的推理当成源码事实”，下面我用三个等级来标注结论来源。

### 3.1 A 级：源码直接可证

可以直接从仓库源码里看到，无需推断。

### 3.2 B 级：由源码公式或参数直接推导

源码没有把结论原样写出来，但可以通过公式或参数稳定推出。

### 3.3 C 级：源码与论文相互印证后的合理推断

这类结论不是“源码明写”，但和论文表述、实现结构、配置参数高度一致。

---

## 4. 论文里那段话，怎么对应到代码

你引用的论文描述大意是：

1. AuT 是一个 AED 型 ASR 模型
2. 对 128 维 Fbank 做 8 倍下采样
3. 得到 12.5Hz 的音频 token 速率
4. 使用 1s 到 8s 的动态 flash attention window
5. 因而一个模型既可流式，又可离线

下面逐条对应。

### 4.1 128 维 Fbank [A]

音频 encoder 配置类 `Qwen3ASRAudioEncoderConfig` 里，`num_mel_bins` 默认就是 `128`：

- [qwen_asr/core/transformers_backend/configuration_qwen3_asr.py](./qwen_asr/core/transformers_backend/configuration_qwen3_asr.py)

这与论文里 “128 dimensions Fbank” 是直接一致的。

### 4.2 8 倍下采样 [A]

音频塔里有 3 层二维卷积，每层 stride 都是 `2`：

- `conv2d1`
- `conv2d2`
- `conv2d3`

见：

- [qwen_asr/core/transformers_backend/modeling_qwen3_asr.py](./qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)

所以时间维总下采样倍数是：

`2 x 2 x 2 = 8`

这和论文里的 `8 times downsampling` 对上了。

### 4.3 12.5Hz token rate [B]

这里源码没有直接写“12.5Hz”，但可以从两件事推出：

1. 处理器使用的是 `WhisperFeatureExtractor` 风格的音频特征管线  
2. 音频 encoder 的时间维做了 8 倍下采样

仓库里处理器写得很清楚：

- [qwen_asr/core/transformers_backend/processing_qwen3_asr.py](./qwen_asr/core/transformers_backend/processing_qwen3_asr.py)

如果按常见 Whisper 体系理解，输入特征时间步大约是 `100Hz` 量级，那么 8 倍下采样后就是：

`100 / 8 = 12.5Hz`

这正好对应论文的说法。

更细一点说，仓库里的 `_get_feat_extract_output_lengths()` 公式显示：

- 每 `100` 个输入特征步，大致会产出 `13` 个音频 token

这不是严格等于 `12.5`，是因为卷积 padding 和边界取整会让短序列长度出现“近似 12.5Hz、局部不完全整除”的现象。  
所以更准确的表述是：

> 源码长度公式与论文给出的 `12.5Hz` 结论是高度一致的，且边界处允许存在离散取整误差。

### 4.4 动态 flash attention window [A+B+C]

这块要分清“代码直接写的”和“我据此推出来的”。

源码直接可见的部分：

1. 音频 encoder 会先把长音频特征切成若干窗口  
2. 后续 attention 不是对整段 padded 序列做普通 dense attention  
3. 它会构造 `cu_seqlens`，走 flash attention / varlen attention 逻辑

对应源码：

- [qwen_asr/core/transformers_backend/modeling_qwen3_asr.py](./qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)
- [qwen_asr/core/vllm_backend/qwen3_asr.py](./qwen_asr/core/vllm_backend/qwen3_asr.py)

关键参数：

- `n_window`
- `n_window_infer`
- `conv_chunksize`

关键数据结构：

- `chunk_lengths`
- `cu_seqlens`

这说明：

> 音频 encoder 的 attention 作用域，是按窗口和 ragged sequence 来控制的，而不是无约束全局展开。

这与论文所说“dynamic flash attention window”是同方向实现。

需要保守一点的地方是：

- 论文里写的是 `1s` 到 `8s`
- 仓库高层并没有在注释里直白写出“这一个 checkpoint 的窗口策略就是 1 到 8 秒”

所以更稳妥的说法是：

> 源码明确实现了面向音频窗口的 flash attention 计算路径；它与论文的“动态 1s 到 8s window”描述相互吻合，但具体秒级策略更像是 checkpoint/config 与内部实现共同决定的，不应只从 demo 参数去机械理解。

### 4.5 projector 和 audio token 注入 [A]

论文提到的 “projector” 在源码里也能找到对应结构。

在音频 encoder 尾部：

- `proj1`
- 激活函数
- `proj2`

见：

- [qwen_asr/core/transformers_backend/modeling_qwen3_asr.py](./qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)

之后这些音频特征不会单独作为另一个 decoder 输入接口，而是被替换进文本 prompt 里的音频占位 token：

- 处理器负责把一个 `<audio>` 逻辑占位扩成“与音频 encoder 输出长度完全一致”的 placeholder 数量
- 模型 forward 阶段再用 `masked_scatter` 把这些位置替换为真正的 audio features

对应源码：

- [qwen_asr/core/transformers_backend/processing_qwen3_asr.py](./qwen_asr/core/transformers_backend/processing_qwen3_asr.py)
- [qwen_asr/core/transformers_backend/modeling_qwen3_asr.py](./qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)

这就是 Qwen3-ASR 把“音频编码结果”并入“大语言模型解码器”的关键连接点。

---

## 5. 一个真正好懂的总体图

### 5.1 模型视角

```text
原始语音
  -> 128维 Fbank
  -> 3层 stride=2 的卷积下采样
  -> 局部窗口 / varlen flash attention 音频编码器
  -> projector
  -> 音频 embedding 序列
  -> 替换 prompt 中的音频占位 token
  -> Qwen3 文本解码器自回归生成
  -> language + <asr_text> + 转写文本
```

### 5.2 工程视角

```text
浏览器麦克风
  -> 每 500ms 推一小段 PCM 到后端
  -> 后端把音频放进 buffer
  -> buffer 攒够 chunk_size_sec 才触发一次真正解码
  -> 当前累计音频 audio_accum 全量送入模型
  -> 模型返回“当前最好的一版全文”
  -> 后端解析语言和文本
  -> 页面刷新字幕
```

### 5.3 最重要的一句话

> 浏览器是“持续推小块”，后端是“攒够再解”，模型接口是“到当前为止整段重解码”。

这三者不是同一件事。

---

## 6. 源码调用链：离线与流式分别怎么走

## 6.1 离线转录调用链

```text
用户/代码
  -> Qwen3ASRModel.transcribe()
  -> normalize_audios()
  -> split_audio_into_chunks()     # 长音频时切块
  -> _infer_asr_vllm() 或 _infer_asr_transformers()
  -> processor(text, audio)
  -> replace_multimodal_special_tokens()
  -> model.generate()
  -> parse_asr_output()
  -> 合并 chunk 文本
```

关键文件：

- [qwen_asr/inference/qwen3_asr.py](./qwen_asr/inference/qwen3_asr.py)
- [qwen_asr/core/transformers_backend/processing_qwen3_asr.py](./qwen_asr/core/transformers_backend/processing_qwen3_asr.py)
- [qwen_asr/core/transformers_backend/modeling_qwen3_asr.py](./qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)

## 6.2 流式网页 demo 调用链

```text
浏览器麦克风
  -> qwen_asr/cli/demo_streaming.py 中的 JS
  -> /api/start
  -> asr.init_streaming_state()
  -> /api/chunk
  -> asr.streaming_transcribe()
  -> /api/finish
  -> asr.finish_streaming_transcribe()
```

关键文件：

- [qwen_asr/cli/demo_streaming.py](./qwen_asr/cli/demo_streaming.py)
- [qwen_asr/inference/qwen3_asr.py](./qwen_asr/inference/qwen3_asr.py)

## 6.3 streaming 核心调用链

```text
streaming_transcribe(new_pcm, state)
  -> state.buffer += new_pcm
  -> while buffer enough for one chunk:
       取出一个 chunk
       state.audio_accum += chunk
       prefix = "" 或者 回滚末尾 K 个 token 后的旧文本
       prompt = state.prompt_raw + prefix
       model.generate(prompt, audio=state.audio_accum)
       state._raw_decoded = prefix + gen_text
       state.language, state.text = parse_asr_output(...)
```

这段逻辑几乎就是当前 streaming 行为的本质。

---

## 7. `Qwen3ASRModel.LLM()` 为什么是关键入口

流式只支持 vLLM，不支持 transformers backend，这不是 README 里随便说说，而是源码里硬性检查出来的。

### 7.1 vLLM 构造入口

`Qwen3ASRModel.LLM()` 会：

1. 构造 `vllm.LLM`
2. 构造 `Qwen3ASRProcessor`
3. 绑定 `SamplingParams`
4. 返回一个 `backend="vllm"` 的 `Qwen3ASRModel`

对应源码：

- [qwen_asr/inference/qwen3_asr.py](./qwen_asr/inference/qwen3_asr.py)

### 7.2 streaming 为什么只支持 vLLM

在 `streaming_transcribe()` 和 `finish_streaming_transcribe()` 开头，都有明确判断：

- 如果 `self.backend != "vllm"` 就直接报错

所以：

> 当前仓库的 streaming API 不是一个对两种 backend 都对称开放的统一接口，而是只在 vLLM 路径上实现了在线推理封装。

---

## 8. `ASRStreamingState` 里到底存了什么

这是理解“它是不是严格增量 streaming”的关键。

`ASRStreamingState` 里主要有这些字段：

- `buffer`
- `audio_accum`
- `prompt_raw`
- `context`
- `force_language`
- `language`
- `text`
- `_raw_decoded`
- `chunk_id`
- `unfixed_chunk_num`
- `unfixed_token_num`
- `chunk_size_sec`

对应源码：

- [qwen_asr/inference/qwen3_asr.py](./qwen_asr/inference/qwen3_asr.py)

### 8.1 这个 state 没存什么，也很重要

它没有保存：

- 音频 encoder 的显式增量缓存
- decoder 的 `past_key_values`
- “上一块音频已经算过、下一块只补差量”的声学状态

因此，从这个 state 的形状就可以直接看出：

> 当前 streaming 实现不是“完整状态式增量解码器”，而是“应用层缓存音频 + 周期性重解码”。

这是一个非常关键的判断。

---

## 9. streaming 的真实工作流程，逐步拆开讲

## 9.1 第一步：浏览器持续推 PCM

在 [qwen_asr/cli/demo_streaming.py](./qwen_asr/cli/demo_streaming.py) 中：

1. 浏览器采集麦克风
2. 重采样到 `16kHz`
3. 每隔一小段时间把 float32 PCM 发给 `/api/chunk`

要注意：

> 浏览器推送频率，不等于模型真正解码频率。

比如 demo 里 JS 可以每 `500ms` 推一次，但后端不一定每 `500ms` 解一次。

## 9.2 第二步：后端先进入 `buffer`

`streaming_transcribe()` 先把新来的音频追加到 `state.buffer`。

此时只是“收数据”，还不一定触发推理。

## 9.3 第三步：攒够 `chunk_size_sec` 才解码

只有当：

`len(state.buffer) >= state.chunk_size_samples`

时，才会消费一个 chunk 并真正调用模型。

这说明：

- 前端小块上传只是 transport 层
- 后端真正的 decoding cadence 由 `chunk_size_sec` 决定

## 9.4 第四步：把 chunk 拼进 `audio_accum`

触发解码后，代码不是只拿“最新那 1 块”去算，而是：

- `state.audio_accum = old_audio_accum + new_chunk`

也就是保存“从开头到当前时刻的全部音频”。

## 9.5 第五步：构造 prefix，但会回滚尾部

代码没有直接把上一次整段文本当 prompt prefix 完整接上。

它的策略是：

1. 前几个 chunk，`prefix=""`
2. 后面的 chunk，拿 `state._raw_decoded`
3. 把最后 `K` 个 token 回滚掉
4. 剩下的文本作为 prefix 继续 prompt

这就是：

- `unfixed_chunk_num`
- `unfixed_token_num`

这两个参数的含义。

直觉上，这样做是为了：

> 不把最不稳定的尾巴强行锁死在下一个 prompt 里。

## 9.6 第六步：真正送给模型的是“到当前为止整段音频”

这一步是最关键的证据。

送给 vLLM 的输入是：

```text
prompt = state.prompt_raw + prefix
audio  = state.audio_accum
```

也就是：

- prompt 用“旧文本去掉尾巴后”的版本
- audio 用“整段累计音频”

所以当前一步不是“增量补算新 audio chunk”，而是“整段重跑，但给一个更稳定的 prefix”。

## 9.7 第七步：模型返回的是“当前最好的一版全文”

模型生成结果后：

1. 保存到 `state._raw_decoded`
2. 用 `parse_asr_output()` 拆成：
   - `language`
   - `text`

于是页面看到的是：

> 截至当前时刻，模型认为最好的整段转写结果

而不是“某一块 chunk 的局部输出”。

## 9.8 第八步：结束时再做一次 flush

`finish_streaming_transcribe()` 会把不足一个 chunk 的尾音频也拼进去，再做最后一次解码。

所以停止录音时你会常看到最后结果又小改一次，这也是正常行为。

---

## 10. 为什么旧字幕会被删掉或改掉

因为这本来就是 streaming 设计的一部分，不是单纯前端 bug。

根因有两个：

### 10.1 后端允许尾巴被修订

既然每次都是：

- 累计音频重解
- 末尾 token 回滚后续写

那么最后若干 token 本来就不是稳定的。

### 10.2 demo 前端最初采用“整段覆盖显示”

`demo_streaming.py` 自带的前端逻辑是把文本区域整体替换为 `j.text`。

这会让用户产生一个很强的感受：

> 刚刚出现过的字怎么没了？

实际上不是“没了”，而是“新一轮整段最佳假设把它修订掉了”。

所以更合理的 UI 设计是：

- 上面显示已经稳定的 committed history
- 下面单独显示仍会变化的 live tail

这也是我们后面本地 `gateway` 页面做的事情。

---

## 11. 真正要回答的问题：这到底算不算“流式”？

这要看你站在哪一层说。

## 11.1 从产品体验层看：算

因为它确实可以：

- 边说边出字
- 不等整段结束
- 不断修订当前结果

从用户体验上，这当然是 streaming ASR。

## 11.2 从模型能力层看：也算

因为音频 encoder 确实不是只适合整段离线大窗，它有局部窗口 attention 设计，能支撑短 chunk 场景。

这和论文里的 unified streaming/offline 是一致的。

## 11.3 从严格工程实现层看：它不是最“纯”的那类增量解码

如果你的标准是：

- 保存 encoder state
- 保存 decoder KV cache
- 每来一个 chunk 只算必要的新部分

那么当前仓库这份 streaming 封装还没做到那一步。

它更准确的表述是：

> 分块接收音频、以整段累计上下文周期性重解码的在线识别封装。

这是本报告最重要的判断之一。

---

## 12. 为什么这样实现也有道理

虽然它不是最“极致”的增量 streaming，但它有很强的工程现实性。

### 12.1 优点

1. 实现简单，接口清楚
2. 与离线推理输出格式完全一致
3. 容易复用同一套 prompt、processor、decoder 逻辑
4. 末尾文本可通过 rollback 策略获得比“纯粹 append”更稳定的结果

### 12.2 代价

1. 同一段前缀音频会被重复编码多次
2. 计算量比真正增量解码大
3. 尾部文本会抖动
4. 不支持时间戳
5. 不支持 batch streaming

这些限制在 README 和源码注释里也都能找到印证。

---

## 13. 一个特别重要的“时间尺度”拆分

很多人会把这三个时间尺度混在一起。

### 13.1 浏览器推送尺度

例如每 `500ms` 发一次 PCM。

这是 transport 频率。

### 13.2 后端触发解码尺度

由 `chunk_size_sec` 决定，常见是 `1.0s` 或 `2.0s`。

这是 API 层真正触发 `generate()` 的频率。

### 13.3 模型音频 token 尺度

论文给的是大约 `12.5Hz`。

也就是音频 encoder 输出 token 的内部时间粒度，大约每 `80ms` 一个 token。

所以一个很容易读懂的关系是：

```text
500ms 浏览器包
  != 1s 后端 chunk
  != 80ms 左右的模型音频 token
```

如果这三层没分清，读 streaming 代码很容易晕。

---

## 14. 音频 encoder 为什么能统一流式和离线

这是论文想强调的点，也是源码确实体现出来的地方。

## 14.1 它不是把整段音频当作一个固定长度块硬塞进去

音频 encoder 的 forward 会：

1. 根据长度把特征切成窗口
2. 做卷积下采样
3. 构造 ragged 序列边界 `cu_seqlens`
4. 用 varlen / flash attention 对这些局部块做编码

这意味着：

> 它天然适合“短块也能跑，长段也能跑”的统一处理方式。

## 14.2 同一个模型可以有两种工作模式

### 离线模式

- 输入一大段音频
- encoder 内部自己按窗口组织
- 最终输出完整结果

### 在线模式

- 输入当前已经收集到的音频
- 同样走同一个 encoder 架构
- 只是应用层不断刷新“截至当前的整段结果”

所以统一 streaming/offline 的“统一”，更准确地说是：

> 统一的是模型架构和编码机制；不一定意味着应用层必须实现成最严格的增量缓存式推理。

---

## 15. 从 prompt 角度看，Qwen3-ASR 为什么能像 LLM 一样工作

这个项目不是一个“传统 CTC-only ASR 包装”。

它的关键做法是：

1. 用 chat template 先生成文本 prompt
2. 在 prompt 中预留音频占位 token
3. 把音频 encoder 产出的特征替换进这些位置
4. 再让 Qwen3 文本 decoder 自回归生成结果

关键函数：

- `_build_text_prompt()`
- `replace_multimodal_special_tokens()`
- `get_audio_features()`
- `get_placeholder_mask()`
- `masked_scatter(...)`

对应文件：

- [qwen_asr/inference/qwen3_asr.py](./qwen_asr/inference/qwen3_asr.py)
- [qwen_asr/core/transformers_backend/processing_qwen3_asr.py](./qwen_asr/core/transformers_backend/processing_qwen3_asr.py)
- [qwen_asr/core/transformers_backend/modeling_qwen3_asr.py](./qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)

这套设计解释了为什么：

- 它能输出 `language Chinese<asr_text>...`
- 它本质上是“音频条件下的语言模型生成”
- 它的流式策略会自然落到“prompt prefix + 新一轮生成”这条路上

---

## 16. 调用链，按文件再串一遍

## 16.1 从浏览器到模型

```text
qwen_asr/cli/demo_streaming.py
  -> JS 采集麦克风
  -> /api/start
  -> asr.init_streaming_state()
  -> /api/chunk
  -> asr.streaming_transcribe()
  -> model.generate(...)
  -> parse_asr_output(...)
  -> 返回 JSON 给浏览器
```

## 16.2 从 `streaming_transcribe()` 到 vLLM

```text
qwen_asr/inference/qwen3_asr.py
  -> prompt = state.prompt_raw + prefix
  -> inp = {"prompt": prompt, "multi_modal_data": {"audio": [state.audio_accum]}}
  -> self.model.generate([inp], ...)
```

这里的 `self.model` 是 `vllm.LLM` 实例，由 `Qwen3ASRModel.LLM()` 构造。

## 16.3 从 processor 到音频特征长度对齐

```text
Qwen3ASRProcessor.__call__()
  -> WhisperFeatureExtractor 提取音频特征
  -> _get_feat_extract_output_lengths()
  -> replace_multimodal_special_tokens()
```

这个步骤的意义是：

> 文本里的音频 placeholder 数量，必须和音频 encoder 输出 token 数量完全一致。

否则 embedding 替换就会 shape 对不上。

## 16.4 从音频塔到 decoder

```text
Qwen3ASRThinkerForConditionalGeneration.forward()
  -> get_audio_features()
  -> audio_tower(...)
  -> get_placeholder_mask()
  -> inputs_embeds.masked_scatter(audio_mask, audio_features)
  -> text decoder forward
  -> lm_head
```

这一步是整个 multimodal ASR 的结构核心。

---

## 17. 我对“论文与源码关系”的最终判断

如果把你的问题压缩成一句：

> “论文说 unified streaming/offline，为什么源码看起来像在重算整段音频？”

我的回答是：

### 17.1 两者关注点不同

论文说的是：

- 模型架构层面，音频 encoder 的设计允许同一个模型支持短块和长段

源码 demo 做的是：

- 工程层面，用最稳妥直接的方法把这种能力暴露成在线 API

### 17.2 仓库没有完全把“模型的流式潜力”榨干

它没有做到：

- 持久化 encoder 增量状态
- 只重算必要的新片段
- 复用 decoder 级别的严格增量缓存

但它已经做到：

- 模型统一
- 接口统一
- 短 chunk 可用
- 在线输出可用

所以我的判断是：

> 这不是“论文吹了，代码没实现”；而是“论文给了统一架构能力，仓库先用了一种工程上更简单、更稳的 streaming 封装方式来落地”。

---

## 18. 如果未来想把它做成更“真”的 streaming，需要改哪

这部分已经超出“当前源码是什么”的范围，但为了完整性，我还是写一下。

如果你想做得更接近“严格增量解码”，理论上要往三处动：

### 18.1 音频 encoder 状态复用

目标：

- 不是每次把 `audio_accum` 整段再过一遍 audio_tower
- 而是只对新来的 chunk 做增量编码，并维护 encoder-side state

### 18.2 decoder cache 复用

目标：

- 让 prefix 不是“重新喂一遍文本”
- 而是让 decoder 真正复用历史 KV cache

### 18.3 输出稳定性机制重构

当前是：

- `rollback last K tokens`

更高级的做法可以是：

- committed / live 双缓冲
- token 级置信度或稳定度策略
- 语言段落边界上的更细粒度稳定机制

不过这些都不是当前仓库已经提供的能力。

---

## 19. 对你现在最有用的简化理解

如果你只想要一个“以后再看代码也不容易忘”的版本，我建议记住下面四句话。

### 19.1 第一句

Qwen3-ASR 的模型本身，确实不是纯离线模型；它的音频 encoder 结构就是为统一 streaming/offline 设计的。

### 19.2 第二句

仓库当前 streaming API 的工作方式，是“不断接收音频小块，但周期性重解截至当前的整段音频”。

### 19.3 第三句

字幕尾部会改，是因为代码故意回滚最后若干 token 再重生成，这不是 bug。

### 19.4 第四句

所以你看到的 streaming，是“架构上支持流式 + 工程上采用累计重解码封装”的组合，而不是最极致的纯增量解码器。

---

## 20. 附录：你接下来继续读源码，建议的顺序

如果你想继续往下读，我建议按这个顺序：

1. [qwen_asr/inference/qwen3_asr.py](./qwen_asr/inference/qwen3_asr.py)  
   先看 `Qwen3ASRModel.LLM()`、`transcribe()`、`init_streaming_state()`、`streaming_transcribe()`

2. [qwen_asr/core/transformers_backend/processing_qwen3_asr.py](./qwen_asr/core/transformers_backend/processing_qwen3_asr.py)  
   重点看 `_get_feat_extract_output_lengths()` 和 `replace_multimodal_special_tokens()`

3. [qwen_asr/core/transformers_backend/modeling_qwen3_asr.py](./qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)  
   重点看 `Qwen3ASRAudioEncoder.forward()`、`get_audio_features()`、`masked_scatter`

4. [qwen_asr/cli/demo_streaming.py](./qwen_asr/cli/demo_streaming.py)  
   重点看 `/api/start`、`/api/chunk`、`/api/finish` 和浏览器 JS

5. [examples/example_qwen3_asr_vllm_streaming.py](./examples/example_qwen3_asr_vllm_streaming.py)  
   它很适合验证你对 streaming 状态机的理解

---

## 21. 附录：一句话回答你最初的问题

**Qwen3-ASR 的流式实现，不是单靠网页不断发 chunk 就自动成立；真正关键的是：音频 encoder 架构允许短块处理，而仓库在应用层通过“累计音频 + 回滚尾 token + 重解码”的策略，把这种能力包装成了一个可工作的流式接口。**
