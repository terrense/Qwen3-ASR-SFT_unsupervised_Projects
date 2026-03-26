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

核心源码：

- [qwen_asr/core/transformers_backend/modeling_qwen3_asr.py](./qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)
- [qwen_asr/core/vllm_backend/qwen3_asr.py](./qwen_asr/core/vllm_backend/qwen3_asr.py)
- [qwen_asr/core/transformers_backend/configuration_qwen3_asr.py](./qwen_asr/core/transformers_backend/configuration_qwen3_asr.py)

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
