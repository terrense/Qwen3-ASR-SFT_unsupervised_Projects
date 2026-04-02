# Qwen3-ASR Streaming 调用链报告

## 0. 先给结论

这个仓库里，`streaming` 不是“每来一点音频，就只增量推进一次声学隐藏状态”的严格流式 ASR。

它实际做的是：

1. 浏览器持续采集麦克风音频。
2. 前端按小块把 `float32`、`16kHz`、单声道 PCM 发给网关。
3. 网关把这些小块累积到 `state.buffer`。
4. 只有当 `buffer` 累积到一个“服务端 chunk”大小时，才触发一次解码。
5. 每次触发解码时，不是只喂新 chunk，而是把“从开头到当前为止的全部音频”重新送进模型。
6. 为了减少尾部抖动，它会把上次结果末尾回滚若干 token，再作为 prefix prompt。
7. 模型重新生成当前整段的最新假设，网关再把它切成 `committed_text + live_text` 返回给前端。

所以它的核心机制可以概括成一句话：

`累计音频 + 全量重解码 + prefix rollback + committed/live split`

---

## 1. 从前端音频到字幕的总调用链

下面这条链，是当前项目里“浏览器麦克风 -> 网关 -> 模型 -> 字幕”的真实主路径。

```text
浏览器页面
local_web/qwen3_asr_gateway.html
  startStreaming()
    -> apiStreamStart()
      -> POST /api/stream/start
    -> navigator.mediaDevices.getUserMedia(...)
    -> AudioContext + ScriptProcessor
    -> onaudioprocess
       -> resampleLinear(...)
       -> concatFloat32(...)
       -> pumpStream()
          -> apiStreamChunk(chunk)
             -> POST /api/stream/chunk?session_id=...

网关服务
local_web/qwen3_asr_gateway.py
  api_stream_start()
    -> asr.init_streaming_state(...)

  api_stream_chunk()
    -> np.frombuffer(raw, dtype=np.float32)
    -> asr.streaming_transcribe(wav, sess.state)
       -> Qwen3ASRModel.streaming_transcribe(...)
          -> 累积 state.buffer
          -> 满一个服务端 chunk 就执行一次解码
          -> 构造 prompt = state.prompt_raw + prefix
          -> self.model.generate([{"prompt": prompt, "multi_modal_data": {"audio": [state.audio_accum]}}])
          -> parse_asr_output(...)
          -> 更新 state.language / state.text / state._raw_decoded / state.chunk_id
    -> _stream_payload(...)
       -> _split_committed_and_live_text(...)
       -> 返回 committed_text/live_text

  api_stream_finish()
    -> asr.finish_streaming_transcribe(sess.state)
    -> _stream_payload(final=True)

高层推理包装
qwen_asr/inference/qwen3_asr.py
  Qwen3ASRModel.init_streaming_state(...)
  Qwen3ASRModel.streaming_transcribe(...)
  Qwen3ASRModel.finish_streaming_transcribe(...)
  Qwen3ASRModel._build_text_prompt(...)

vLLM 多模态路径
qwen_asr/core/vllm_backend/qwen3_asr.py
  Qwen3ASRProcessingInfo
  Qwen3ASRMultiModalProcessor._get_prompt_updates(...)
  Qwen3ASRForConditionalGeneration.embed_multimodal(...)
    -> _parse_and_validate_audio_input(...)
    -> _process_audio_input(...)
       -> Qwen3ASRAudioEncoder.forward(...)
  Qwen3ASRForConditionalGeneration.embed_input_ids(...)
    -> _merge_multimodal_embeddings(...)
  Qwen3ASRForConditionalGeneration.forward(...)
    -> language_model.model(...)
    -> compute_logits(...)
```

要注意一个边界：

- `Qwen3ASRModel.streaming_transcribe()` 负责“流式状态机”和“什么时候解码”
- `qwen_asr/core/vllm_backend/qwen3_asr.py` 负责“音频怎么变成 embedding 并注入文本模型”
- 浏览器和网关的 `committed/live` 只是展示层稳定化，不是模型本体能力

---

## 2. 前端这一层到底做了什么

前端文件：

`local_web/qwen3_asr_gateway.html`

### 2.1 开始流式识别

入口函数是：

- `startStreaming()`

它做了 5 件事：

1. 先调 `apiStreamStart()`，向后端申请一个新的流式会话。
2. 后端返回 `session_id` 和初始 state 信息。
3. 浏览器打开麦克风：`navigator.mediaDevices.getUserMedia(...)`
4. 创建 `AudioContext`、`MediaStreamSource`、`ScriptProcessor`
5. 在 `processor.onaudioprocess` 里持续接收音频帧

### 2.2 浏览器里的音频怎么变成上传 chunk

关键函数：

- `processor.onaudioprocess`
- `resampleLinear(input, srcSr, dstSr)`
- `concatFloat32(a, b)`
- `pumpStream()`

真实流程：

1. 浏览器音频回调给出一段原始输入帧。
2. `getChannelData(0)` 取单声道。
3. `resampleLinear(...)` 把浏览器当前采样率重采样到 `16kHz`。
4. 结果追加到前端内存缓冲 `buffer`。
5. `pumpStream()` 每次从 `buffer` 里取固定大小的块，调用 `apiStreamChunk(chunk)` 上传。

这里有一个很容易忽略、但非常关键的点：

- 前端常量 `CHUNK_MS = 500`
- 也就是浏览器每次上传 `500ms` 音频

但服务端默认不是每收到 `500ms` 就解码一次。

---

## 3. 网关层怎么接这些 chunk

网关文件：

`local_web/qwen3_asr_gateway.py`

启动脚本：

`local_web/start_qwen3_asr_gateway.ps1`

### 3.1 服务启动时干了什么

`start_qwen3_asr_gateway.ps1` 最终跑的是：

- `python /workspace/local_web/qwen3_asr_gateway.py ...`

网关 `main()` 里最关键的一句是：

- `asr = Qwen3ASRModel.LLM(**llm_kwargs)`

这说明：

1. 网关只加载一份模型
2. backend 是 `vllm`
3. streaming 走的是 `Qwen3ASRModel` 这层封装，而不是直接自己操作底层张量

### 3.2 会话是怎么创建的

接口：

- `POST /api/stream/start`

对应函数：

- `api_stream_start()`

它调用：

- `asr.init_streaming_state(...)`

这一步创建了一个 `ASRStreamingState`，里面最关键的字段有：

- `buffer`
  含义：还没凑够一个服务端 chunk 的尾部音频
- `audio_accum`
  含义：从流开始到当前为止的全部音频
- `prompt_raw`
  含义：固定的基础 prompt，不带动态 prefix
- `chunk_id`
  含义：当前已经完成过几次服务端解码
- `unfixed_chunk_num`
  含义：前几个 chunk 不复用历史文本
- `unfixed_token_num`
  含义：复用历史文本时，末尾要回滚多少 token

### 3.3 上传音频 chunk 时发生了什么

接口：

- `POST /api/stream/chunk?session_id=...`

对应函数：

- `api_stream_chunk()`

它做的事情很直接：

1. 从 query 参数取 `session_id`
2. 从 `SESSIONS` 里找到这个 session 的 state
3. 要求请求体必须是 `application/octet-stream`
4. `np.frombuffer(raw, dtype=np.float32).reshape(-1)` 把字节流还原成 `float32` PCM
5. 调 `asr.streaming_transcribe(wav, sess.state)`
6. 把更新后的 state 交给 `_stream_payload(...)`
7. 返回 `language / text / committed_text / live_text`

这里要再强调一次：

- 前端上传粒度是 `500ms`
- 但 `streaming_transcribe()` 内部是先塞进 `state.buffer`
- 只有 `buffer` 足够大，才会真的触发一次模型解码

---

## 4. 真正的 streaming 状态机在什么地方

核心文件：

`qwen_asr/inference/qwen3_asr.py`

核心类型和函数：

- `ASRStreamingState`
- `Qwen3ASRModel.init_streaming_state()`
- `Qwen3ASRModel.streaming_transcribe()`
- `Qwen3ASRModel.finish_streaming_transcribe()`

### 4.1 init_streaming_state()

这个函数负责初始化 streaming 需要的全部状态。

最重要的逻辑有三条：

1. streaming 只允许 `backend == "vllm"`
2. 把 `chunk_size_sec` 转成 `chunk_size_samples`
3. 提前把固定 prompt 做好，存进 `state.prompt_raw`

其中：

- `prompt_raw = self._build_text_prompt(context=context, force_language=force_language)`

这意味着：

- system/user 框架 prompt 不会每次重建
- 每一轮只是在 `prompt_raw` 后面再拼一个动态 `prefix`

### 4.2 streaming_transcribe() 的真实行为

这是整个 streaming 机制的核心。

函数入口：

- `Qwen3ASRModel.streaming_transcribe(self, pcm16k, state)`

#### 第一步：规范化输入

它先把输入变成：

- 一维
- `float32`
- 单声道

如果传的是 `int16`，会缩放到 `[-1, 1]`。

#### 第二步：把前端送来的小块先放进 `state.buffer`

```text
state.buffer = state.buffer + 新来的 pcm
```

此时还不一定解码。

#### 第三步：只要 `buffer` 足够长，就消费一个“服务端 chunk”

循环条件是：

```text
while state.buffer.shape[0] >= state.chunk_size_samples
```

也就是说，真正决定一次模型解码的，不是前端上传频率，而是：

- `state.chunk_size_samples`
- 它来自 `chunk_size_sec`

当前网关默认：

- 服务端 `chunk_size_sec = 1.0`

因此在默认配置下：

- 浏览器每传两次 `500ms`
- 服务端才真正解码一次 `1.0s`

#### 第四步：把这个服务端 chunk 追加进 `state.audio_accum`

这一步是理解整个 streaming 的关键。

代码逻辑是：

```text
state.audio_accum = 之前所有音频 + 当前新 chunk
```

注意这里不是只保留最近一个 chunk，也不是维护声学缓存。

它保存的是：

- 从音频开头到当前时刻为止的全部音频

#### 第五步：构造 prefix

这是减少字幕尾巴抖动的关键机制。

规则是：

1. 如果 `state.chunk_id < state.unfixed_chunk_num`
   - `prefix = ""`
   - 也就是前几个 chunk 不信任历史结果，不拿它当前缀

2. 否则
   - 用 tokenizer 对 `state._raw_decoded` 编码
   - 删掉最后 `unfixed_token_num` 个 token
   - 再 decode 回字符串
   - 如果 decode 出现 `\ufffd`，继续多删几个，直到 Unicode 边界安全

这一步的本质是：

- 历史结果不是全部当真
- 只保留前面比较稳定的一段
- 尾巴留给当前轮重新生成

#### 第六步：构造当前轮 prompt

```text
prompt = state.prompt_raw + prefix
```

所以每一轮模型看到的是：

- 固定 prompt 框架
- 加上一段“已经相对稳定的历史文本”
- 再配上“从头到当前为止的全部音频”

#### 第七步：真正调用模型

关键输入长这样：

```python
inp = {
    "prompt": prompt,
    "multi_modal_data": {
        "audio": [state.audio_accum]
    }
}
```

然后调用：

```python
outputs = self.model.generate([inp], sampling_params=self.sampling_params, use_tqdm=False)
```

这里要非常明确：

- 每一轮送进模型的不是“本轮新 chunk”
- 而是 `state.audio_accum`
- 也就是“全量累积音频”

这正是为什么我前面说它不是严格的隐藏状态增量流式 ASR。

#### 第八步：解析输出并更新状态

生成完成后：

1. `gen_text = outputs[0].outputs[0].text`
2. `state._raw_decoded = prefix + gen_text`
3. `parse_asr_output(state._raw_decoded, user_language=state.force_language)`
4. 更新：
   - `state.language`
   - `state.text`
   - `state.chunk_id += 1`

---

## 5. finish_streaming_transcribe() 在做什么

函数：

- `Qwen3ASRModel.finish_streaming_transcribe()`

它负责处理最后一个不足一个 chunk 的尾巴。

逻辑和 `streaming_transcribe()` 基本一致，只是：

1. 不再等 `buffer` 凑满
2. 直接把剩余 `tail` 拼进 `audio_accum`
3. 再做最后一次全量重解码
4. 更新最终 `language/text`

所以结尾不是“直接把尾巴拼接到现有字幕后面”，而是：

- 仍然走一次“全量音频 + prefix”生成

---

## 6. 模型输出的字符串是怎么被拆成语言和文本的

函数：

- `qwen_asr/inference/utils.py : parse_asr_output()`

Qwen3-ASR 的原始生成串不是 JSON，而是一个轻量文本协议，典型形式类似：

```text
language Chinese<asr_text>你好世界
```

`parse_asr_output()` 负责把它拆成：

- `language`
- `text`

它还会做两件清洗工作：

1. `detect_and_fix_repetitions(...)`
   - 去掉明显的循环重复输出
2. 如果用户已经强制指定语言
   - 就把整个生成串直接当作纯文本，不再解析 `language ... <asr_text>`

因此，`state._raw_decoded` 和 `state.text` 不是一回事：

- `_raw_decoded` 是“原始协议串”
- `text` 是“解析后、适合展示的识别文本”

---

## 7. 进入 vLLM 以后，音频是怎么变成模型可吃的输入

核心文件：

`qwen_asr/core/vllm_backend/qwen3_asr.py`

这里要分清两个层次：

1. `Qwen3ASRModel.streaming_transcribe()` 决定何时调用 `generate`
2. `qwen3_asr.py` 决定 `generate` 里的音频怎么进入模型

### 7.1 哪些是我们仓库给 vLLM 提供的“钩子”

最关键的类有：

- `Qwen3ASRProcessingInfo`
- `Qwen3ASRMultiModalDataParser`
- `Qwen3ASRMultiModalProcessor`
- `Qwen3ASRForConditionalGeneration`

可以把它们理解成：

- `ProcessingInfo`
  告诉 vLLM 应该加载哪个 HF config / processor / feature extractor
- `DataParser`
  负责把原始音频或预提取特征整理成多模态输入
- `MultiModalProcessor`
  负责 placeholder 扩写和字段布局
- `ForConditionalGeneration`
  负责音频 embedding 编码、和文本 embedding 合并、然后交给文本模型

### 7.2 placeholder 是在哪里被扩成 N 个位置的

关键函数：

- `Qwen3ASRMultiModalProcessor._get_prompt_updates()`

它会：

1. 取出音频真实长度
2. 调 `_get_feat_extract_output_lengths(...)`
3. 算出这条音频最终会产出多少个 `audio feature`
4. 返回一个 `PromptReplacement`
5. 把一个逻辑音频占位符扩成 `N` 个 `audio_token_id`

这一步非常关键，因为：

- 文本 token 序列必须提前给音频 embedding 预留 N 个位置
- 后面 embedding 合并时，长度必须一一对应

### 7.3 音频特征是在哪里编码出来的

关键链路：

- `Qwen3ASRForConditionalGeneration.embed_multimodal()`
- `Qwen3ASRForConditionalGeneration._process_audio_input()`
- `Qwen3ASRAudioEncoder.forward()`

其中：

1. `embed_multimodal()` 负责识别当前请求里有哪些 audio 输入
2. `_process_audio_input()` 负责：
   - 先算 `audio_output_lengths`
   - 再调用 `audio_tower`
   - 最后按每条音频的长度把总输出切回去
3. `Qwen3ASRAudioEncoder.forward()` 才是真正的音频塔

### 7.4 音频塔内部为什么能支持长音频

`Qwen3ASRAudioEncoder.forward()` 的核心不是“整段音频一次 attention”，而是：

1. 先按 `n_window * 2` 把 mel 特征切成 chunk
2. 每个 chunk 过 3 层 `stride=2` 的 `Conv2d`
3. 再把 padding 后的 chunk batch 压成 packed hidden states
4. 用 `cu_seqlens` 告诉 attention 每段边界
5. 在局部窗口里做音频 encoder attention

这说明两件事：

1. 同一套音频塔权重可以处理短前缀和长音频
2. “统一 streaming/offline”的结构基础在音频塔这里

但注意：

- 这是“模型结构支持 long-form / chunked audio”
- 不是“服务层 streaming 已经做成声学状态增量缓存”

### 7.5 音频 embedding 是怎么并进文本序列的

关键函数：

- `Qwen3ASRForConditionalGeneration.embed_input_ids()`

它做的是：

1. 先得到普通文本 embedding
2. 再调用 `_merge_multimodal_embeddings(...)`
3. 把音频 embedding 填到 placeholder 位置

这件事的思想，和 Transformers 后端里的：

- `Qwen3ASRProcessor.replace_multimodal_special_tokens()`
- `Qwen3ASRThinkerForConditionalGeneration.get_audio_features()`
- `Qwen3ASRThinkerForConditionalGeneration.forward()`
- `masked_scatter(...)`

是同一件事，只是 vLLM 路径换成了自己的 planner 和 merge 接口。

---

## 8. committed_text 和 live_text 不是模型直接输出的

这个非常重要。

前端页面看到的：

- `Committed History`
- `Live Tail`

不是模型直接生成两个字段。

它们来自网关函数：

- `local_web/qwen3_asr_gateway.py : _split_committed_and_live_text()`
- `local_web/qwen3_asr_gateway.py : _stream_payload()`

处理逻辑是：

1. 取 `state._raw_decoded`
2. 用 tokenizer 再编码一次
3. 把末尾 `unfixed_token_num` 个 token 视为“不稳定尾巴”
4. 前面部分作为 `stable_raw`
5. `stable_raw` 再经过 `parse_asr_output(...)`
6. 得到：
   - `committed_text`
   - `live_text`

因此：

- `committed_text` 是“后处理认为已经稳定的前缀”
- `live_text` 是“仍可能在下一轮被改写的尾巴”

这不是模型内部分别维护的两段，而是 UI/网关层的人为切分。

---

## 9. 为什么说当前仓库的 streaming 不是“严格增量流式”

因为它不满足“新音频到来后，只在旧隐藏状态基础上继续推进”的条件。

当前实现每轮都在做：

```text
新 chunk 到来
-> 拼进 state.audio_accum
-> 把从开头到现在的全部音频重新喂给模型
-> 带一个裁掉尾巴的 prefix prompt
-> 重新生成当前整段最新结果
```

所以它更准确的定义是：

- `re-decoding streaming`
- 或者
- `prefix-conditioned full-context streaming`

它的优点是：

1. 工程实现简单
2. 输出格式和离线一致
3. 不需要单独维护复杂的声学缓存结构

它的代价是：

1. 越到后面，每一轮重解码音频越长
2. 不是理论上最省算力的 streaming

---

## 10. 这个项目里真正“支撑 streaming 可行”的三层机制

### 第 1 层：浏览器分块上传

前端不断把音频切成小块上传，保证服务端可以边录边处理。

对应：

- `qwen3_asr_gateway.html`

### 第 2 层：高层状态机

`Qwen3ASRModel.streaming_transcribe()` 负责：

- buffer
- chunk 触发
- 累计音频
- prefix rollback
- 每轮重解码

对应：

- `qwen_asr/inference/qwen3_asr.py`

### 第 3 层：模型结构允许长音频 chunk 化编码

`Qwen3ASRAudioEncoder.forward()` 负责：

- chunk 化音频塔
- packed hidden states
- `cu_seqlens`

对应：

- `qwen_asr/core/vllm_backend/qwen3_asr.py`
- `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`

---

## 11. 你最该记住的 8 个关键函数

如果你要真正吃透当前 streaming 机制，先盯死下面 8 个函数：

1. `local_web/qwen3_asr_gateway.html : startStreaming`
2. `local_web/qwen3_asr_gateway.html : pumpStream`
3. `local_web/qwen3_asr_gateway.py : api_stream_chunk`
4. `qwen_asr/inference/qwen3_asr.py : init_streaming_state`
5. `qwen_asr/inference/qwen3_asr.py : streaming_transcribe`
6. `qwen_asr/inference/qwen3_asr.py : finish_streaming_transcribe`
7. `qwen_asr/core/vllm_backend/qwen3_asr.py : Qwen3ASRMultiModalProcessor._get_prompt_updates`
8. `qwen_asr/core/vllm_backend/qwen3_asr.py : Qwen3ASRAudioEncoder.forward`

这 8 个函数串起来，你就能把“前端音频是怎么一路变成字幕的”完整说出来。

---

## 12. 一句话复盘

Qwen3-ASR 在这个仓库里的 streaming，不是“新音频只做一次增量前向”，而是：

- 前端持续送 `float32 16k` 小块
- 服务端先缓冲到固定 chunk
- 把全部累计音频重新送进 vLLM
- 用回滚后的旧文本做 prefix
- 通过多模态 processor 把音频展开成 placeholder
- 音频塔产出连续 embedding
- embedding 注入文本模型
- 重新生成整段最新结果
- 网关再把结果切成 `committed_text + live_text`

如果把这条链压成一行，就是：

```text
麦克风 -> 重采样 -> HTTP chunk -> buffer -> audio_accum -> prefix rollback -> vLLM generate -> audio embeddings 注入 -> 文本生成 -> committed/live split
```
