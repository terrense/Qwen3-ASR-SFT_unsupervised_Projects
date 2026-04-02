# Qwen3-ASR vLLM Backend 学习与接口总结

## 1. 先把结论说清

如果你现在最困惑的是“`vLLM backend` 和 `Transformers backend` 到底差在哪”，先记这一句：

> 两个 backend 用的是同一套 Qwen-ASR 模型思路，但承担的职责不一样。  
> `Transformers backend` 更像“我们自己手工把张量准备好，再调用 Hugging Face 模型”。  
> `vLLM backend` 更像“我们把模型适配进 vLLM 的运行时，让 vLLM 接管请求调度、批处理、KV cache 和生成执行”。

所以：

- `Transformers` 重点是“模型怎么计算”
- `vLLM` 重点是“模型怎么作为服务高效跑起来”

---

## 2. 先看三个总入口

高层推理包装：

- [qwen_asr/inference/qwen3_asr.py](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/inference/qwen3_asr.py)
- [Qwen3ASRModel](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/inference/qwen3_asr.py#L159)
- [_infer_asr_transformers](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/inference/qwen3_asr.py#L559)
- [_infer_asr_vllm](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/inference/qwen3_asr.py#L598)

两个 backend 的分工，在这里最清楚：

- `Transformers` 路径：`processor(...) -> model.generate(...)`
- `vLLM` 路径：`prompt + multi_modal_data -> vLLM generate(...)`

---

## 3. 两个 backend 的真正区别

### 3.1 它们共同做的事

不管走哪条后端，核心任务都一样：

1. 把原始音频变成 mel 特征
2. 算出音频最终会对应多少个 placeholder 位置
3. 把音频编码成连续 `audio embeddings`
4. 把这些 embedding 填进文本序列里的音频占位位置
5. 继续走 Qwen 文本生成

也就是说，**模型语义是一套**，不是两套。

### 3.2 Transformers backend 在做什么

Transformers 路径主要文件：

- [qwen_asr/core/transformers_backend/processing_qwen3_asr.py](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/transformers_backend/processing_qwen3_asr.py)
- [qwen_asr/core/transformers_backend/modeling_qwen3_asr.py](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)

关键入口：

- [Qwen3ASRProcessor.__call__](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/transformers_backend/processing_qwen3_asr.py#L137)
- [replace_multimodal_special_tokens](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/transformers_backend/processing_qwen3_asr.py#L230)
- [get_audio_features](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py#L1412)
- [get_placeholder_mask](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py#L1460)
- [Qwen3ASRThinkerForConditionalGeneration.forward](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py#L1490)

这条路的风格是：

- 先在 Python 侧把 `input_ids / input_features / attention_mask / feature_attention_mask` 全准备好
- placeholder 扩写也是我们自己在 processor 里做
- 进模型后，用 `masked_scatter` 这类操作把音频 embedding 填回文本 embedding
- 最后调用 Hugging Face 的 `generate()`

一句话：

> `Transformers backend` 是“显式拼张量”的路线。

### 3.3 vLLM backend 在做什么

vLLM 路径主要文件：

- [qwen_asr/core/vllm_backend/qwen3_asr.py](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py)

关键入口：

- [_get_feat_extract_output_lengths](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L136)
- [Qwen3ASRAudioEncoder](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L344)
- [Qwen3ASRProcessingInfo](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L607)
- [Qwen3ASRMultiModalProcessor](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L717)
- [_get_prompt_updates](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L740)
- [Qwen3ASRForConditionalGeneration](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L803)
- [_process_audio_input](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L902)
- [embed_multimodal](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L927)
- [embed_input_ids](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L954)

这条路的风格是：

- 外部调用时，不是先手搓一大堆张量
- 而是把请求交给 vLLM：

```python
{
    "prompt": prompt,
    "multi_modal_data": {"audio": [wav]}
}
```

- 然后由 vLLM 再回调我们注册进去的 processor / model hook
- 我们负责告诉 vLLM：
  - 音频要怎么预处理
  - placeholder 要怎么扩
  - 音频 embedding 要怎么并进文本 embedding
- vLLM 负责：
  - 请求调度
  - continuous batching
  - prefix caching / KV cache
  - 生成执行

一句话：

> `vLLM backend` 是“把模型接进推理引擎运行时”的路线。

---

## 4. 最重要的差异，不在数学，在职责边界

### 4.1 Transformers 的边界

你可以把它理解成：

```text
我们自己
-> 处理输入
-> 组织张量
-> 组织多模态位置
-> 调 HF 模型
```

也就是：

- 易读
- 更接近论文和原始 PyTorch 逻辑
- 对“怎么从音频变成 embedding”更直观
- 但服务化调度能力不是它的重点

### 4.2 vLLM 的边界

你可以把它理解成：

```text
我们负责教 vLLM 认识这个模型
-> vLLM 负责把这个模型高效跑起来
```

也就是：

- 代码里会多很多“胶水层”
- 这些代码不是在重新发明模型数学
- 而是在告诉 vLLM：
  - 多模态输入长什么样
  - prompt 怎样更新
  - encoder 输出怎样缓存
  - embedding 怎样喂给统一执行器

所以 `qwen_asr/core/vllm_backend/qwen3_asr.py` 看起来更“工程”，不如 Transformers 路径那样直给。

---

## 5. 你应该怎么学 `vllm_backend/qwen3_asr.py`

不要一行一行啃。按这 4 步走。

### 第一步：只看“音频占位符怎么扩”

看这里：

- [Qwen3ASRProcessingInfo](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L607)
- [Qwen3ASRMultiModalProcessor](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L717)
- [_get_prompt_updates](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L740)

这一层只回答一个问题：

> 为什么一个逻辑上的 `<audio>`，最后会在 prompt 里变成 N 个 placeholder 位置？

这是因为 audio encoder 最终会输出 N 个时间步 embedding。  
placeholder 数必须和 embedding 数一一对应。

### 第二步：再看“音频怎么编码”

看这里：

- [_get_feat_extract_output_lengths](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L136)
- [Qwen3ASRAudioEncoder](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L344)
- [Qwen3ASRAudioEncoder.forward](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L446)

这一层只回答两个问题：

1. mel 特征经过卷积后长度怎么变
2. audio tower 怎么把它变成连续 embedding

### 第三步：再看“embedding 怎么并进文本模型”

看这里：

- [_process_audio_input](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L902)
- [embed_multimodal](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L927)
- [embed_input_ids](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L954)
- [Qwen3ASRForConditionalGeneration.forward](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L985)

这一层要吃透一句话：

> Qwen-ASR 不是“音频先变离散 speech token id 再喂给 LLM”，而是“音频变成连续 embedding，然后覆盖文本里的音频占位位置”。

### 第四步：最后看 vLLM 特有接口

比如这些：

- [MULTIMODAL_REGISTRY](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L84)
- [@MULTIMODAL_REGISTRY.register_processor(...)](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L798)
- [get_generation_prompt](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L1125)

这些不是“模型数学”，而是“如何接进 vLLM”。

---

## 6. 为什么当前 streaming 只支持 vLLM

高层 streaming 状态机在这里：

- [init_streaming_state](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/inference/qwen3_asr.py#L668)
- [streaming_transcribe](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/inference/qwen3_asr.py#L743)

当前仓库的 streaming 本质是：

```text
新 chunk
-> 追加到 audio_accum
-> 回滚一点旧前缀
-> 再发起一次新的 generate(prompt + audio_accum)
```

也就是：

> `累计音频 + 全量重解码 + prefix rollback`

这套服务状态机目前只包了 vLLM 路径，所以 streaming API 只支持 vLLM，不支持 Transformers。

---

## 7. 现在这套网关到底暴露了哪些 HTTP 接口

网关文件：

- [local_web/qwen3_asr_gateway.py](/f:/Qwen_codes/Qwen3-ASR-main/local_web/qwen3_asr_gateway.py)

### 7.1 页面与状态接口

- [GET /](/f:/Qwen_codes/Qwen3-ASR-main/local_web/qwen3_asr_gateway.py#L154)
- [GET /healthz](/f:/Qwen_codes/Qwen3-ASR-main/local_web/qwen3_asr_gateway.py#L161)
- [GET /api/info](/f:/Qwen_codes/Qwen3-ASR-main/local_web/qwen3_asr_gateway.py#L167)

### 7.2 一次性转写接口

- [POST /api/transcribe](/f:/Qwen_codes/Qwen3-ASR-main/local_web/qwen3_asr_gateway.py#L193)

### 7.3 流式转写接口

- [POST /api/stream/start](/f:/Qwen_codes/Qwen3-ASR-main/local_web/qwen3_asr_gateway.py#L228)
- [POST /api/stream/chunk](/f:/Qwen_codes/Qwen3-ASR-main/local_web/qwen3_asr_gateway.py#L253)
- [POST /api/stream/finish](/f:/Qwen_codes/Qwen3-ASR-main/local_web/qwen3_asr_gateway.py#L280)

所以现在这套网关对外一共 7 个路由。

如果你从别的项目调用，基地址就是你当前容器映射出来的：

`http://localhost:8003`

---

## 8. 最后用一句话收尾

如果你只想记住最关键的区别，就记这两句：

- `Transformers backend`：重点是“模型怎么计算”，代码更像标准 PyTorch / HF 多模态实现
- `vLLM backend`：重点是“模型怎么作为高性能服务运行”，代码里大量内容是运行时接入和调度胶水

这就是为什么：

- 学模型结构时，先看 `transformers_backend`
- 学服务化、streaming、在线推理时，必须看 `vllm_backend`
