# Qwen3-ASR vLLM Backend 学习路线与接口盘点

## 1. 这份文档解决两个问题

1. `qwen_asr/core/vllm_backend/qwen3_asr.py` 这么长，到底应该怎么学。
2. 现在这套 `qwen3-asr-gateway` 容器服务一共暴露了哪些 HTTP 接口，别的项目怎么调。

这份总结默认针对你当前这套服务形态：

- 容器名：`qwen3-asr-gateway`
- 镜像：`qwenllm/qwen3-asr:latest`
- 端口映射：`8003:8000`

也就是：

- 容器内服务地址：`http://0.0.0.0:8000`
- 你本机访问地址：`http://localhost:8003`

---

## 2. 先用一句话说清 `vllm_backend/qwen3_asr.py`

文件：

[qwen3_asr.py](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py)

一句话概括：

> 这不是“又写了一套新模型”，而是把 Qwen-ASR 的音频塔、placeholder 扩写、音频 embedding 注入、mRoPE 位置编号，全部改造成 vLLM 能调度的运行时版本。

你可以把它理解成：

- Transformers 版：我们自己把张量都准备好，再喂给 HF 模型
- vLLM 版：我们告诉 vLLM“音频怎么进来、怎么扩 placeholder、怎么变 embedding”，其余调度、批处理、KV cache、生成执行由 vLLM 引擎接管

---

## 3. 不要一行一行读，先按四层框架学

学习这份文件，推荐按下面 4 层走。

### 第 1 层：先认清它在整个系统里的位置

先看这些入口：

- [qwen3_asr.py#L268](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/inference/qwen3_asr.py#L268)
- [qwen3_asr.py#L598](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/inference/qwen3_asr.py#L598)
- [qwen3_asr.py#L803](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L803)

这一层只回答一个问题：

> 这份 `qwen3_asr.py` 是怎么被真正调用起来的？

答案是：

1. 高层包装 `Qwen3ASRModel.LLM(...)` 创建 vLLM 引擎
2. 请求通过 `self.model.generate(...)` 进入 vLLM
3. vLLM 再回调我们注册进去的多模态处理器和模型钩子

这一层先别管张量细节，只要建立“谁调谁”的脑图。

### 第 2 层：先学音频怎么进来

重点看：

- [qwen3_asr.py#L607](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L607)
- [qwen3_asr.py#L717](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L717)
- [qwen3_asr.py#L740](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L740)

这一层只看：

- 音频输入先被谁解析
- 一个逻辑 `<audio>` 占位符怎么扩成 N 个位置
- 这个 N 为什么必须和 audio encoder 输出长度一致

### 第 3 层：再学音频怎么变成 embedding

重点看：

- [qwen3_asr.py#L344](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L344)
- [qwen3_asr.py#L446](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L446)
- [qwen3_asr.py#L902](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L902)
- [qwen3_asr.py#L927](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L927)
- [qwen3_asr.py#L954](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L954)

这一层回答：

- 音频塔怎么复刻 Transformers 版
- 为什么要切 chunk、卷积下采样、packed hidden states、`cu_seqlens`
- 音频 embedding 怎么塞进文本 embedding 序列

### 第 4 层：最后再补位置编码、权重映射、服务 prompt

重点看：

- [qwen3_asr.py#L985](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L985)
- [qwen3_asr.py#L1040](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L1040)
- [qwen3_asr.py#L1125](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L1125)

这一层是“补全认知”，不是第一遍必须看懂。

---

## 4. 这份文件的真正骨架

如果把整份文件压缩成骨架，只剩 5 个核心块：

### A. 长度公式

- [qwen3_asr.py#L112](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L112)

作用：

- 保证 vLLM 路径和 Transformers 路径对“音频最终输出长度”的理解完全一致

这一步如果错了，后面 placeholder 数和 audio embedding 数一定对不上。

### B. 音频塔

- [qwen3_asr.py#L344](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L344)
- [qwen3_asr.py#L446](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L446)

作用：

- 把 mel 特征编码成连续 `audio embeddings`

这块是“模型数学”最重的部分。

### C. vLLM 多模态胶水层

- [qwen3_asr.py#L607](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L607)
- [qwen3_asr.py#L717](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L717)
- [qwen3_asr.py#L740](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L740)

作用：

- 把“原始音频 + 文本 prompt”翻译成 vLLM 能调度的多模态输入

### D. 顶层模型壳

- [qwen3_asr.py#L803](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L803)
- [qwen3_asr.py#L902](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L902)
- [qwen3_asr.py#L927](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L927)
- [qwen3_asr.py#L954](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L954)

作用：

- 接收 audio feature
- 跑 audio tower
- 和文本 embedding 合并
- 交给 Qwen3 文本模型继续生成

### E. 服务 prompt 适配

- [qwen3_asr.py#L1125](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L1125)

作用：

- 给服务层生成最薄的一层 prompt 外壳

---

## 5. 学这份文件的推荐顺序

如果你现在就开始啃，我建议按下面顺序。

### 第一步：只看 6 个符号

第一遍只看这 6 个位置：

- [qwen3_asr.py#L344](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L344)
- [qwen3_asr.py#L446](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L446)
- [qwen3_asr.py#L717](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L717)
- [qwen3_asr.py#L740](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L740)
- [qwen3_asr.py#L927](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L927)
- [qwen3_asr.py#L954](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L954)

只回答三件事：

1. 音频哪里编码
2. placeholder 哪里扩写
3. embedding 哪里合并

### 第二步：再去对照 Transformers 后端

对照看：

- [processing_qwen3_asr.py#L137](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/transformers_backend/processing_qwen3_asr.py#L137)
- [processing_qwen3_asr.py#L230](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/transformers_backend/processing_qwen3_asr.py#L230)
- [modeling_qwen3_asr.py#L1412](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py#L1412)
- [modeling_qwen3_asr.py#L1460](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py#L1460)
- [modeling_qwen3_asr.py#L1490](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py#L1490)

你会发现两条路本质上在做同一件事：

- 都要算音频长度
- 都要扩 placeholder
- 都要把音频 embedding 注进文本序列

只是：

- Transformers 版是我们自己显式拼张量
- vLLM 版是通过 registry / processor / runtime hook 拼进去

### 第三步：最后再学 vLLM 独有的东西

比如：

- `MULTIMODAL_REGISTRY`
- `BaseProcessingInfo`
- `PromptReplacement`
- `MultiModalFieldConfig`
- `SupportsMultiModal`
- `SupportsMRoPE`

这些都是 vLLM 世界里的“运行时接口语言”。

---

## 6. vLLM 原理，用这份代码能看懂的版本

你现在不用先学 vLLM 官方大而全的理论，只要理解它在这份代码里扮演什么角色。

### 6.1 vLLM 不要求你先把所有张量都拼好

Transformers 路径里，我们通常这样：

1. 先 `processor(text, audio, return_tensors="pt")`
2. 得到完整 `input_ids / input_features / masks`
3. 再喂给模型 `generate`

对应：

- [qwen3_asr.py#L559](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/inference/qwen3_asr.py#L559)

vLLM 路径里，我们只交一个更抽象的 request：

```python
{
    "prompt": prompt,
    "multi_modal_data": {"audio": [wav]}
}
```

对应：

- [qwen3_asr.py#L598](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/inference/qwen3_asr.py#L598)

然后由 vLLM 再回调我们注册进去的多模态处理器和模型钩子。

### 6.2 vLLM 关心的是“运行时调度”

这份代码里，vLLM 最核心的作用不是改数学，而是接管：

- 请求组织
- 多模态输入规划
- 生成调度
- KV cache 管理
- 高并发服务执行

### 6.3 这份文件里你能直接看到的 vLLM 思想

#### 思想 1：多模态 registry

- [qwen3_asr.py#L717](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L717)
- [qwen3_asr.py#L803](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L803)

意思是：

> 模型本身不只是一堆层，它还要向 vLLM 声明“我支持哪些模态、这些模态怎么解析、怎么放进 prompt、怎么映射到 embedding”。

#### 思想 2：prompt 规划和 embedding 合并分离

- [qwen3_asr.py#L740](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L740)
- [qwen3_asr.py#L927](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L927)
- [qwen3_asr.py#L954](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L954)

意思是：

1. 先决定文本序列里哪些位置要给音频留坑
2. 再单独把音频编码成 embedding
3. 最后再合并

#### 思想 3：底层执行器与模型定义分离

音频塔和顶层模型虽然是我们写的，但真正怎么批处理、怎么跑生成、怎么管理 KV cache，不在这份文件里，而在 vLLM 引擎内部。

也就是说：

- 这份文件负责“接入协议”
- vLLM 框架负责“高性能执行”

---

## 7. Transformers backend 和 vLLM backend 的最重要区别

### 7.1 相同点

两条路做的核心数学其实是同一件事：

1. 音频 -> mel 特征
2. 根据长度公式扩 placeholder
3. 音频塔 -> 连续 `audio embeddings`
4. 把这些 embedding 注入文本序列
5. 文本模型继续生成识别结果

### 7.2 不同点

#### Transformers 路径

主要文件：

- [processing_qwen3_asr.py](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/transformers_backend/processing_qwen3_asr.py)
- [modeling_qwen3_asr.py](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)

特点：

- 我们自己先构造 `BatchFeature`
- 自己显式处理 `input_ids / input_features / masks`
- 模型内部用 `masked_scatter` 做音频 embedding 注入
- 更好读源码、更适合学习

#### vLLM 路径

主要文件：

- [qwen3_asr.py](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py)

特点：

- 输入更抽象：`prompt + multi_modal_data`
- 多模态处理通过 vLLM registry / processor hook 完成
- 更适合服务化和高并发
- 当前仓库里的 streaming 只支持它

### 7.3 一句话对比

Transformers：

> 我们自己把饭做好，再端给模型。

vLLM：

> 我们把食材和做法说明交给餐厅流水线，厨房自己高效出餐。

---

## 8. 现在这套 `qwen3-asr-gateway` 一共有多少个接口

当前服务文件：

[qwen3_asr_gateway.py](/f:/Qwen_codes/Qwen3-ASR-main/local_web/qwen3_asr_gateway.py)

### 8.1 如果按“全部路由”来算

一共 **7 个路由**：

1. `GET /`
2. `GET /healthz`
3. `GET /api/info`
4. `POST|OPTIONS /api/transcribe`
5. `POST|OPTIONS /api/stream/start`
6. `POST|OPTIONS /api/stream/chunk`
7. `POST|OPTIONS /api/stream/finish`

### 8.2 如果按“你在别的项目里最常调用的 API 接口”来算

一共 **5 个可调用接口**：

1. `/healthz`
2. `/api/info`
3. `/api/transcribe`
4. `/api/stream/start`
5. `/api/stream/chunk`
6. `/api/stream/finish`

严格说这是 6 个，其中：

- `/healthz` 和 `/api/info` 更偏“服务探活/能力发现”
- 真正业务接口是 4 个：
  - `/api/transcribe`
  - `/api/stream/start`
  - `/api/stream/chunk`
  - `/api/stream/finish`

---

## 9. 每个接口的作用与调用方式

假设你的容器映射是 `8003:8000`，那么对外基地址就是：

`http://localhost:8003`

### 9.1 `GET /`

完整地址：

`http://localhost:8003/`

作用：

- 打开测试前端网页

返回：

- HTML 页面

### 9.2 `GET /healthz`

完整地址：

`http://localhost:8003/healthz`

作用：

- 健康检查
- 最适合判断服务有没有真的起来

返回示例：

```json
{"ok": true}
```

### 9.3 `GET /api/info`

完整地址：

`http://localhost:8003/api/info`

作用：

- 查询服务能力
- 查询当前模型名、后端类型、各接口路径

返回示例重点字段：

```json
{
  "service": "qwen3-asr-gateway",
  "backend": "vllm",
  "capabilities": {
    "transcribe": true,
    "streaming": true,
    "timestamps": false,
    "committed_live_split": true
  },
  "routes": {
    "transcribe": "/api/transcribe",
    "stream_start": "/api/stream/start",
    "stream_chunk": "/api/stream/chunk",
    "stream_finish": "/api/stream/finish"
  }
}
```

### 9.4 `POST /api/transcribe`

完整地址：

`http://localhost:8003/api/transcribe`

作用：

- 一次性离线转录

请求体：

```json
{
  "audio_data_url": "...",
  "context": "可选上下文",
  "language": "可选强制语言"
}
```

说明：

- `audio_data_url` 必填
- 也兼容 `audio_url`
- 这里不是传原始二进制，而是传 data URL / URL 字符串

返回：

```json
{
  "mode": "transcribe",
  "language": "Chinese",
  "text": "你好世界",
  "request_ms": 1234
}
```

### 9.5 `POST /api/stream/start`

完整地址：

`http://localhost:8003/api/stream/start`

作用：

- 创建一个新的 streaming session

请求体：

```json
{
  "context": "可选上下文",
  "language": "可选强制语言"
}
```

返回重点字段：

```json
{
  "session_id": "....",
  "language": "",
  "text": "",
  "committed_text": "",
  "live_text": "",
  "chunk_id": 0
}
```

后续所有流式 chunk 都要带这个 `session_id`。

### 9.6 `POST /api/stream/chunk`

完整地址：

`http://localhost:8003/api/stream/chunk?session_id=...`

作用：

- 上传一段 `float32` 单声道 PCM chunk
- 返回当前最新 streaming 假设

请求头：

```text
Content-Type: application/octet-stream
```

请求体：

- 原始二进制
- 内容必须是 `float32` PCM 字节流

返回字段：

```json
{
  "language": "Chinese",
  "text": "完整当前文本",
  "committed_text": "稳定前缀",
  "live_text": "仍可能变化的尾巴",
  "chunk_id": 3,
  "unfixed_chunk_num": 4,
  "unfixed_token_num": 5
}
```

### 9.7 `POST /api/stream/finish`

完整地址：

`http://localhost:8003/api/stream/finish?session_id=...`

作用：

- 告诉后端“流结束了”
- 把最后不足一个完整 chunk 的尾巴也刷进去
- 返回最终结果

返回：

```json
{
  "language": "Chinese",
  "text": "最终文本",
  "committed_text": "最终文本",
  "live_text": "",
  "chunk_id": 4,
  "unfixed_chunk_num": 4,
  "unfixed_token_num": 5
}
```

---

## 10. 别的项目怎么接这套服务

### 10.1 最简单的离线调用流程

```text
准备音频
-> 转成 data URL 或可访问 URL
-> POST /api/transcribe
-> 取回 language / text
```

### 10.2 最简单的 streaming 调用流程

```text
POST /api/stream/start
-> 得到 session_id
-> 循环 POST /api/stream/chunk?session_id=...
-> 每次取 committed_text / live_text
-> 结束时 POST /api/stream/finish?session_id=...
```

这说明：

> 是的，你完全可以在别的项目里把它当一个独立 HTTP ASR 服务来调用。

---

## 11. 需要特别记住的限制

### 11.1 当前这个统一网关后端是 vLLM

返回 `/api/info` 里会看到：

- `backend: "vllm"`

所以：

- 你现在这套容器服务不是 Transformers backend 服务
- 它走的是 vLLM 路径

### 11.2 当前服务不支持 timestamps

`/api/info` 里明确写了：

- `timestamps: false`

也就是说：

- `/api/transcribe` 目前只返回 `language + text`
- 不返回词级或句级时间戳

### 11.3 当前 streaming 接口依赖 session

`/api/stream/chunk` 和 `/api/stream/finish` 都要求：

- query 参数里有 `session_id`

### 11.4 当前 streaming 上传的不是 WAV 文件

它要的是：

- `float32`
- `16kHz`
- 单声道 PCM 二进制

不是直接传 `.wav` 文件头。

---

## 12. 你下一步最值得怎么学

我建议我们按下面 3 课走。

### 第 1 课：只学“音频怎么进 prompt”

看：

- [qwen3_asr.py#L717](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L717)
- [qwen3_asr.py#L740](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L740)

目标：

- 搞懂一个 audio 占位符为什么会扩成 N 个位置

### 第 2 课：只学“音频怎么变 embedding”

看：

- [qwen3_asr.py#L344](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L344)
- [qwen3_asr.py#L446](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L446)
- [qwen3_asr.py#L902](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L902)

目标：

- 搞懂 audio tower、chunk、卷积、packed hidden states、`cu_seqlens`

### 第 3 课：只学“embedding 怎么并入文本模型”

看：

- [qwen3_asr.py#L927](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L927)
- [qwen3_asr.py#L954](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L954)
- [qwen3_asr.py#L985](/f:/Qwen_codes/Qwen3-ASR-main/qwen_asr/core/vllm_backend/qwen3_asr.py#L985)

目标：

- 搞懂音频 embedding 最后怎么进入 Qwen3 文本模型

---

## 13. 最短复盘

如果你今天只记住 4 句话，就记这 4 句：

1. `vllm_backend/qwen3_asr.py` 不是另一套新模型，而是 Qwen-ASR 的 vLLM 运行时适配版。
2. 学它不要一行一行抠，先学“placeholder 扩写 -> audio tower -> embedding 合并”这三步。
3. 你现在这套 `qwen3-asr-gateway` 对外一共 7 个路由，其中 4 个是真正业务接口：`transcribe/start/chunk/finish`。
4. 这套服务完全可以当成独立 HTTP ASR 服务被别的项目调用。
