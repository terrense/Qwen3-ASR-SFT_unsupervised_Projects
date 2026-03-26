## Qwen3-ASR 代码导读

这份文档不是 API 使用说明，而是“读源码路线图”。目标是帮你从 Python 结构、Qwen3-ASR 模型原理、训练/推理/流式/对齐这几条线，把整个仓库串起来。

### 1. 先看哪些文件

推荐阅读顺序：

1. [qwen_asr/__init__.py](./qwen_asr/__init__.py)
2. [qwen_asr/inference/utils.py](./qwen_asr/inference/utils.py)
3. [qwen_asr/inference/qwen3_asr.py](./qwen_asr/inference/qwen3_asr.py)
4. [qwen_asr/core/transformers_backend/processing_qwen3_asr.py](./qwen_asr/core/transformers_backend/processing_qwen3_asr.py)
5. [qwen_asr/core/transformers_backend/configuration_qwen3_asr.py](./qwen_asr/core/transformers_backend/configuration_qwen3_asr.py)
6. [qwen_asr/core/transformers_backend/modeling_qwen3_asr.py](./qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)
7. [qwen_asr/inference/qwen3_forced_aligner.py](./qwen_asr/inference/qwen3_forced_aligner.py)
8. [qwen_asr/core/vllm_backend/qwen3_asr.py](./qwen_asr/core/vllm_backend/qwen3_asr.py)
9. [finetuning/qwen3_asr_sft.py](./finetuning/qwen3_asr_sft.py)
10. [finetuning/qwen3_asr_sft_semisup.py](./finetuning/qwen3_asr_sft_semisup.py)

### 2. 这项目的核心结构

Qwen3-ASR 在这个仓库里可以理解成三层：

1. 输入与后处理层
文件：`qwen_asr/inference/utils.py`
职责：接收 URL、本地路径、base64、`(np.ndarray, sr)` 等多种音频输入，统一成模型接受的 16k 单声道波形；把模型输出的 `"language Chinese<asr_text>..."` 解析成结构化结果。

2. 高层推理包装层
文件：`qwen_asr/inference/qwen3_asr.py`
职责：对用户暴露统一的 `Qwen3ASRModel` 接口；负责长音频切块、batch 广播、选择 Transformers/vLLM 后端、合并 chunk 结果、接入 forced aligner。

3. 底层模型实现层
文件：`qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`
职责：真正定义“音频编码器 + 文本解码器 + 多模态拼接”的计算图。

### 3. 模型原理怎么落到代码里

Qwen3-ASR 在代码里的思路不是 CTC，而是“音频条件下的自回归文本生成”。

流程如下：

1. 音频先被转换成声学特征。
入口在 [qwen_asr/core/transformers_backend/processing_qwen3_asr.py](./qwen_asr/core/transformers_backend/processing_qwen3_asr.py)

2. 音频编码器把声学特征压缩成更短的 speech embeddings。
核心实现在 [qwen_asr/core/transformers_backend/modeling_qwen3_asr.py](./qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)
里面你会看到：
   - 卷积下采样
   - Audio Transformer encoder
   - 投影到文本侧 hidden size

3. 文本 prompt 里有音频占位 token。
模型不会直接“把音频和文字拼成字符串”，而是先构造带音频占位符的文本序列，再把占位 token 对应的 embedding 替换成音频编码结果。

4. 文本解码器自回归生成结果。
生成的文本可以包含：
   - 语言标记
   - `<asr_text>` 后面的转写文本

### 4. 为什么输出是 `language X<asr_text>...`

文件：`qwen_asr/inference/utils.py`

这不是随便设计的字符串格式，而是训练协议的一部分。

好处有两个：

1. 同一个 decoder 同时学“语言识别 + 文本转写”
2. 后处理只要解析一个轻量协议，不需要模型直接输出 JSON

所以在训练脚本里你会看到 target 也是这个格式，而不是单纯一句中文文本。

### 5. Python 语法上几个最重要的点

#### 5.1 `dataclass`

比如：

- [qwen_asr/inference/qwen3_asr.py](./qwen_asr/inference/qwen3_asr.py)
- [qwen_asr/cli/demo_streaming.py](./qwen_asr/cli/demo_streaming.py)

这里的 `@dataclass` 用来定义“主要装数据、不强调行为”的对象，例如：

- `ASRTranscription`
- `ASRStreamingState`
- `Session`

这类对象适合用 dataclass，因为字段清晰、初始化方便、可读性好。

#### 5.2 闭包

比如：

- [finetuning/qwen3_asr_sft.py](./finetuning/qwen3_asr_sft.py)
- [finetuning/qwen3_asr_sft_semisup.py](./finetuning/qwen3_asr_sft_semisup.py)

`make_preprocess_fn_prefix_only(processor)` 返回一个内部函数 `_preprocess`。  
这是典型闭包：把 `processor` 绑定进后续 `datasets.map()` 用的函数里。

好处是不用写全局变量，也不用为了一个参数定义复杂类。

#### 5.3 组合优于继承

`Qwen3ASRModel` 没有强行继承某个具体后端模型类，而是把底层模型对象作为成员保存。

这属于典型的 Python 组合式设计：

- 对外 API 稳定
- 对内可以切换 Transformers / vLLM
- 上层业务代码不用知道底层差异

#### 5.4 延迟导入

比如：

- [finetuning/generate_pseudo_labels.py](./finetuning/generate_pseudo_labels.py)
- [finetuning/qwen3_asr_sft_semisup.py](./finetuning/qwen3_asr_sft_semisup.py)

有些导入被放到函数内部，是为了让 `--help`、简单数据脚本、部分未安装环境也能先工作。

这不是“偷懒”，而是 CLI 工具常见的工程实践。

### 6. 推理链路怎么走

入口：
[qwen_asr/inference/qwen3_asr.py](./qwen_asr/inference/qwen3_asr.py)

建议重点看这些函数：

1. `from_pretrained`
2. `LLM`
3. `transcribe`
4. `_build_text_prompt`
5. `_infer_asr_transformers`
6. `_infer_asr_vllm`

理解顺序：

1. `transcribe()` 接收用户输入
2. 调用 `normalize_audios()` 统一音频格式
3. 长音频按 chunk 切开
4. 构造带音频占位的 prompt
5. 调用具体后端
6. 用 `parse_asr_output()` 把原始字符串解析为 `(language, text)`
7. 把多个 chunk 的结果重新拼回来

### 7. Streaming 为什么单独一套

文件：

- [qwen_asr/inference/qwen3_asr.py](./qwen_asr/inference/qwen3_asr.py)
- [qwen_asr/cli/demo_streaming.py](./qwen_asr/cli/demo_streaming.py)
- [examples/example_qwen3_asr_vllm_streaming.py](./examples/example_qwen3_asr_vllm_streaming.py)

Streaming 的难点不是“分批喂音频”这么简单，而是：

1. 需要保留历史状态
2. 要处理 chunk 边界抖动
3. 不同 chunk 的 partial hypothesis 会反复修正

所以你会看到：

- `init_streaming_state`
- `streaming_transcribe`
- `finish_streaming_transcribe`

其中 `unfixed_chunk_num` 和 `unfixed_token_num` 本质上是在做边界回滚，减少流式 ASR 常见的尾部闪烁问题。

### 8. Forced aligner 和主 ASR 模型是什么关系

主 ASR 模型负责“识别文本”，forced aligner 负责“给文本对齐时间”。

文件：
[qwen_asr/inference/qwen3_forced_aligner.py](./qwen_asr/inference/qwen3_forced_aligner.py)

这意味着：

1. ASR 可以单独跑
2. 时间戳是可选增强能力
3. 某些场景下先识别、再对齐，比强行让同一个模型同时做两件事更稳定

### 9. 训练脚本怎么理解

文件：
[finetuning/qwen3_asr_sft.py](./finetuning/qwen3_asr_sft.py)

这个脚本的关键点不是“普通 Trainer 微调”，而是它怎么把多模态输入和 decoder-only loss 对齐起来。

重点看：

1. `patch_outer_forward`
2. `make_preprocess_fn_prefix_only`
3. `DataCollatorForQwen3ASRFinetuning`
4. `TrainingArguments`

真正重要的训练原则：

1. prompt 部分不算 loss
2. 只有 `<asr_text>` 后面的目标文本算 loss
3. padding token 也不算 loss

所以 collator 会把 prefix 和 pad 的 label 设成 `-100`。

### 10. 半监督扩展怎么理解

文件：
[finetuning/qwen3_asr_sft_semisup.py](./finetuning/qwen3_asr_sft_semisup.py)

这个文件没有改模型结构，只改了“数据组织方式”和“loss 归约方式”。

关键思想：

1. 真标注和伪标签都走同一个输入格式
2. 每条样本多一个 `loss_weight`
3. 训练时先算每个样本的 token loss，再做样本级加权平均

这特别适合方言场景，因为：

- 标注数据少但质量高
- 无标注数据多但质量参差不齐

### 11. 如果你想继续深读，应该盯哪些代码点

#### 想看音频怎么进模型
看：
[qwen_asr/core/transformers_backend/processing_qwen3_asr.py](./qwen_asr/core/transformers_backend/processing_qwen3_asr.py)

#### 想看音频 token 怎样替换文本占位
看：
[qwen_asr/core/transformers_backend/modeling_qwen3_asr.py](./qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)

搜这些关键词：

- `get_audio_features`
- `get_placeholder_mask`
- `masked_scatter`

#### 想看 vLLM 适配层
看：
[qwen_asr/core/vllm_backend/qwen3_asr.py](./qwen_asr/core/vllm_backend/qwen3_asr.py)

#### 想看高层 API 是怎么把这些复杂性藏起来的
看：
[qwen_asr/inference/qwen3_asr.py](./qwen_asr/inference/qwen3_asr.py)

### 12. 怎么继续做“带注释读源码”

如果你下一步想更细，我建议按文件分批做，而不是继续全仓库一起加：

1. 先精读 [qwen_asr/inference/qwen3_asr.py](./qwen_asr/inference/qwen3_asr.py)
2. 再精读 [qwen_asr/core/transformers_backend/modeling_qwen3_asr.py](./qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)
3. 最后看训练脚本 [finetuning/qwen3_asr_sft.py](./finetuning/qwen3_asr_sft.py)

如果你要，我下一轮可以直接继续做一件更细的事：

- 把 `qwen_asr/inference/qwen3_asr.py` 逐段加中文专业注释
- 或把 `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py` 逐模块讲透
