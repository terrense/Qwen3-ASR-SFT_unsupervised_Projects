# Ubuntu 服务器部署 Qwen3-ASR（vLLM 后端，局域网可访问）

本文档面向“**不使用 Docker**、直接在 Ubuntu 服务器上部署 vLLM 后端”的场景，目标是让局域网内其它项目通过 HTTP API 访问 Qwen3-ASR 服务。

> 说明
> - vLLM 对外提供的是 OpenAI 风格 API（常见是 `POST /v1/chat/completions`、`POST /v1/audio/transcriptions`）。
> - Qwen3-ASR 的“流式能力”在本仓库里也提供了可选的 Web 网关与分块接口（`/api/stream/*`），但该网关属于“额外服务层”。如果你只想先把 vLLM 拉起来并对内网提供标准 OpenAI 风格接口，按本文步骤即可。

---

## 0. 准备工作（先确认这些再开始）

### 0.1 硬件与系统建议

- **Ubuntu**：推荐 22.04 LTS / 24.04 LTS
- **NVIDIA GPU**：建议至少 12GB 显存
  - 首次部署建议先用 `Qwen/Qwen3-ASR-0.6B`（更省显存、更稳）
  - 显存足够再换 `Qwen/Qwen3-ASR-1.7B`
- **网络**：服务器与调用方在同一局域网，或已打通路由/端口转发

### 0.2 你需要准备的信息

- 服务器局域网 IP（例如 `192.168.1.10`）
- 计划开放的端口（本文用 `8000`）
- 模型选择：
  - `Qwen/Qwen3-ASR-0.6B` 或
  - `Qwen/Qwen3-ASR-1.7B`

---

## 1. 检查 NVIDIA 驱动与 GPU 可用性

在服务器上执行：

```bash
nvidia-smi
```

你应该能看到 GPU 型号、驱动版本、显存信息。如果 `nvidia-smi` 不存在或报错，需要先安装/修复 NVIDIA 驱动（不同机房/镜像安装方式不同，建议按你机器/云厂商的 GPU 驱动指引完成）。

---

## 2. 安装系统依赖与创建 Python 虚拟环境

```bash
sudo apt update
sudo apt install -y git curl wget build-essential python3 python3-venv python3-pip
```

创建并进入虚拟环境：

```bash
python3 -m venv ~/venvs/qwen3asr-vllm
source ~/venvs/qwen3asr-vllm/bin/activate
python -m pip install -U pip wheel setuptools
```

---

## 3. 安装 vLLM（含音频依赖）

### 3.1 推荐：使用 vLLM 的官方 wheel（按你的 CUDA 匹配）

vLLM 的安装与 **CUDA/驱动环境强相关**。最稳妥的做法是按 vLLM 官方文档选择与你机器匹配的 wheel。

下面给出一个“nightly + CUDA 12.9（cu129）”示例（如果你的环境不是 cu129，请替换成匹配的版本）：

```bash
source ~/venvs/qwen3asr-vllm/bin/activate

pip install -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match

pip install -U "vllm[audio]"
```

### 3.2 可选：安装本仓库的 Python 包入口（仅用于本地代码/工具链）

如果你还希望用 `qwen-asr` 包内的工具函数（例如 `parse_asr_output`）：

```bash
pip install -U "qwen-asr[vllm]"
```

> 说明：即使你不装 `qwen-asr`，只用 vLLM 也可以提供 `OpenAI /v1/...` API；`qwen-asr` 更偏向“库 + 示例 + 额外工具”。

---

## 4. （强烈建议）提前下载模型到本地目录

如果你的服务器运行时网络不稳定（或不允许在线下载），建议先把模型拉到本地目录，再用本地路径启动 vLLM。

目标路径示例：`~/models/Qwen3-ASR-0.6B`

### 4.1 方式 A：Hugging Face CLI

```bash
source ~/venvs/qwen3asr-vllm/bin/activate
pip install -U "huggingface_hub[cli]"

mkdir -p ~/models/Qwen3-ASR-0.6B
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir ~/models/Qwen3-ASR-0.6B
```

如需 1.7B，把 `0.6B` 替换为 `1.7B`。

### 4.2 方式 B：ModelScope（国内网络更友好）

```bash
source ~/venvs/qwen3asr-vllm/bin/activate
pip install -U modelscope

mkdir -p ~/models/Qwen3-ASR-0.6B
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ~/models/Qwen3-ASR-0.6B
```

---

## 5. 启动 vLLM 服务（对局域网开放）

### 5.1 最小启动（推荐先用 0.6B 验证）

#### 直接用模型名（会在运行时自动下载）

```bash
source ~/venvs/qwen3asr-vllm/bin/activate
vllm serve Qwen/Qwen3-ASR-0.6B --host 0.0.0.0 --port 8000
```

#### 用本地模型目录（推荐，更可控）

```bash
source ~/venvs/qwen3asr-vllm/bin/activate
vllm serve ~/models/Qwen3-ASR-0.6B --host 0.0.0.0 --port 8000
```

> 关键点：一定要带 `--host 0.0.0.0`，否则只监听 `127.0.0.1`，局域网其它机器访问不到。

### 5.2 常用调参（OOM/显存紧张时）

以下参数经常用于降低显存压力/提升稳定性（按需添加）：

- `--gpu-memory-utilization 0.8`：限制显存占用比例（常用 0.7~0.9）
- `--max-model-len <N>`：降低上下文长度节省显存（例如 `24576` 或更小）

示例：

```bash
vllm serve ~/models/Qwen3-ASR-0.6B \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.8
```

---

## 6. 放通端口（UFW / 安全组）

### 6.1 Ubuntu UFW

如果启用了 UFW：

```bash
sudo ufw allow 8000/tcp
sudo ufw reload
sudo ufw status
```

### 6.2 云服务器安全组（如果有）

如果这是云服务器，还需要在云厂商安全组中放通 `8000/tcp`。

**建议**：只放通你的局域网网段（例如 `192.168.1.0/24`），不要直接对公网全开放。

---

## 7. 局域网调用方验证（API 可用性）

假设服务器 IP 是 `192.168.1.10`。

### 7.0 最小健康检查（不发音频也能验证服务已启动）

先用一个最轻量的请求确认“端口可达 + vLLM 已响应”：

```bash
curl http://192.168.1.10:8000/v1/models
```

如果返回 JSON（包含 `data` 等字段），说明服务已起来并能被局域网访问；如果超时/拒绝连接，优先回到第 5/6 节检查监听地址与端口放通。

### 7.1 非流式：`/v1/chat/completions`（README 示例同款）

```bash
curl http://192.168.1.10:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen/Qwen3-ASR-0.6B",
    "messages":[
      {"role":"user","content":[
        {"type":"audio_url","audio_url":{
          "url":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
        }}
      ]}
    ]
  }'
```

你会在返回的 JSON 里看到 `choices[0].message.content`，其中包含 ASR 结果文本（你可以在业务侧解析/抽取）。

### 7.2 非流式：`/v1/audio/transcriptions`（适合“上传音频文件”场景）

该接口通常以 multipart 方式上传音频文件。不同 vLLM 版本/客户端写法略有差异，推荐在调用方直接使用 OpenAI SDK（把 `base_url` 指到你的 vLLM 服务）或参考你当前 vLLM 版本的文档示例。

---

## 8. 推荐：用 systemd 将服务做成常驻进程

下面给出一个“写死用户名”的简单方案（推荐）。请把其中的 `YOUR_USER` 改成你的 Linux 用户名，并确认路径与实际一致：

- 虚拟环境：`/home/YOUR_USER/venvs/qwen3asr-vllm`
- 模型目录：`/home/YOUR_USER/models/Qwen3-ASR-0.6B`

创建服务文件：

```bash
sudo tee /etc/systemd/system/qwen3asr-vllm.service > /dev/null <<'EOF'
[Unit]
Description=Qwen3-ASR vLLM Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER
Environment=HF_HOME=/home/YOUR_USER/.cache/huggingface
Environment=TRANSFORMERS_CACHE=/home/YOUR_USER/.cache/huggingface
ExecStart=/home/YOUR_USER/venvs/qwen3asr-vllm/bin/vllm serve /home/YOUR_USER/models/Qwen3-ASR-0.6B --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.8
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
```

启动与开机自启：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now qwen3asr-vllm.service
sudo systemctl status qwen3asr-vllm.service -n 50
```

查看实时日志：

```bash
journalctl -u qwen3asr-vllm.service -f
```

---

## 9. 常见问题排查（最短路径）

### 9.1 本机能访问，局域网访问不了

优先检查三件事：

- vLLM 是否使用了 `--host 0.0.0.0`
- UFW 是否放通 `8000/tcp`
- 云安全组/交换机 ACL 是否放通并允许局域网来源

### 9.2 启动时 OOM / 显存不足

建议按顺序尝试：

- 换用 `Qwen/Qwen3-ASR-0.6B`
- 添加/调低 `--gpu-memory-utilization`（例如 0.7~0.85）
- 适当降低 `--max-model-len`

### 9.3 模型下载失败或很慢

- 用 ModelScope 或 Hugging Face CLI **提前下载到本地目录**，然后用本地路径启动：
  - `vllm serve ~/models/Qwen3-ASR-0.6B ...`

### 9.4 服务需要更“内网安全”

- 推荐在防火墙/安全组层做“只允许内网网段访问”的限制
- 如需更细粒度鉴权，可在 vLLM 前增加一层反向代理（Nginx/Caddy）或网关服务

---

## 10. 附：你应该暴露给内网项目的 API（最常用）

- `POST /v1/chat/completions`：支持 `messages` 中的 `audio_url` 形式输入音频（仓库 README 示例使用）
- `POST /v1/audio/transcriptions`：上传文件进行转写（vLLM 支持的 OpenAI 风格转写 API）

如果你同时还想要“浏览器分块流式 + committed/live 字幕”的 `/api/stream/*`，那是本仓库额外的 Flask 网关（`local_web/qwen3_asr_gateway.py`）能力，和“只启动 vLLM”是两种部署形态；建议先把 vLLM 拉起来验证可用，再按需要叠加网关。

