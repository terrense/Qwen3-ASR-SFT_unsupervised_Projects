## Qwen3-ASR Local Web Test

This folder is a minimal Windows-friendly setup for testing Qwen3-ASR locally:

1. Build the Docker image from the repository Dockerfile.
2. Start a local Qwen3-ASR server in Docker.
3. Open a browser UI and test file upload or streaming.

## Recommended: one container, multiple capabilities

If you want to avoid loading the model multiple times, use the unified gateway.
It serves one page and multiple API routes from one model process:

- `/api/info`: capability discovery
- `/api/transcribe`: one-shot file transcription
- `/api/stream/*`: live streaming transcription

Start it with:

```powershell
.\local_web\start_qwen3_asr_gateway.ps1
```

Open:

```text
http://localhost:8003
```

This is now the preferred setup if you care about GPU memory.

### 1) Build the image

```powershell
.\local_web\build_qwen3_asr_image.ps1
```

### 2) Start the ASR server container

Default model is `Qwen/Qwen3-ASR-0.6B`, which is the safer first choice on a 12 GB GPU.

```powershell
.\local_web\start_qwen3_asr_server.ps1
```

If your local image build is blocked by Docker Hub base-layer issues, you can
use the official prebuilt image as a fallback:

```powershell
.\local_web\start_qwen3_asr_server.ps1 -ImageTag qwenllm/qwen3-asr:latest
```

If vLLM startup seems to stall on Windows/WSL, try eager mode:

```powershell
.\local_web\start_qwen3_asr_server.ps1 -ImageTag qwenllm/qwen3-asr:latest -EnforceEager
```

Server endpoint:

```text
http://localhost:8000/v1/chat/completions
```

### 3) Start the local static web page

```powershell
.\local_web\start_local_web.ps1
```

Open:

```text
http://localhost:8081
```

### 4) Start the built-in streaming web demo

For live partial transcription, use the custom subtitle-oriented launcher that
lowers vLLM memory pressure on smaller GPUs and keeps committed subtitle history
visible while only refreshing the unstable tail:

```powershell
.\local_web\start_qwen3_asr_streaming.ps1
```

Open:

```text
http://localhost:8003
```

If your GPU still struggles, try:

```powershell
.\local_web\start_qwen3_asr_streaming.ps1 -MaxModelLen 24576 -EnforceEager
```

### Notes

- The page sends local audio as a base64 data URL to the OpenAI-style `chat/completions` endpoint.
- If Docker image build fails while pulling base layers, rerun the build script once. Layer pull failures are often transient.
- If the browser reports a network error, verify the container is actually serving on `localhost:8000`.
- Qwen3-ASR streaming revises the trailing hypothesis on each chunk. The new `8003` page shows this as `committed history + live tail` instead of replacing the whole subtitle block.
- The unified gateway is the better long-term shape: one container, one model instance, many HTTP routes.
