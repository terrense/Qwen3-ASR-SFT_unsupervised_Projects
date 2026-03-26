## Qwen3-ASR Local Web Test

This folder is a minimal Windows-friendly setup for testing Qwen3-ASR locally:

1. Build the Docker image from the repository Dockerfile.
2. Start a local Qwen3-ASR server in Docker.
3. Start a tiny static web server for `index.html`.
4. Open the page in a browser and upload audio.

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

For live partial transcription, use the custom launcher that lowers vLLM memory
pressure on smaller GPUs:

```powershell
.\local_web\start_qwen3_asr_streaming.ps1
```

Open:

```text
http://localhost:8002
```

If your GPU still struggles, try:

```powershell
.\local_web\start_qwen3_asr_streaming.ps1 -MaxModelLen 24576 -EnforceEager
```

### Notes

- The page sends local audio as a base64 data URL to the OpenAI-style `chat/completions` endpoint.
- If Docker image build fails while pulling base layers, rerun the build script once. Layer pull failures are often transient.
- If the browser reports a network error, verify the container is actually serving on `localhost:8000`.
