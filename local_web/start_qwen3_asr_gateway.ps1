param(
    [string]$ImageTag = "qwenllm/qwen3-asr:latest",
    [string]$ContainerName = "qwen3-asr-gateway",
    [string]$Model = "Qwen/Qwen3-ASR-0.6B",
    [int]$HostPort = 8003,
    [double]$GpuMemoryUtilization = 0.88,
    [int]$MaxModelLen = 32768,
    [int]$MaxNewTokens = 32,
    [string]$HfCache = "$env:USERPROFILE\.cache\huggingface",
    [switch]$EnforceEager
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path $HfCache)) {
    New-Item -ItemType Directory -Force -Path $HfCache | Out-Null
}

$legacyContainers = @(
    $ContainerName,
    "qwen3-asr-stream-custom",
    "qwen3-asr-demo",
    "qwen3-asr-local",
    "qwen3-asr-stream"
)

$existing = docker ps -a --format "{{.Names}}"
foreach ($name in $legacyContainers | Select-Object -Unique) {
    if ($existing -contains $name) {
        Write-Host "[docker] removing old container $name" -ForegroundColor Yellow
        docker rm -f $name | Out-Null
    }
}

$command = @(
    "python", "/workspace/local_web/qwen3_asr_gateway.py",
    "--asr-model-path", $Model,
    "--gpu-memory-utilization", "$GpuMemoryUtilization",
    "--max-model-len", "$MaxModelLen",
    "--max-new-tokens", "$MaxNewTokens",
    "--host", "0.0.0.0",
    "--port", "8000"
)

if ($EnforceEager) {
    $command += "--enforce-eager"
}

$dockerArgs = @(
    "run",
    "--gpus", "all",
    "--name", $ContainerName,
    "-p", "${HostPort}:8000",
    "--mount", "type=bind,source=$HfCache,target=/root/.cache/huggingface",
    "--mount", "type=bind,source=$repoRoot,target=/workspace",
    "--workdir", "/workspace",
    "--shm-size", "8gb",
    "-d",
    $ImageTag,
    "bash", "-lc", ($command -join " ")
)

Write-Host "[run] docker $($dockerArgs -join ' ')" -ForegroundColor Cyan
docker @dockerArgs

Write-Host ""
Write-Host "Unified gateway should come up at: http://localhost:$HostPort" -ForegroundColor Green
Write-Host "Capability info: http://localhost:$HostPort/api/info" -ForegroundColor Green
Write-Host "Follow logs with:" -ForegroundColor Green
Write-Host "docker logs -f $ContainerName"
