param(
    [string]$ImageTag = "qwen3-asr-local:cu128",
    [string]$ContainerName = "qwen3-asr-local",
    [string]$Model = "Qwen/Qwen3-ASR-0.6B",
    [int]$HostPort = 8000,
    [double]$GpuMemoryUtilization = 0.75,
    [string]$HfCache = "$env:USERPROFILE\.cache\huggingface",
    [switch]$EnforceEager
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $HfCache)) {
    New-Item -ItemType Directory -Force -Path $HfCache | Out-Null
}

$existing = docker ps -a --format "{{.Names}}" | Where-Object { $_ -eq $ContainerName }
if ($existing) {
    Write-Host "[docker] removing old container $ContainerName" -ForegroundColor Yellow
    docker rm -f $ContainerName | Out-Null
}

$command = "qwen-asr-serve $Model --gpu-memory-utilization $GpuMemoryUtilization --host 0.0.0.0 --port 8000"
if ($EnforceEager) {
    $command += " --enforce-eager"
}

$args = @(
    "run",
    "--gpus", "all",
    "--name", $ContainerName,
    "-p", "${HostPort}:8000",
    "--mount", "type=bind,source=$HfCache,target=/root/.cache/huggingface",
    "--shm-size", "8gb",
    "-d",
    $ImageTag,
    "bash", "-lc", $command
)

Write-Host "[run] docker $($args -join ' ')" -ForegroundColor Cyan
docker @args

Write-Host ""
Write-Host "Server should become available at: http://localhost:$HostPort/v1/chat/completions" -ForegroundColor Green
Write-Host "Follow logs with:" -ForegroundColor Green
Write-Host "docker logs -f $ContainerName"
