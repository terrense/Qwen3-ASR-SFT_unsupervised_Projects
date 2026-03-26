param(
    [string]$ImageTag = "qwen3-asr-local:cu128",
    [switch]$BundleFlashAttention
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

try {
    $args = @(
        "build",
        "-f", "docker/Dockerfile-qwen3-asr-cu128",
        "-t", $ImageTag
    )

    if (-not $BundleFlashAttention) {
        $args += @("--build-arg", "BUNDLE_FLASH_ATTENTION=false")
    }

    $args += "."
    Write-Host "[build] docker $($args -join ' ')" -ForegroundColor Cyan
    docker @args
}
finally {
    Pop-Location
}
