$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root

try {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        python -m http.server 8081 --directory local_web
        exit $LASTEXITCODE
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 -m http.server 8081 --directory local_web
        exit $LASTEXITCODE
    }

    throw "Neither python nor py is available in PATH."
}
finally {
    Pop-Location
}
