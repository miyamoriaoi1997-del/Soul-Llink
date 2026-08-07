param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8765
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($HostAddress -ne "127.0.0.1") {
    throw "SoulLink WebUI autostart is localhost-only; HostAddress must be 127.0.0.1."
}

$healthUrl = "http://${HostAddress}:${Port}/api/v1/health"
try {
    $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
    if ($health.ok -eq $true -and $health.service -eq "soullink-monitor") {
        exit 0
    }
} catch {
    # Not running yet; continue with a local start.
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "SoulLink Python was not found at $python. Run 'uv sync' in $repoRoot before starting the WebUI."
}

$logDir = Join-Path $env:LOCALAPPDATA "hermes\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "soullink-webui.stdout.log"
$stderr = Join-Path $logDir "soullink-webui.stderr.log"

$packages = Join-Path $repoRoot "packages"
$adapters = Join-Path $repoRoot "adapters"
$env:PYTHONPATH = "$packages;$adapters"

$arguments = @(
    "-m", "pcltm.cli", "webui",
    "--host", $HostAddress,
    "--port", $Port.ToString(),
    "--no-open-browser"
)

Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr
