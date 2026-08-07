param(
    [ValidateSet("install", "remove", "status", "run")]
    [string]$Action = "status"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$taskName = "SoulLink WebUI"
$startScript = (Resolve-Path (Join-Path $PSScriptRoot "start-webui.ps1")).Path
$taskCommand = "powershell.exe -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$startScript`""

switch ($Action) {
    "install" {
        & schtasks.exe /Create /TN $taskName /SC ONLOGON /TR $taskCommand /RL LIMITED /F | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Failed to register $taskName (exit $LASTEXITCODE)." }
        Write-Host "Registered '$taskName' for the current user's interactive logon."
    }
    "remove" {
        & schtasks.exe /Delete /TN $taskName /F | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Failed to remove $taskName (exit $LASTEXITCODE)." }
        Write-Host "Removed '$taskName'."
    }
    "run" {
        & powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $startScript
        if ($LASTEXITCODE -ne 0) { throw "WebUI start script failed (exit $LASTEXITCODE)." }
    }
    "status" {
        & schtasks.exe /Query /TN $taskName /FO LIST /V
        exit $LASTEXITCODE
    }
}
