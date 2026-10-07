# One canonical CPU Docker build + unit tests (Windows). Do not start overlapping builds.
# Set VERIFY_TAG / VERIFY_LOG / VERIFY_EXIT_MARKER to pin one invocation; read the .exit file for DOCKER_EXIT.
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Utc = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$VerifyTag = if ($env:VERIFY_TAG) { $env:VERIFY_TAG } else { "local/breeze-lora-server:cpu-verify-$Utc" }
$Log = if ($env:VERIFY_LOG) { $env:VERIFY_LOG } else { Join-Path $env:TEMP "breeze-lora-cpu-verify-$Utc.log" }
$ExitMarker = if ($env:VERIFY_EXIT_MARKER) { $env:VERIFY_EXIT_MARKER } else { Join-Path $env:TEMP "breeze-lora-cpu-verify-$Utc.exit" }

function Write-ExitMarker {
    param(
        [Parameter(Mandatory = $true)][int]$DockerExit,
        [Parameter(Mandatory = $true)][string]$Tag,
        [Parameter(Mandatory = $true)][string]$Path
    )
    $utcNow = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    @(
        "DOCKER_EXIT=$DockerExit"
        "VERIFY_TAG=$Tag"
        "UTC=$utcNow"
    ) | Set-Content -Path $Path -Encoding utf8
}

# UTF-8 log (avoid UTF-16 from Tee-Object default on Windows PowerShell 5.1)
"VERIFY_TAG=$VerifyTag" | Out-File -FilePath $Log -Encoding utf8
"VERIFY_LOG=$Log" | Out-File -FilePath $Log -Append -Encoding utf8
"VERIFY_EXIT_MARKER=$ExitMarker" | Out-File -FilePath $Log -Append -Encoding utf8
"REPO_ROOT=$RepoRoot" | Out-File -FilePath $Log -Append -Encoding utf8

# Daemon gate: capture a numeric status; on failure write DOCKER_EXIT and stop (no build).
$ErrorActionPreference = "Continue"
docker info *> $null
$InfoExit = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($null -eq $InfoExit -or -not ($InfoExit -is [int])) {
    $InfoExit = 1
}
$InfoExit = [int]$InfoExit

if ($InfoExit -ne 0) {
    "Docker daemon not ready (docker info exit $InfoExit). Restart Docker Desktop; cancel other overlapping builds first." |
        Out-File -FilePath $Log -Append -Encoding utf8
    Write-ExitMarker -DockerExit $InfoExit -Tag $VerifyTag -Path $ExitMarker
    Write-Error "Docker daemon not ready (exit $InfoExit). See $Log / $ExitMarker"
    exit $InfoExit
}

# Run docker build via cmd so progress on stderr does not become terminating NativeCommandError.
# Keep $ErrorActionPreference=Continue only around the external docker invocations.
Push-Location $RepoRoot
try {
    $ErrorActionPreference = "Continue"
    cmd /c "docker build --progress=plain -f .devops/breeze_lora_cpu.Dockerfile -t `"$VerifyTag`" . >> `"$Log`" 2>&1"
    $BuildExit = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
} finally {
    Pop-Location
}
if ($null -eq $BuildExit -or -not ($BuildExit -is [int])) {
    $BuildExit = 1
}
$BuildExit = [int]$BuildExit

Write-ExitMarker -DockerExit $BuildExit -Tag $VerifyTag -Path $ExitMarker

if ($BuildExit -ne 0) {
    Write-Error "Docker build failed with exit $BuildExit. See $Log"
    exit $BuildExit
}

"Running unit tests in $VerifyTag..." | Out-File -FilePath $Log -Append -Encoding utf8
# Image ENTRYPOINT is breeze_lora_server; override it to run unit test binaries.
$ErrorActionPreference = "Continue"
cmd /c "docker run --rm --entrypoint /app/breeze_lora_math_test `"$VerifyTag`" >> `"$Log`" 2>&1"
if ($LASTEXITCODE -ne 0) { $ErrorActionPreference = "Stop"; exit $LASTEXITCODE }
cmd /c "docker run --rm --entrypoint /app/breeze_lora_manifest_test `"$VerifyTag`" >> `"$Log`" 2>&1"
if ($LASTEXITCODE -ne 0) { $ErrorActionPreference = "Stop"; exit $LASTEXITCODE }
cmd /c "docker run --rm --entrypoint /app/breeze_lora_server_config_test `"$VerifyTag`" >> `"$Log`" 2>&1"
if ($LASTEXITCODE -ne 0) { $ErrorActionPreference = "Stop"; exit $LASTEXITCODE }
cmd /c "docker run --rm --entrypoint /app/breeze_lora_server_ui_test `"$VerifyTag`" >> `"$Log`" 2>&1"
$TestExit = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($TestExit -ne 0) { exit $TestExit }

@(
    "UNIT_TESTS_EXIT=0"
    "VERIFY_TAG=$VerifyTag"
) | Add-Content -Path $ExitMarker -Encoding utf8

"OK: $VerifyTag built and unit tests passed. DOCKER_EXIT=0" | Out-File -FilePath $Log -Append -Encoding utf8
