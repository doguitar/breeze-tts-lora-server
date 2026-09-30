# Base-only smoke: health, /v1/models, and one short /v1/audio/speech WAV for breeze-base.
param(
    [string]$BaseUrl = "http://127.0.0.1:8080",
    [string]$OutDir = ($env:TEMP + "\breeze-base-smoke")
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host ("GET " + $BaseUrl + "/health")
$health = Invoke-RestMethod -Uri ($BaseUrl + "/health") -Method Get
$health | ConvertTo-Json -Depth 6 | Tee-Object -FilePath ($OutDir + "\health.json")

Write-Host ("GET " + $BaseUrl + "/v1/models")
$models = Invoke-RestMethod -Uri ($BaseUrl + "/v1/models") -Method Get
$models | ConvertTo-Json -Depth 8 | Tee-Object -FilePath ($OutDir + "\models.json")
$ids = @($models.data | ForEach-Object { $_.id })
if ($ids -notcontains "breeze-base") {
    throw ("breeze-base missing from /v1/models: " + ($ids -join ", "))
}

$wav = $OutDir + "\breeze-base.wav"
$body = (@{
    model = "breeze-base"
    input = "The train arrives in five minutes."
    seed = 42
    max_tokens = 64
} | ConvertTo-Json)

Write-Host ("POST " + $BaseUrl + "/v1/audio/speech -> " + $wav)
Invoke-WebRequest -Uri ($BaseUrl + "/v1/audio/speech") -Method Post -ContentType "application/json" -Body $body -OutFile $wav

$bytes = [System.IO.File]::ReadAllBytes($wav)
if ($bytes.Length -lt 44) { throw ("WAV too short: " + $bytes.Length + " bytes") }
$riff = [System.Text.Encoding]::ASCII.GetString($bytes, 0, 4)
$wave = [System.Text.Encoding]::ASCII.GetString($bytes, 8, 4)
if ($riff -ne "RIFF" -or $wave -ne "WAVE") {
    throw ("Not a RIFF/WAVE file: " + $riff + "/" + $wave)
}

Write-Host ("OK: breeze-base produced RIFF/WAVE (" + $bytes.Length + " bytes) at " + $wav)
