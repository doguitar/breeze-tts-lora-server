param(
  [string]$BaseUrl = 'http://127.0.0.1:8080',
  [string]$OutDir = $(Join-Path $env:TEMP 'breeze-clone-smoke')
)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$health = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get
if ($health.status -ne 'ok') { throw "health failed: $($health | ConvertTo-Json -Compress)" }

$models = Invoke-RestMethod -Uri "$BaseUrl/v1/models" -Method Get
$models | ConvertTo-Json -Depth 6 | Set-Content -Path (Join-Path $OutDir 'models.json') -Encoding utf8
$serling = $models.data | Where-Object { $_.id -eq 'serling' }
if (-not $serling) { throw 'serling model missing from /v1/models' }
if (-not $serling.has_voice_ref) { throw 'serling.has_voice_ref expected true' }
if ([string]::IsNullOrWhiteSpace($serling.default_instruction)) { throw 'default_instruction missing' }

$body = @{
  model = 'serling'
  input = 'Welcome to another dimension.'
  seed = 7
  max_tokens = 64
} | ConvertTo-Json
$outWav = Join-Path $OutDir 'serling-clone.wav'
Invoke-WebRequest -Uri "$BaseUrl/v1/audio/speech" -Method Post -ContentType 'application/json' -Body $body -OutFile $outWav
$bytes = [System.IO.File]::ReadAllBytes($outWav)
if ($bytes.Length -lt 44) { throw "wav too small: $($bytes.Length)" }
$riff = [System.Text.Encoding]::ASCII.GetString($bytes[0..3])
$wave = [System.Text.Encoding]::ASCII.GetString($bytes[8..11])
if ($riff -ne 'RIFF' -or $wave -ne 'WAVE') { throw "bad wav magic: $riff/$wave" }
Write-Output "OK models=$(($models.data | ForEach-Object { $_.id }) -join ',') has_voice_ref=$($serling.has_voice_ref) wav_bytes=$($bytes.Length) out=$outWav"
