#!/usr/bin/env bash
# Breeze LoRA server smoke test.
#
# Checks /health and /v1/models (singleton breeze-base), then runs a WAV
# sequence for the unadapted base (no voice) plus each configured voice id,
# verifying every response is a RIFF/WAVE file. Also requests an explicit MP3
# response and validates Content-Type plus ffprobe format_name=mp3. Finishes
# by checking that model=adapter-a and voice=missing are rejected with 400.
#
# Requires: curl, python3 (or PYTHON), ffprobe (from FFmpeg).
#
# Usage:
#   BASE_URL=http://127.0.0.1:8080 \
#   SMOKE_VOICES="adapter-a adapter-b" \
#     ./scripts/smoke_breeze_lora_server.sh
#
# Environment:
#   BASE_URL        server origin (default http://127.0.0.1:8080)
#   OUT_DIR         output directory (default /tmp/breeze-lora-smoke)
#   SMOKE_VOICES    space-separated voice ids to request after the no-voice base
#   SMOKE_TEXT      request text (default "The train arrives in five minutes.")
#   SMOKE_SEED      request seed (default 42)
#   SMOKE_MAX_TOKENS  request max_tokens (default 64)
#   PYTHON          python interpreter (default python3)
set -euo pipefail

PYTHON="${PYTHON:-python3}"

for tool in curl ffprobe; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: required tool not found: $tool (ffprobe comes from FFmpeg)" >&2
        exit 1
    fi
done
if ! command -v "$PYTHON" >/dev/null 2>&1 && [ ! -x "$PYTHON" ] && [ ! -f "$PYTHON" ]; then
    echo "error: required tool not found: $PYTHON (set PYTHON to a working interpreter)" >&2
    exit 1
fi

BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
OUT_DIR="${OUT_DIR:-/tmp/breeze-lora-smoke}"
SMOKE_VOICES="${SMOKE_VOICES:-adapter-a adapter-b}"
SMOKE_TEXT="${SMOKE_TEXT:-The train arrives in five minutes.}"
SMOKE_SEED="${SMOKE_SEED:-42}"
SMOKE_MAX_TOKENS="${SMOKE_MAX_TOKENS:-64}"

mkdir -p "$OUT_DIR"

curl -fsS "$BASE_URL/health" | tee "$OUT_DIR/health.json"
echo
curl -fsS "$BASE_URL/v1/models" | tee "$OUT_DIR/models.json"
echo
"$PYTHON" - "$OUT_DIR/models.json" <<'PY'
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
ids = [item["id"] for item in data.get("data", [])]
assert ids == ["breeze-base"], "expected singleton breeze-base, got %r" % ids
print("models list ok:", ids)
PY

request_wav() {
    local label="$1"
    local payload="$2"
    local wav="$OUT_DIR/${label}.wav"
    curl -fsS "$BASE_URL/v1/audio/speech" \
        -H 'Content-Type: application/json' \
        -o "$wav" \
        -d "$payload"
    "$PYTHON" -c '
import sys
from pathlib import Path
p = Path(sys.argv[1])
b = p.read_bytes()
assert b[:4] == b"RIFF" and b[8:12] == b"WAVE", "%s is not RIFF/WAVE" % p
print("%s bytes=%d" % (p.name, len(b)))
' "$wav"
}

i=0
i=$((i + 1))
request_wav "${i}-breeze-base" \
    "{\"model\":\"breeze-base\",\"input\":\"$SMOKE_TEXT\",\"seed\":$SMOKE_SEED,\"max_tokens\":$SMOKE_MAX_TOKENS}"

for voice in $SMOKE_VOICES; do
    i=$((i + 1))
    request_wav "${i}-${voice}" \
        "{\"model\":\"breeze-base\",\"voice\":\"$voice\",\"input\":\"$SMOKE_TEXT\",\"seed\":$SMOKE_SEED,\"max_tokens\":$SMOKE_MAX_TOKENS}"
done

mp3="$OUT_DIR/explicit-mp3.mp3"
mp3_headers="$OUT_DIR/explicit-mp3.headers"
curl -fsS "$BASE_URL/v1/audio/speech" \
    -H 'Content-Type: application/json' \
    -D "$mp3_headers" \
    -o "$mp3" \
    -d "{\"model\":\"breeze-base\",\"input\":\"$SMOKE_TEXT\",\"seed\":$SMOKE_SEED,\"max_tokens\":$SMOKE_MAX_TOKENS,\"response_format\":\"mp3\"}"
"$PYTHON" -c '
import sys
from pathlib import Path
headers = Path(sys.argv[1]).read_text(errors="replace").lower()
assert "content-type: audio/mpeg" in headers, "missing Content-Type: audio/mpeg in %s" % sys.argv[1]
print("mp3 content-type ok")
' "$mp3_headers"
ffprobe -v error -show_entries format=format_name -of default=nw=1 "$mp3" | tee "$OUT_DIR/explicit-mp3.ffprobe"
"$PYTHON" -c '
import sys
from pathlib import Path
line = Path(sys.argv[1]).read_text().strip()
assert line == "format_name=mp3", "expected format_name=mp3, got %r" % line
print("%s %s" % (Path(sys.argv[2]).name, line))
' "$OUT_DIR/explicit-mp3.ffprobe" "$mp3"

"$PYTHON" - "$BASE_URL" <<'PY'
import json
import sys
import urllib.error
import urllib.request

base = sys.argv[1]

def expect_400(payload, label):
    req = urllib.request.Request(
        base + "/v1/audio/speech",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req)
        raise SystemExit("expected 400 for %s" % label)
    except urllib.error.HTTPError as exc:
        assert exc.code == 400, exc
        body = exc.read().decode(errors="replace")
        print("%s ->" % label, exc.code, body)

expect_400(
    {"model": "adapter-a", "input": "hi", "seed": 1, "max_tokens": 16},
    "adapter id in model",
)
expect_400(
    {"model": "breeze-base", "voice": "missing", "input": "hi", "seed": 1, "max_tokens": 16},
    "unknown voice",
)
print("smoke ok")
PY
