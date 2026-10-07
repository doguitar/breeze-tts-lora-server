#!/usr/bin/env bash
# Breeze LoRA server smoke test.
#
# Checks /health and /v1/models, then runs a hot-swap WAV sequence (default
# breeze-base -> adapter-a -> adapter-b -> adapter-a -> breeze-base) and
# verifies every response is a RIFF/WAVE file. Also requests an explicit MP3
# response and validates Content-Type plus ffprobe format_name=mp3. Finishes
# by checking that an unknown model id is rejected with 400.
#
# Requires: curl, python3 (or PYTHON), ffprobe (from FFmpeg).
#
# Usage:
#   BASE_URL=http://127.0.0.1:8080 \
#   SMOKE_MODELS="breeze-base adapter-a adapter-b adapter-a breeze-base" \
#     ./scripts/smoke_breeze_lora_server.sh
#
# Environment:
#   BASE_URL        server origin (default http://127.0.0.1:8080)
#   OUT_DIR         output directory (default /tmp/breeze-lora-smoke)
#   SMOKE_MODELS    space-separated model ids to request in order
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
SMOKE_MODELS="${SMOKE_MODELS:-breeze-base adapter-a adapter-b adapter-a breeze-base}"
SMOKE_TEXT="${SMOKE_TEXT:-The train arrives in five minutes.}"
SMOKE_SEED="${SMOKE_SEED:-42}"
SMOKE_MAX_TOKENS="${SMOKE_MAX_TOKENS:-64}"

mkdir -p "$OUT_DIR"

curl -fsS "$BASE_URL/health" | tee "$OUT_DIR/health.json"
echo
curl -fsS "$BASE_URL/v1/models" | tee "$OUT_DIR/models.json"
echo

first_model=""
i=0
for model in $SMOKE_MODELS; do
    if [ -z "$first_model" ]; then
        first_model="$model"
    fi
    i=$((i + 1))
    wav="$OUT_DIR/${i}-${model}.wav"
    curl -fsS "$BASE_URL/v1/audio/speech" \
        -H 'Content-Type: application/json' \
        -o "$wav" \
        -d "{\"model\":\"$model\",\"input\":\"$SMOKE_TEXT\",\"seed\":$SMOKE_SEED,\"max_tokens\":$SMOKE_MAX_TOKENS}"
    "$PYTHON" -c '
import sys
from pathlib import Path
p = Path(sys.argv[1])
b = p.read_bytes()
assert b[:4] == b"RIFF" and b[8:12] == b"WAVE", "%s is not RIFF/WAVE" % p
print("%s bytes=%d" % (p.name, len(b)))
' "$wav"
done

mp3="$OUT_DIR/explicit-mp3.mp3"
mp3_headers="$OUT_DIR/explicit-mp3.headers"
curl -fsS "$BASE_URL/v1/audio/speech" \
    -H 'Content-Type: application/json' \
    -D "$mp3_headers" \
    -o "$mp3" \
    -d "{\"model\":\"$first_model\",\"input\":\"$SMOKE_TEXT\",\"seed\":$SMOKE_SEED,\"max_tokens\":$SMOKE_MAX_TOKENS,\"response_format\":\"mp3\"}"
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
req = urllib.request.Request(
    base + "/v1/audio/speech",
    data=json.dumps(
        {"model": "does-not-exist", "input": "hi", "seed": 1, "max_tokens": 16}
    ).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    urllib.request.urlopen(req)
    raise SystemExit("expected 400 for unknown model")
except urllib.error.HTTPError as exc:
    assert exc.code == 400, exc
    print("unknown model ->", exc.code)
print("smoke ok")
PY
