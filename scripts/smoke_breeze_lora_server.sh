#!/usr/bin/env bash
# Breeze LoRA server smoke test.
#
# Checks /health and /v1/models, then runs a hot-swap WAV sequence (default
# breeze-base -> adapter-a -> adapter-b -> adapter-a -> breeze-base) and
# verifies every response is a RIFF/WAVE file. Finishes by checking that an
# unknown model id is rejected with 400.
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
set -euo pipefail

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

i=0
for model in $SMOKE_MODELS; do
    i=$((i + 1))
    wav="$OUT_DIR/${i}-${model}.wav"
    curl -fsS "$BASE_URL/v1/audio/speech" \
        -H 'Content-Type: application/json' \
        -o "$wav" \
        -d "{\"model\":\"$model\",\"input\":\"$SMOKE_TEXT\",\"seed\":$SMOKE_SEED,\"max_tokens\":$SMOKE_MAX_TOKENS}"
    python3 -c '
import sys
from pathlib import Path
p = Path(sys.argv[1])
b = p.read_bytes()
assert b[:4] == b"RIFF" and b[8:12] == b"WAVE", "%s is not RIFF/WAVE" % p
print("%s bytes=%d" % (p.name, len(b)))
' "$wav"
done

python3 - "$BASE_URL" <<'PY'
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
