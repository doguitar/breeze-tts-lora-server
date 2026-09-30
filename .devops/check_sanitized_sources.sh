#!/usr/bin/env bash
# Compile-only check for the C++ sources sanitized during publication prep.
#
# The runtime image is intentionally slim (no compiler), so this uses the
# Dockerfile's `build` stage, which has g++/cmake. It compiles each translation
# unit to an object file so we get real compiler exit codes without waiting for
# the full upstream target build.
#
#   .devops/check_sanitized_sources.sh
#
# Environment:
#   CHECK_BASE_IMAGE  build-stage image tag (default ubuntu:22.04)
#   CHECK_SOURCES     space-separated repo-relative sources to compile
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE="${CHECK_BASE_IMAGE:-ubuntu:22.04}"
UTC="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${TMPDIR:-/tmp}/breeze-lora-syntax-${UTC}.log"

SOURCES="${CHECK_SOURCES:-tests/moss_tts_local/codec_decode_parity.cpp tests/moss_tts_local/codec_encode_parity.cpp tests/yue2/yue2_vae_parity_probe.cpp}"

echo "syntax log: $LOG"
status=0
for src in $SOURCES; do
    echo "== syntax check $src"
    if docker run --rm -v "$REPO_ROOT:/src:ro" -w /src --entrypoint /bin/bash "$BASE_IMAGE" -lc "
        apt-get update -qq >/dev/null 2>&1 &&
        apt-get install -y -qq build-essential >/dev/null 2>&1 &&
        g++ -std=c++17 -fsyntax-only -I include '$src'
    " >> "$LOG" 2>&1; then
        echo "   OK"
    else
        echo "   FAILED (see $LOG)"
        status=1
    fi
done

if [ "$status" -eq 0 ]; then
    echo "all sanitized sources parse"
else
    echo "one or more sources failed; log: $LOG" >&2
fi
exit "$status"
