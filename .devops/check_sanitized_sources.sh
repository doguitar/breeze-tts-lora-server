#!/usr/bin/env bash
# Compile-only check for the C++ sources sanitized during publication prep.
#
# The runtime image is intentionally slim (no compiler), so this uses a plain
# ubuntu base with build-essential. ggml ships as a git submodule under
# external/ggml, so it must be initialised for the include path to resolve.
#
#   .devops/check_sanitized_sources.sh
#
# Environment:
#   CHECK_BASE_IMAGE  image providing g++ (default ubuntu:22.04)
#   CHECK_SOURCES     space-separated repo-relative sources to compile
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE="${CHECK_BASE_IMAGE:-ubuntu:22.04}"

SOURCES="${CHECK_SOURCES:-tests/moss_tts_local/codec_decode_parity.cpp tests/moss_tts_local/codec_encode_parity.cpp tests/yue2/yue2_vae_parity_probe.cpp}"

if [ ! -f "$REPO_ROOT/external/ggml/include/ggml.h" ]; then
    echo "external/ggml/include/ggml.h is missing; run 'git submodule update --init' first" >&2
    exit 1
fi

status=0
for src in $SOURCES; do
    echo "== syntax check $src"
    if docker run --rm -v "$REPO_ROOT:/src:ro" -w /src --entrypoint /bin/bash "$BASE_IMAGE" -lc "
        apt-get update -qq >/dev/null 2>&1 &&
        apt-get install -y -qq build-essential >/dev/null 2>&1 &&
        g++ -std=c++17 -fsyntax-only -I include -I external/ggml/include '$src'
    " 2>&1; then
        echo "   OK"
    else
        echo "   FAILED"
        status=1
    fi
done

if [ "$status" -eq 0 ]; then
    echo "all sanitized sources parse"
else
    echo "one or more sources failed" >&2
fi
exit "$status"
