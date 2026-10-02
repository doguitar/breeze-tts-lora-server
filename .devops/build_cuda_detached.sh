#!/usr/bin/env bash
# Build the CUDA image detached, with a repo-local log and exit marker so the
# result survives the launching shell being killed (the CUDA compile is longer
# than a single tool invocation window).
#
#   .devops/build_cuda_detached.sh <tag>
#
# Writes:
#   .build/cuda-<tag>.log     full docker build output
#   .build/cuda-<tag>.exit    DOCKER_EXIT=<code> VERIFY_TAG=<tag>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:?usage: $0 <image-tag>}"
BUILD_DIR="$REPO_ROOT/.build"
mkdir -p "$BUILD_DIR"
LOG="$BUILD_DIR/cuda-${TAG}.log"
MARKER="$BUILD_DIR/cuda-${TAG}.exit"
rm -f "$MARKER"

nohup bash -c '
    set +e
    docker build --progress=plain \
        -f "'"$REPO_ROOT"'/.devops/breeze_lora_cuda.Dockerfile" \
        -t "'"$TAG"'" \
        "'"$REPO_ROOT"'" > "'"$LOG"'" 2>&1
    code=$?
    printf "DOCKER_EXIT=%s\nVERIFY_TAG=%s\nUTC=%s\n" \
        "$code" "'"$TAG"'" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "'"$MARKER"'"
' > /dev/null 2>&1 &

echo "tag=$TAG"
echo "log=$LOG"
echo "marker=$MARKER"
