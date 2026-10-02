#!/usr/bin/env bash
# Build and verify a breeze-lora-server container image.
#
#   .devops/verify_docker.sh cpu     # build CPU image, then run unit tests in it
#   .devops/verify_docker.sh cuda    # build CUDA image (no unit-test run: needs a GPU host)
#
# The CPU flavor is the reference verification path: it builds the image and runs
# breeze_lora_math_test / breeze_lora_manifest_test /
# breeze_lora_server_config_test inside it. The CUDA flavor builds the CUDA image
# only, because the unit-test binaries link the CUDA backend and need a GPU host
# to run.
#
# Environment:
#   VERIFY_TAG          image tag (default local/breeze-lora-server:<flavor>-<UTC>)
#   VERIFY_LOG          log path (default $TMPDIR/breeze-lora-<flavor>-<UTC>.log)
#   VERIFY_EXIT_MARKER  marker path (default <log>.exit) holding DOCKER_EXIT
#
# Exit status is 0 only when the build (and, for cpu, the unit tests) succeeded.
set -euo pipefail

FLAVOR="${1:-cpu}"
case "$FLAVOR" in
    cpu) DOCKERFILE=".devops/breeze_lora_cpu.Dockerfile" ;;
    cuda) DOCKERFILE=".devops/breeze_lora_cuda.Dockerfile" ;;
    *)
        echo "usage: $0 [cpu|cuda]" >&2
        exit 2
        ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UTC="$(date -u +%Y%m%dT%H%M%SZ)"
TMP_ROOT="${TMPDIR:-/tmp}"
VERIFY_TAG="${VERIFY_TAG:-local/breeze-lora-server:${FLAVOR}-${UTC}}"
VERIFY_LOG="${VERIFY_LOG:-${TMP_ROOT}/breeze-lora-${FLAVOR}-${UTC}.log}"
VERIFY_EXIT_MARKER="${VERIFY_EXIT_MARKER:-${TMP_ROOT}/breeze-lora-${FLAVOR}-${UTC}.exit}"

write_marker() {
    printf 'DOCKER_EXIT=%s\nVERIFY_TAG=%s\nUTC=%s\n' \
        "$1" "$VERIFY_TAG" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$VERIFY_EXIT_MARKER"
}

{
    echo "VERIFY_TAG=$VERIFY_TAG"
    echo "VERIFY_LOG=$VERIFY_LOG"
    echo "VERIFY_EXIT_MARKER=$VERIFY_EXIT_MARKER"
    echo "REPO_ROOT=$REPO_ROOT"
    echo "DOCKERFILE=$DOCKERFILE"
} > "$VERIFY_LOG"

if ! docker info > /dev/null 2>&1; then
    echo "docker daemon not ready; start Docker and retry" | tee -a "$VERIFY_LOG" >&2
    write_marker 1
    exit 1
fi

if ! docker build --progress=plain -f "$DOCKERFILE" -t "$VERIFY_TAG" "$REPO_ROOT" >> "$VERIFY_LOG" 2>&1; then
    echo "docker build failed; see $VERIFY_LOG" | tee -a "$VERIFY_LOG" >&2
    write_marker 1
    exit 1
fi

if [ "$FLAVOR" = "cpu" ]; then
    for test_binary in breeze_lora_math_test breeze_lora_manifest_test breeze_lora_server_config_test; do
        echo "running $test_binary in $VERIFY_TAG" | tee -a "$VERIFY_LOG"
        if ! docker run --rm --entrypoint "/app/$test_binary" "$VERIFY_TAG" >> "$VERIFY_LOG" 2>&1; then
            echo "$test_binary failed; see $VERIFY_LOG" | tee -a "$VERIFY_LOG" >&2
            write_marker 1
            exit 1
        fi
    done
else
    echo "cuda flavor: image built; unit tests run via the cpu flavor (GPU host required)" \
        | tee -a "$VERIFY_LOG"
fi

write_marker 0
echo "OK: $VERIFY_TAG built (DOCKER_EXIT=0); log $VERIFY_LOG" | tee -a "$VERIFY_LOG"
