#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${MONOWIRE_BUILD_DIR:-"$ROOT_DIR/build"}
BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
MODE=${1:-fast}
MODEL_PATH=${2:-${MONOWIRE_TEST_MODELFILE:-}}

usage() {
    cat <<'EOF'
usage: ./scripts/run-tests.sh [fast|full|model|all] [path/to/model.gguf]

Modes:
  fast   Build and run the fast offline regression suite. Default mode.
  full   Build and run the full offline suite, excluding model-backed tests.
  model  Build and run model-backed integration tests. Requires a GGUF model.
  all    Build and run every test currently enabled in the build directory.

Environment:
  MONOWIRE_BUILD_DIR      CMake build directory. Default: ./build
  MONOWIRE_BUILD_JOBS     Parallel build jobs. Auto-detected when unset.
  CMAKE_BUILD_TYPE        CMake build type. Default: Release
  MONOWIRE_TEST_MODELFILE Model path for integration tests.
EOF
}

detect_jobs() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
        return
    fi

    if command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu
        return
    fi

    echo 4
}

case "$MODE" in
    fast)
        TARGET=check-fast
        ;;
    full)
        TARGET=check
        ;;
    model)
        TARGET=check-model
        if [[ -z "$MODEL_PATH" ]]; then
            echo "error: model mode requires a GGUF path as the second argument or via MONOWIRE_TEST_MODELFILE." >&2
            exit 1
        fi
        ;;
    all)
        TARGET=check-all
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "error: unknown mode '$MODE'." >&2
        usage >&2
        exit 1
        ;;
esac

JOBS=${MONOWIRE_BUILD_JOBS:-$(detect_jobs)}

cmake \
    -S "$ROOT_DIR" \
    -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DMONOWIRE_BUILD_TESTS=ON \
    -DMONOWIRE_TEST_MODELFILE="$MODEL_PATH"

cmake --build "$BUILD_DIR" --parallel "$JOBS"
cmake --build "$BUILD_DIR" --target "$TARGET"
