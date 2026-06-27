#!/usr/bin/env bash
# Build LLVM at the exact revision pinned by Triton (see triton/cmake/llvm-hash.txt).
# Triton 3.7.1 → LLVM 23.0.0git @ 7f77ca0dbda4 (2026).
set -euo pipefail
HASH="${1:-7f77ca0dbda4abbf9af06537b2c475f20ccd6007}"
ROOT="${LLVM_SRC:-$HOME/workspace/repos/llvm-project}"
BUILD="${LLVM_BUILD:-$HOME/workspace/repos/llvm-build-triton}"
JOBS="${MAX_JOBS:-$(nproc)}"

if [[ ! -d "$ROOT/.git" ]]; then
  git clone --depth 1 https://github.com/llvm/llvm-project.git "$ROOT"
fi
cd "$ROOT"
git fetch --depth 1 origin "$HASH" 2>/dev/null || git fetch origin
git checkout "$HASH"

cmake -G Ninja -B "$BUILD" -S llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld;clang" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDIGPU" \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/llvm-triton"

ninja -C "$BUILD" install -j"$JOBS"
echo "Installed to $HOME/.local/llvm-triton — add bin to PATH:"
echo "  export PATH=$HOME/.local/llvm-triton/bin:\$PATH"
