#!/usr/bin/env bash
# Build and verify L3.5 LLVM CountAndDCEPass lab.
# Prefers LLVM 20 (matches Triton/PyTorch toolchain era); falls back to Docker silkeh/clang:20.
set -euo pipefail
LAB="$(cd "$(dirname "$0")" && pwd)"
cd "$LAB"
LOG="$LAB/llvm_pass.log"
: > "$LOG"

run_native() {
  export PATH="/usr/lib/llvm-20/bin:${LLVM_PREFIX:-/usr}/bin:$PATH"
  echo "[llvm] native $(llvm-config --version)" | tee -a "$LOG"
  bash -c '
    set -euo pipefail
    clang -O0 -Xclang -disable-O0-optnone -S -emit-llvm demo.c -o demo.ll
    rm -rf build
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j"$(nproc)"
    PLUGIN="./build/libCountAndDCEPass.so"
    opt -load-pass-plugin="$PLUGIN" -passes="count-and-dce" -S demo.ll -o demo.opt.ll
    cat > driver.ll <<'"'"'EOF'"'"'
declare i32 @compute(i32)
define i32 @main() {
  %r = call i32 @compute(i32 7)
  ret i32 %r
}
EOF
    llvm-link demo.ll driver.ll -S -o before.ll
    llvm-link demo.opt.ll driver.ll -S -o after.ll
    BEFORE=$(lli before.ll; echo $?)
    AFTER=$(lli after.ll; echo $?)
    echo "[semantic] before=$BEFORE after=$AFTER"
    test "$BEFORE" = "8" && test "$AFTER" = "8"
  ' 2>&1 | tee -a "$LOG"
}

run_docker() {
  echo "[llvm] docker silkeh/clang:20" | tee -a "$LOG"
  docker run --rm -u "$(id -u):$(id -g)" -v "$LAB:/work" -w /work silkeh/clang:20 bash -c '
    set -euo pipefail
    clang -O0 -Xclang -disable-O0-optnone -S -emit-llvm demo.c -o demo.ll
    rm -rf build
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j"$(nproc)"
    PLUGIN="./build/libCountAndDCEPass.so"
    opt -load-pass-plugin="$PLUGIN" -passes="count-and-dce" -S demo.ll -o demo.opt.ll
    cat > driver.ll <<'"'"'EOF'"'"'
declare i32 @compute(i32)
define i32 @main() {
  %r = call i32 @compute(i32 7)
  ret i32 %r
}
EOF
    llvm-link demo.ll driver.ll -S -o before.ll
    llvm-link demo.opt.ll driver.ll -S -o after.ll
    BEFORE=$(lli before.ll; echo $?)
    AFTER=$(lli after.ll; echo $?)
    echo "[semantic] before=$BEFORE after=$AFTER"
    test "$BEFORE" = "8" && test "$AFTER" = "8"
  ' 2>&1 | tee -a "$LOG"
}

if llvm-config --version 2>/dev/null | grep -qE '^20\.'; then
  run_native
elif command -v docker >/dev/null 2>&1; then
  run_docker
else
  echo "[fail] need LLVM 20 or Docker" | tee -a "$LOG"
  exit 1
fi
echo "[ok] LLVM Pass semantic check passed" | tee -a "$LOG"
