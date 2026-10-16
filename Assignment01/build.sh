#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
gcc -O2 -o build/matmul src/matmul.c
javac -d build src/MatMul.java
echo "build ok"
