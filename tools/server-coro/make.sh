#!/bin/bash
# Build script for swoole_llama PHP extension
# Usage: ./make.sh [llama_cpp_dir]
#
# Prerequisites:
#   - PHP development headers (php-dev)
#   - Swoole extension installed (pecl install swoole or from source)
#   - llama.cpp built (cmake --build build -j6)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${1:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

echo "=== Building swoole_llama extension ==="
echo "llama.cpp directory: $LLAMA_DIR"
echo "Extension source: $SCRIPT_DIR"

cd "$SCRIPT_DIR"

# Clean previous build
if [ -f Makefile ]; then
    make clean 2>/dev/null || true
fi
phpize --clean

# Generate configure
phpize

# Configure with llama.cpp path
./configure \
    --enable-swoole-llama \
    --with-llama-dir="$LLAMA_DIR"

# Build
make -j6

echo ""
echo "=== Build complete ==="
echo "Extension: $SCRIPT_DIR/modules/swoole_llama.so"
echo ""
echo "To install system-wide: sudo make install"
echo "To load: php -d extension=$SCRIPT_DIR/modules/swoole_llama.so"
