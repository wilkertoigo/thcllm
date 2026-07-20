#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEMP_DIR=$(mktemp -d)
DIST_DIR="$PROJECT_ROOT/dist"

echo "📦 Construindo pacote THC CLI..."

cp -r "$PROJECT_ROOT/thc_cli" "$TEMP_DIR/thc_cli"

find "$TEMP_DIR/thc_cli" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

mkdir -p "$DIST_DIR"

tar -czf "$DIST_DIR/thc-cli.tar.gz" -C "$TEMP_DIR" thc_cli

echo "✅ Pacote gerado: $DIST_DIR/thc-cli.tar.gz"

rm -rf "$TEMP_DIR"

echo "📁 Tamanho: $(du -h "$DIST_DIR/thc-cli.tar.gz" | cut -f1)"