#!/bin/bash

set -e

echo "🚀 Instalador THC CLI"
echo "===================="

if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "❌ ERRO: Este instalador só funciona em Linux."
    echo "   Sistema detectado: $OSTYPE"
    exit 1
fi

echo "📋 Detectado Linux..."

TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "⬇️  Baixando pacote THC CLI..."
if ! command -v curl &> /dev/null; then
    echo "❌ ERRO: curl não encontrado. Instale com: sudo apt install curl"
    exit 1
fi

if ! curl -fsSL "https://hulktoigo-thcllm.hf.space/download/thc-cli.tar.gz" -o "$TEMP_DIR/thc-cli.tar.gz"; then
    echo "❌ ERRO: Falha ao baixar o pacote."
    echo "   Verifique sua conexão e tente novamente."
    exit 1
fi

echo "📦 Extraindo pacote..."
tar -xzf "$TEMP_DIR/thc-cli.tar.gz" -C "$TEMP_DIR"

echo "🔧 Instalando..."

if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "❌ ERRO: pip não encontrado."
    echo "   Instale com: sudo apt install python3-pip"
    exit 1
fi

PIP_CMD="pip"
if ! command -v pip &> /dev/null; then
    PIP_CMD="pip3"
fi

if ! "$PIP_CMD" install --user "$TEMP_DIR/thc_cli"; then
    echo "❌ ERRO: Falha na instalação com pip."
    exit 1
fi

if ! command -v thc &> /dev/null; then
    echo ""
    echo "⚠️  Aviso: O comando 'thc' não foi encontrado no PATH."
    echo "   Adicione ~/.local/bin ao seu PATH:"
    echo ""
    echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
    echo "   Para adicionar permanentemente ao ~/.bashrc:"
    echo "   echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
    echo "   source ~/.bashrc"
fi

echo ""
echo "✅ Instalação concluída!"
echo ""
echo "Exemplo de uso:"
echo "   thc chat \"oi\""
echo ""
echo "Para ver todos os comandos:"
echo "   thc --help"