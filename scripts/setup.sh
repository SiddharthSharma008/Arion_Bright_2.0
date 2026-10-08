#!/usr/bin/env bash
set -euo pipefail

echo "Setting up Arion Bright 2.0..."

if ! command -v python3 &> /dev/null; then
    echo "python3 not found. Install Python 3.11+ first."
    exit 1
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"

if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "Created .env - add your ANTHROPIC_API_KEY before running."
fi

echo ""
echo "Done. Next:"
echo "  source .venv/bin/activate"
echo "  arion ingest --path data/docs"
echo "  arion chat"
