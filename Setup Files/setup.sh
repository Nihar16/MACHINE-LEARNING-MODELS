#!/usr/bin/env bash
# Setup script for macOS and Linux
# Usage: ./setup.sh [--launch]   (--launch opens Jupyter when setup finishes)
set -euo pipefail

VENV_DIR="venv"
MIN_MAJOR=3
MIN_MINOR=9

case "$(uname -s)" in
  Darwin) OS="macOS" ;;
  Linux)  OS="Linux" ;;
  *) echo "Unsupported OS: $(uname -s). On Windows use setup.ps1 or setup.bat."; exit 1 ;;
esac
echo "==> Detected $OS"

# Find Python
if command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else
  echo "Python not found."
  if [ "$OS" = "macOS" ]; then echo "Install it with: brew install python"
  else echo "Install it with: sudo apt install python3 python3-venv python3-pip   (Debian/Ubuntu)"; fi
  exit 1
fi

# Check version
if ! $PY -c "import sys; sys.exit(0 if sys.version_info >= ($MIN_MAJOR, $MIN_MINOR) else 1)"; then
  echo "Python $MIN_MAJOR.$MIN_MINOR+ is required. Found: $($PY --version)"
  exit 1
fi
echo "==> Using $($PY --version)"

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating virtual environment in ./$VENV_DIR"
  $PY -m venv "$VENV_DIR" || {
    echo "venv creation failed. On Debian/Ubuntu run: sudo apt install python3-venv"
    exit 1
  }
else
  echo "==> Virtual environment already exists, reusing it"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
python -m pip install --upgrade pip

echo "==> Installing dependencies"
pip install -r requirements.txt

echo "==> Registering Jupyter kernel"
python -m ipykernel install --user --name machine-learning-models --display-name "Python (ML Models)"

echo ""
echo "Setup complete."
echo "Activate the environment later with: source $VENV_DIR/bin/activate"

if [ "${1:-}" = "--launch" ]; then
  echo "==> Launching Jupyter Notebook"
  jupyter notebook
else
  echo "Start notebooks with: jupyter notebook"
fi
