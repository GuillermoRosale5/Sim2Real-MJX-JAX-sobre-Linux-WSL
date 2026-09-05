#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
cd "${REPO_ROOT}"

bash tests/test_pretrained_model.sh

python3 -m unittest discover -s tests -p 'test_*.py' -v

UV_REQUIRED_VERSION="0.11.8"
if command -v uv >/dev/null 2>&1; then
  UV_CURRENT_VERSION="$(uv --version | awk '{print $2}')"
  if [[ "${UV_CURRENT_VERSION}" != "${UV_REQUIRED_VERSION}" ]]; then
    echo "Se requiere uv ${UV_REQUIRED_VERSION} para verificar uv.lock; activa esa version o ejecuta ./scripts/install.sh." >&2
    exit 1
  fi
  uv lock --check
else
  echo "Aviso: uv ${UV_REQUIRED_VERSION} no esta instalado; se omite uv lock --check." >&2
fi

echo "Pruebas estaticas OK."
