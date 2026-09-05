#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
# shellcheck source=scripts/lib/platform.sh
source "${SCRIPT_DIR}/lib/platform.sh"

export PYTHONPATH="${PWD}"
ACCELERATOR="$(sim2real_resolve_accelerator "${SIM2REAL_ACCELERATOR:-auto}" "${PWD}")"
PYTHON="$(sim2real_python_executable "${ACCELERATOR}" "${PWD}")"
[[ -x "${PYTHON}" ]] || {
  echo "Falta el entorno ${PYTHON%/bin/python}. Ejecuta ./scripts/install.sh --accelerator ${ACCELERATOR}." >&2
  exit 1
}
sim2real_run_with_runtime_lock "${PWD}" \
  "${PYTHON}" "${PWD}/scripts/ver_recompensas_en_directo.py" "$@"
