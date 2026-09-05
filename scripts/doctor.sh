#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/lib/platform.sh
source "${SCRIPT_DIR}/lib/platform.sh"

ACCELERATOR="auto"
QUICK=0
JSON=0

usage() {
  cat <<'EOF'
Uso: ./scripts/doctor.sh [--accelerator auto|nvidia|amd|intel|cpu] [--quick] [--json]

Comprueba plataforma, GPU/runtime vendor, versiones, backend JAX e integridad del XML.
`auto` revisa el perfil que dejo activo la instalacion; no es la deteccion
`compatible` del instalador. Usa un nombre explicito solo para revisar otro
entorno ya instalado. `--quick` evita inicializar JAX/MuJoCo. `--json` produce
salida estructurada.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --accelerator)
      if [[ $# -lt 2 || "$2" == -* ]]; then
        echo "Falta el valor de --accelerator. Usa auto, nvidia, amd, intel o cpu." >&2
        usage >&2
        exit 2
      fi
      ACCELERATOR="$2"
      shift 2
      ;;
    --accelerator=*) ACCELERATOR="${1#*=}"; shift ;;
    --quick) QUICK=1; shift ;;
    --json) JSON=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Argumento no reconocido: $1" >&2; usage >&2; exit 1 ;;
  esac
done

sim2real_validate_accelerator "${ACCELERATOR}"
sim2real_acquire_runtime_lock shared "${REPO_ROOT}" wait
RESOLVED_ACCELERATOR="$(sim2real_resolve_accelerator "${ACCELERATOR}" "${REPO_ROOT}")"
SUPPORT_LEVEL="$(sim2real_support_level "${RESOLVED_ACCELERATOR}" "$(sim2real_detect_host)")"
sim2real_configure_accelerator_env "${RESOLVED_ACCELERATOR}"
sim2real_check_profile_host "${RESOLVED_ACCELERATOR}" "$(sim2real_detect_host)" "${REPO_ROOT}"

PYTHON="$(sim2real_python_executable "${RESOLVED_ACCELERATOR}" "${REPO_ROOT}")"
if [[ ! -x "${PYTHON}" ]]; then
  echo "No existe el entorno ${PYTHON%/bin/python}. Ejecuta ./scripts/install.sh --accelerator ${RESOLVED_ACCELERATOR}." >&2
  exit 1
fi

ARGS=(
  --repo-root "${REPO_ROOT}"
  --accelerator "${RESOLVED_ACCELERATOR}"
  --environment "$(sim2real_environment_dir "${RESOLVED_ACCELERATOR}" "${REPO_ROOT}")"
  --support-level "${SUPPORT_LEVEL}"
)
(( QUICK == 1 )) && ARGS+=(--quick)
(( JSON == 1 )) && ARGS+=(--json)

exec "${PYTHON}" "${SCRIPT_DIR}/doctor_runtime.py" "${ARGS[@]}"
