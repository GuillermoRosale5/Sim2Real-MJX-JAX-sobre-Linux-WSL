#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOGS_DIR="${REPO_ROOT}/logs_sim2real_mjx"
ULTIMA_EJECUCION="${LOGS_DIR}/ultima_ejecucion.txt"
# shellcheck source=scripts/lib/platform.sh
source "${SCRIPT_DIR}/lib/platform.sh"
FASE=""
PARAR_ACTUAL=1

usage() {
  cat <<'EOF'
Uso:
  scripts/cambiar_fase_sim2real.sh [fase]

Fases:
  1  mantener_pose_xml
  2  llegar_desde_suelo
  3  recuperar_desde_caida

Opciones:
  --sin-parar   solo escribe la solicitud; no para el entrenamiento actual

Este script esta pensado para usarse con curriculo_automatico_sim2real.sh.
Escribe fase_solicitada.txt en la ultima ejecucion y, por defecto, para el bloque
actual para que el supervisor relance inmediatamente con la fase elegida.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sin-parar) PARAR_ACTUAL=0; shift ;;
    --help|-h) usage; exit 0 ;;
    1|2|3) FASE="$1"; shift ;;
    *) echo "Argumento no reconocido: $1" >&2; usage; exit 1 ;;
  esac
done

elegir_fase() {
  cat >&2 <<'EOF'
Cambiar fase curricular:
  1) mantener_pose_xml
  2) llegar_desde_suelo
  3) recuperar_desde_caida
EOF
  local opcion
  read -r -p "Elige fase [1-3]: " opcion < /dev/tty
  case "${opcion}" in
    1|2|3) printf '%s\n' "${opcion}" ;;
    *) echo "Opcion no reconocida: ${opcion}. Usa 1, 2 o 3." >&2; exit 1 ;;
  esac
}

if [[ -z "${FASE}" ]]; then
  if [[ -t 0 && -t 1 ]]; then
    FASE="$(elegir_fase)"
  else
    echo "Indica fase 1, 2 o 3." >&2
    exit 1
  fi
fi

if [[ ! -f "${ULTIMA_EJECUCION}" || -L "${ULTIMA_EJECUCION}" ]]; then
  echo "No encuentro ${ULTIMA_EJECUCION}; no se cual es la ultima ejecucion." >&2
  exit 1
fi

EJECUCION_GUARDADA="$(<"${ULTIMA_EJECUCION}")"
if ! DIRECTORIO_EJECUCION="$(sim2real_safe_existing_run "${REPO_ROOT}" "${EJECUCION_GUARDADA}" 2>/dev/null)"; then
  echo "La ultima ejecucion no existe o no es segura: ${EJECUCION_GUARDADA}" >&2
  exit 1
fi

printf '%s\n' "${FASE}" > "${DIRECTORIO_EJECUCION}/fase_solicitada.txt"
python3 - "${DIRECTORIO_EJECUCION}/fase_solicitada.json" "${FASE}" <<'PY'
import json
import sys
from datetime import datetime

ruta, fase = sys.argv[1:]
nombres = {
    "1": "mantener_pose_xml",
    "2": "llegar_desde_suelo",
    "3": "recuperar_desde_caida",
}
with open(ruta, "w", encoding="utf-8") as archivo:
    json.dump(
        {
            "fase_solicitada": int(fase),
            "nombre": nombres[str(fase)],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        archivo,
        indent=2,
        sort_keys=True,
    )
PY

echo "Solicitud escrita: fase ${FASE} en ${DIRECTORIO_EJECUCION}"

if (( PARAR_ACTUAL == 1 )); then
  echo "Parando entrenamiento actual para que el supervisor relance la fase..."
  "${SCRIPT_DIR}/sim2real.sh" parar \
    --solo-entrenamiento \
    --ejecucion "$(basename "${DIRECTORIO_EJECUCION}")" || true
else
  echo "No paro el entrenamiento actual; el cambio se aplicara cuando termine el bloque."
fi
