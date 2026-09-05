#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOGS_DIR="${REPO_ROOT}/logs_sim2real_mjx"
ULTIMA_EJECUCION="${LOGS_DIR}/ultima_ejecucion.txt"
VENV_PYTHON=""
XML_DIR="${REPO_ROOT}/sim2real_mjx/xmls"
# shellcheck source=scripts/lib/platform.sh
source "${SCRIPT_DIR}/lib/platform.sh"

cd "${REPO_ROOT}"

configurar_entorno_visualizador() {
  export PYTHONPATH="${REPO_ROOT}"
  export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.10}"
  export MUJOCO_GL="${MUJOCO_VIEWER_GL:-glfw}"
}

ejecucion_actual() {
  if [[ -f "${ULTIMA_EJECUCION}" && ! -L "${ULTIMA_EJECUCION}" ]]; then
    local saved safe_saved
    saved="$(<"${ULTIMA_EJECUCION}")"
    if safe_saved="$(sim2real_safe_existing_run "${REPO_ROOT}" "${saved}" 2>/dev/null)"; then
      printf '%s\n' "${safe_saved}"
      return
    fi
  fi
  local safe_logs
  safe_logs="$(sim2real_logs_root "${REPO_ROOT}")" || return 1
  find "${safe_logs}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null |
    sort -nr |
    awk 'NR == 1 {print $2}' || true
}

elegir_de_lista() {
  local mensaje="$1"
  local cantidad="$2"
  local seleccionada
  read -r -p "${mensaje}" seleccionada
  if [[ ! "${seleccionada}" =~ ^[0-9]+$ ]] || (( seleccionada < 1 || seleccionada > cantidad )); then
    echo "Opcion no valida: ${seleccionada}. Elige un numero entre 1 y ${cantidad}." >&2
    exit 1
  fi
  printf '%s\n' "${seleccionada}"
}

visualizar_ultimo_checkpoint() {
  echo "Abriendo ultimo checkpoint local (puede ser un entrenamiento parcial)..."
  exec ./scripts/sim2real.sh visualizar-resultados "$@"
}

visualizar_modelo_preentrenado() {
  echo "Abriendo el modelo preentrenado recomendado: fase 2, paso 45.932.544..."
  echo "Esta opcion no consulta logs_sim2real_mjx/ultima_ejecucion.txt."
  exec ./scripts/sim2real.sh visualizar-modelo-preentrenado "$@"
}

visualizar_checkpoint_anterior() {
  local directorio_ejecucion
  directorio_ejecucion="$(ejecucion_actual)"
  if [[ -z "${directorio_ejecucion}" ]]; then
    echo "No encuentro ninguna ejecucion en ${LOGS_DIR}." >&2
    exit 1
  fi

  mapfile -t checkpoints < <(
    find "${directorio_ejecucion}/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%P\n' 2>/dev/null |
      awk '/^[0-9]+$/' |
      sort -nr
  )

  if (( ${#checkpoints[@]} <= 1 )); then
    echo "No hay checkpoints anteriores en la ejecucion actual: ${directorio_ejecucion}" >&2
    echo "Ejecucion encontrada, pero checkpoints disponibles: ${#checkpoints[@]}" >&2
    exit 1
  fi

  echo ""
  echo "Ejecucion actual:"
  echo "  ${directorio_ejecucion}"
  echo ""
  echo "Checkpoints anteriores:"
  local i paso ruta_checkpoint
  for (( i = 1; i < ${#checkpoints[@]}; i++ )); do
    paso="${checkpoints[$i]}"
    ruta_checkpoint="${directorio_ejecucion}/checkpoints/${paso}"
    printf '  %2d) paso %-12s  %s\n' "${i}" "${paso}" "$(date -d "@$(stat -c %Y "${ruta_checkpoint}")" '+%Y-%m-%d %H:%M:%S')"
  done

  local seleccion
  local seleccion_maxima="$(( ${#checkpoints[@]} - 1 ))"
  seleccion="$(elegir_de_lista "Elige checkpoint anterior [1-${seleccion_maxima}]: " "${seleccion_maxima}")"
  ruta_checkpoint="${directorio_ejecucion}/checkpoints/${checkpoints[$seleccion]}"
  echo "Abriendo checkpoint: ${ruta_checkpoint}"
  exec ./scripts/sim2real.sh visualizar-resultados --ruta-checkpoint "${ruta_checkpoint}" "$@"
}

visualizar_xml() {
  mapfile -t xmls < <(find "${XML_DIR}" -maxdepth 1 -type f -name '*.xml' -printf '%f\n' | sort)
  if (( ${#xmls[@]} == 0 )); then
    echo "No encuentro XMLs en ${XML_DIR}." >&2
    exit 1
  fi

  echo ""
  echo "XMLs disponibles:"
  local i xml_path
  for (( i = 0; i < ${#xmls[@]}; i++ )); do
    xml_path="${XML_DIR}/${xmls[$i]}"
    printf '  %2d) %-36s  %s\n' "$(( i + 1 ))" "${xmls[$i]}" "$(date -d "@$(stat -c %Y "${xml_path}")" '+%Y-%m-%d %H:%M:%S')"
  done

  local choice xml
  choice="$(elegir_de_lista "Elige XML [1-${#xmls[@]}]: " "${#xmls[@]}")"
  xml="${XML_DIR}/${xmls[$(( choice - 1 ))]}"

  local accelerator
  accelerator="$(sim2real_resolve_accelerator "${SIM2REAL_ACCELERATOR:-auto}" "${REPO_ROOT}")"
  sim2real_configure_accelerator_env "${accelerator}"
  VENV_PYTHON="$(sim2real_python_executable "${accelerator}" "${REPO_ROOT}")"
  configurar_entorno_visualizador
  [[ -x "${VENV_PYTHON}" ]] || {
    echo "Falta ${VENV_PYTHON}. Ejecuta ./scripts/install.sh --accelerator ${accelerator}." >&2
    exit 1
  }
  echo "Abriendo XML sin simular: ${xml}"
  sim2real_run_with_runtime_lock "${REPO_ROOT}" "${VENV_PYTHON}" - "${xml}" <<'PY'
from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

xml = Path(sys.argv[1]).resolve()
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

print(f"XML: {xml.name}")
print(f"qpos0 z cuerpo: {data.qpos[2]:.4f} m")
print("Vista estatica. Cierra la ventana para terminar.")

with mujoco.viewer.launch_passive(model, data) as viewer:
  while viewer.is_running():
    viewer.sync()
    time.sleep(0.02)
PY
}

main() {
  echo "=============================================="
  echo "Visualizador SIM2REAL"
  echo "=============================================="
  echo "  1) Modelo preentrenado recomendado (fase 2, paso 45.932.544)"
  echo "  2) Ultimo checkpoint local (puede ser parcial)"
  echo "  3) Checkpoint local anterior"
  echo "  4) Ver XML sin simular"
  echo ""

  local choice
  choice="$(elegir_de_lista "Elige opcion [1-4]: " 4)"
  case "${choice}" in
    1) visualizar_modelo_preentrenado "$@" ;;
    2) visualizar_ultimo_checkpoint "$@" ;;
    3) visualizar_checkpoint_anterior "$@" ;;
    4) visualizar_xml ;;
  esac
}

main "$@"
