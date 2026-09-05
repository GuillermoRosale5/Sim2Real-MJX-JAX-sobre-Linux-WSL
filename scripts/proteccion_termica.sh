#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/lib/platform.sh
source "${SCRIPT_DIR}/lib/platform.sh"

TRAIN_PID=""
RUN_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pid) TRAIN_PID="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    *) echo "Argumento no reconocido para proteccion_termica: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${TRAIN_PID}" || -z "${RUN_DIR}" ]]; then
  echo "Uso: proteccion_termica.sh --pid PID --run-dir DIR" >&2
  exit 1
fi

RUN_DIR="$(sim2real_safe_existing_run "${REPO_ROOT}" "${RUN_DIR}")" || {
  echo "La ruta de run no es segura o no pertenece a este repositorio." >&2
  exit 1
}

# El padre escribe proteccion_termica.pid justo despues de crear este proceso.
# Esperamos ese enlace de identidad antes de abrir logs o vigilar sensores.
for _ in $(seq 1 50); do
  if [[ -f "${RUN_DIR}/proteccion_termica.pid" && ! -L "${RUN_DIR}/proteccion_termica.pid" ]] \
      && [[ "$(<"${RUN_DIR}/proteccion_termica.pid")" == "$$" ]]; then
    break
  fi
  sleep 0.1
done
if ! sim2real_thermal_guard_pid_matches_run "${REPO_ROOT}" "${RUN_DIR}" "$$" "${TRAIN_PID}"; then
  echo "No inicio el guard: su PID/cmdline o el entrenamiento no corresponden a esta run." >&2
  exit 1
fi
if ! sim2real_training_pid_matches_run "${REPO_ROOT}" "${RUN_DIR}" "${TRAIN_PID}"; then
  echo "No inicio el guard: el PID de entrenamiento no corresponde a esta run." >&2
  exit 1
fi

TEMP_WARN_C="${SIM2REAL_GPU_TEMP_WARN_C:-78}"
TEMP_STOP_C="${SIM2REAL_GPU_TEMP_STOP_C:-84}"
TEMP_CRITICAL_C="${SIM2REAL_GPU_TEMP_CRITICAL_C:-87}"
INTERVAL_SECONDS="${SIM2REAL_THERMAL_INTERVAL_SECONDS:-5}"
MAX_CONSECUTIVE="${SIM2REAL_THERMAL_MAX_CONSECUTIVE:-3}"
STATE_PATH="${RUN_DIR}/proteccion_termica_estado.csv"
PROGRESS_PATH="${RUN_DIR}/progreso.csv"
START_EPOCH="$(date +%s)"
ACCELERATOR="$(sim2real_resolve_accelerator "${SIM2REAL_ACCELERATOR:-auto}" "${REPO_ROOT}")"

if [[ ! -f "${STATE_PATH}" ]]; then
  printf 'timestamp,elapsed_seconds,elapsed_hours,train_pid,latest_num_steps,steps_per_second,training_steps_per_second,gpu_temp_c,gpu_util_percent,vram_used_mib,vram_total_mib,estado,consecutive_hot\n' > "${STATE_PATH}"
fi

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

pid_alive() {
  sim2real_pid_alive "${1:-}"
}

parar_entrenamiento() {
  local reason="$1"
  if ! sim2real_training_pid_matches_run "${REPO_ROOT}" "${RUN_DIR}" "${TRAIN_PID}"; then
    log "PARADA RECHAZADA: PID ${TRAIN_PID} ya no corresponde al entrenamiento de esta run."
    return 1
  fi
  log "PARADA DE SEGURIDAD: ${reason}. Enviando SIGTERM a PID ${TRAIN_PID}."
  sim2real_signal_training_process "${REPO_ROOT}" "${RUN_DIR}" "${TRAIN_PID}" TERM >/dev/null 2>&1 || return 1
  for _ in $(seq 1 12); do
    pid_alive "${TRAIN_PID}" || return 0
    sleep 1
  done
  if sim2real_training_pid_matches_run "${REPO_ROOT}" "${RUN_DIR}" "${TRAIN_PID}"; then
    log "PID ${TRAIN_PID} sigue activo; enviando SIGKILL."
    sim2real_signal_training_process "${REPO_ROOT}" "${RUN_DIR}" "${TRAIN_PID}" KILL >/dev/null 2>&1 || true
  else
    log "No se envia SIGKILL: la identidad del PID cambio o el entrenamiento termino."
  fi
}

progress_snapshot() {
  if [[ ! -f "${PROGRESS_PATH}" ]]; then
    printf ',,\n'
    return
  fi
  awk -F',' '
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        idx[$i] = i
      }
      next
    }
    NF {
      row = $0
    }
    END {
      if (row == "") {
        print ",,"
        exit
      }
      split(row, values, ",")
      print values[idx["num_steps"]] "," values[idx["steps_per_second"]] "," values[idx["training_steps_per_second"]]
    }
  ' "${PROGRESS_PATH}"
}

log "Proteccion termica iniciada para PID ${TRAIN_PID}."
log "Sensor seleccionado: ${ACCELERATOR}."
log "Umbrales: warn=${TEMP_WARN_C}C stop=${TEMP_STOP_C}C critical=${TEMP_CRITICAL_C}C intervalo=${INTERVAL_SECONDS}s consecutivos=${MAX_CONSECUTIVE}."

consecutive_hot=0
while sim2real_training_pid_matches_run "${REPO_ROOT}" "${RUN_DIR}" "${TRAIN_PID}"; do
  gpu_line="$(sim2real_gpu_metrics_csv "${ACCELERATOR}" 2>/dev/null || true)"
  if [[ -z "${gpu_line}" ]]; then
    log "No puedo leer el sensor ${ACCELERATOR}; sigo vigilando."
    sleep "${INTERVAL_SECONDS}"
    continue
  fi

  IFS=',' read -r temp util mem_used mem_total <<< "${gpu_line}"
  temp="${temp//[[:space:]]/}"
  util="${util//[[:space:]]/}"
  mem_used="${mem_used//[[:space:]]/}"
  mem_total="${mem_total//[[:space:]]/}"

  estado="ok"
  now_epoch="$(date +%s)"
  elapsed_seconds=$(( now_epoch - START_EPOCH ))
  elapsed_hours="$(awk -v seconds="${elapsed_seconds}" 'BEGIN { printf "%.6f", seconds / 3600 }')"
  IFS=',' read -r latest_num_steps steps_per_second training_steps_per_second <<< "$(progress_snapshot)"
  if (( temp >= TEMP_CRITICAL_C )); then
    estado="critical"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$(date --iso-8601=seconds)" "${elapsed_seconds}" "${elapsed_hours}" "${TRAIN_PID}" \
      "${latest_num_steps}" "${steps_per_second}" "${training_steps_per_second}" \
      "${temp}" "${util}" "${mem_used}" "${mem_total}" "${estado}" "${consecutive_hot}" \
      >> "${STATE_PATH}"
    parar_entrenamiento "GPU a ${temp}C, supera umbral critico ${TEMP_CRITICAL_C}C"
    exit 2
  fi

  if (( temp >= TEMP_STOP_C )); then
    consecutive_hot=$(( consecutive_hot + 1 ))
    estado="hot"
  else
    consecutive_hot=0
    if (( temp >= TEMP_WARN_C )); then
      estado="warn"
    fi
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date --iso-8601=seconds)" "${elapsed_seconds}" "${elapsed_hours}" "${TRAIN_PID}" \
    "${latest_num_steps}" "${steps_per_second}" "${training_steps_per_second}" \
    "${temp}" "${util}" "${mem_used}" "${mem_total}" "${estado}" "${consecutive_hot}" \
    >> "${STATE_PATH}"

  if [[ "${estado}" == "warn" ]]; then
    log "Aviso: GPU a ${temp}C."
  elif [[ "${estado}" == "hot" ]]; then
    log "Temperatura alta ${temp}C (${consecutive_hot}/${MAX_CONSECUTIVE})."
  fi

  if (( consecutive_hot >= MAX_CONSECUTIVE )); then
    parar_entrenamiento "GPU >= ${TEMP_STOP_C}C durante ${consecutive_hot} chequeos seguidos"
    exit 2
  fi

  sleep "${INTERVAL_SECONDS}"
done

log "Entrenamiento PID ${TRAIN_PID} termino o cambio de identidad; proteccion termica termina."
