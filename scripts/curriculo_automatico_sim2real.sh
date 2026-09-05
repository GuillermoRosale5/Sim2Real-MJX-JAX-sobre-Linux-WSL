#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/lib/platform.sh
source "${SCRIPT_DIR}/lib/platform.sh"

PASOS_TOTALES=200000000
PASOS_BLOQUE=5000000
PERFIL_PPO="${SIM2REAL_PERFIL_PPO:-ligero}"
NOMBRE_EJECUCION="Sim2RealCurriculo-$(date +%Y%m%d-%H%M%S)"
FASE="${SIM2REAL_FASE_RECOMPENSA:-1}"
PREPARAR_PRIMERO=1
REINICIAR_PRIMERO=1
PARAR_AL_SOLICITAR=1

usage() {
  cat <<'EOF'
Uso:
  scripts/curriculo_automatico_sim2real.sh [opciones]

Opciones:
  --perfil-ppo depuracion|ligero|ligero_rapido|completo
  --nombre-ejecucion NOMBRE
  --fase-inicial 1|2|3
  --pasos-totales N              valor inicial: 200000000
  --pasos-por-bloque N           valor inicial: 5000000
  --sin-preparacion
  --sin-reinicio-inicial
  --sin-parada-al-solicitar      no para el bloque al recibir fase_solicitada.txt

Control manual:
  scripts/cambiar_fase_sim2real.sh

El supervisor entrena por bloques. Al terminar cada bloque evalua metricas recientes
y sube de fase si el rendimiento es suficiente. Si pides una fase manual, para
el bloque actual y relanza restaurando el ultimo checkpoint.
EOF
}

validar_fase() {
  case "${1:-}" in
    1|2|3) return 0 ;;
    *) echo "Fase invalida: ${1:-<vacia>}. Usa 1, 2 o 3." >&2; exit 1 ;;
  esac
}

validar_perfil() {
  case "${1:-}" in
    depuracion|ligero|ligero_rapido|completo) return 0 ;;
    *) echo "Perfil PPO invalido: ${1:-<vacio>}." >&2; exit 1 ;;
  esac
}

validar_nombre_ejecucion() {
  if [[ "${1:-}" == "." || "${1:-}" == ".." || ! "${1:-}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
    echo "Nombre de ejecucion no valido: ${1:-<vacio>}. No se admiten rutas." >&2
    exit 1
  fi
}

validar_entero_positivo() {
  local nombre="$1" valor="$2"
  if [[ ! "${valor}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${nombre} debe ser un entero positivo: ${valor:-<vacio>}." >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --perfil-ppo) PERFIL_PPO="$2"; shift 2 ;;
    --perfil-ppo=*) PERFIL_PPO="${1#*=}"; shift ;;
    --nombre-ejecucion) NOMBRE_EJECUCION="$2"; shift 2 ;;
    --nombre-ejecucion=*) NOMBRE_EJECUCION="${1#*=}"; shift ;;
    --fase-inicial) FASE="$2"; shift 2 ;;
    --fase-inicial=*) FASE="${1#*=}"; shift ;;
    --pasos-totales) PASOS_TOTALES="$2"; shift 2 ;;
    --pasos-totales=*) PASOS_TOTALES="${1#*=}"; shift ;;
    --pasos-por-bloque) PASOS_BLOQUE="$2"; shift 2 ;;
    --pasos-por-bloque=*) PASOS_BLOQUE="${1#*=}"; shift ;;
    --sin-preparacion) PREPARAR_PRIMERO=0; shift ;;
    --sin-reinicio-inicial) REINICIAR_PRIMERO=0; shift ;;
    --sin-parada-al-solicitar) PARAR_AL_SOLICITAR=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Argumento no reconocido: $1" >&2; usage; exit 1 ;;
  esac
done

validar_perfil "${PERFIL_PPO}"
validar_fase "${FASE}"
validar_nombre_ejecucion "${NOMBRE_EJECUCION}"
validar_entero_positivo "--pasos-totales" "${PASOS_TOTALES}"
validar_entero_positivo "--pasos-por-bloque" "${PASOS_BLOQUE}"

DIRECTORIO_EJECUCION="$(sim2real_safe_run_dir "${REPO_ROOT}" "${NOMBRE_EJECUCION}")"
ARCHIVO_SOLICITUD="${DIRECTORIO_EJECUCION}/fase_solicitada.txt"
ARCHIVO_ESTADO="${DIRECTORIO_EJECUCION}/curriculo_automatico_estado.json"
ARCHIVO_PID_SUPERVISOR="${DIRECTORIO_EJECUCION}/curriculo_automatico.pid"
ARCHIVO_PARADA_TOTAL="${DIRECTORIO_EJECUCION}/parada_total_solicitada"
ARCHIVO_PARADA_ENTRENAMIENTO="${DIRECTORIO_EJECUCION}/parada_entrenamiento_solicitada"
mkdir -p "${DIRECTORIO_EJECUCION}"

nombre_fase() {
  case "$1" in
    1) printf '%s' "mantener_pose_xml" ;;
    2) printf '%s' "llegar_desde_suelo" ;;
    3) printf '%s' "recuperar_desde_caida" ;;
  esac
}

escribir_estado() {
  local estado="$1"
  local detalle="${2:-}"
  python3 - "$ARCHIVO_ESTADO" "$estado" "$NOMBRE_EJECUCION" "$DIRECTORIO_EJECUCION" "$FASE" "$PASOS_TOTALES" "$PASOS_BLOQUE" "$PASOS_COMPLETADOS" "$PERFIL_PPO" "$detalle" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ruta, estado, nombre_ejecucion, directorio_ejecucion, fase, total, bloque, completados, perfil, detalle = sys.argv[1:]
ruta = Path(ruta)
if ruta.is_symlink():
    raise SystemExit(f"El estado curricular no puede ser un enlace simbolico: {ruta}")
datos = {}
if ruta.is_file():
    try:
        anterior = json.loads(ruta.read_text(encoding="utf-8"))
        if isinstance(anterior, dict):
            datos.update(anterior)
    except (OSError, ValueError, TypeError):
        pass
datos.update({
    "estado": estado,
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "pid_supervisor": int(os.environ.get("SIM2REAL_CURRICULUM_PID", "0") or 0),
    "nombre_ejecucion": nombre_ejecucion,
    "directorio_ejecucion": directorio_ejecucion,
    "fase_actual": int(fase),
    "fase_nombre": {
        "1": "mantener_pose_xml",
        "2": "llegar_desde_suelo",
        "3": "recuperar_desde_caida",
    }.get(str(fase), "desconocida"),
    "perfil_ppo": perfil,
    "pasos_totales_objetivo": int(total),
    "pasos_por_bloque": int(bloque),
    "pasos_confirmados_por_supervisor": int(completados),
    "pasos_lanzados_por_supervisor": int(completados),
})
if detalle:
    datos["detalle"] = detalle
descriptor, nombre_temporal = tempfile.mkstemp(
    prefix=f".{ruta.name}.", suffix=".tmp", dir=ruta.parent
)
temporal = Path(nombre_temporal)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, indent=2, sort_keys=True)
        archivo.write("\n")
        archivo.flush()
        os.fsync(archivo.fileno())
    os.replace(temporal, ruta)
finally:
    temporal.unlink(missing_ok=True)
PY
}

leer_fase_solicitada() {
  if [[ ! -f "${ARCHIVO_SOLICITUD}" ]]; then
    return 1
  fi
  local solicitada
  solicitada="$(tr -cd '0-9' < "${ARCHIVO_SOLICITUD}" | head -c 1)"
  rm -f "${ARCHIVO_SOLICITUD}"
  case "${solicitada}" in
    1|2|3) printf '%s\n' "${solicitada}"; return 0 ;;
    *) echo "Solicitud de fase ignorada: ${solicitada:-<vacia>}" >&2; return 1 ;;
  esac
}

estado_entrenamiento() {
  python3 - "${DIRECTORIO_EJECUCION}/estado.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError, TypeError):
    print("desconocido")
else:
    print(data.get("estado", "desconocido"))
PY
}

contar_filas_progreso() {
  python3 - "${DIRECTORIO_EJECUCION}/progreso.csv" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print(0)
else:
    with path.open(newline="", encoding="utf-8") as handle:
        print(sum(1 for _ in csv.DictReader(handle)))
PY
}

pasos_confirmados_desde_fila() {
  local primera_fila="$1"
  python3 - "${DIRECTORIO_EJECUCION}/progreso.csv" "${primera_fila}" <<'PY'
import csv
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
offset = int(sys.argv[2])
if not path.is_file():
    print(0)
    raise SystemExit
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))[offset:]
steps = []
for row in rows:
    try:
        value = float(row.get("num_steps", ""))
    except (TypeError, ValueError):
        continue
    if math.isfinite(value) and value >= 0:
        steps.append(int(value))
print(max(steps, default=0))
PY
}

esperar_fin_entrenamiento_o_solicitud() {
  local pid="" launcher_pid="" parada_por_fase_enviada=0
  while :; do
    if [[ -f "${ARCHIVO_PARADA_TOTAL}" ]]; then
      echo "Parada total detectada; no se lanzara ningun bloque nuevo."
      "${SCRIPT_DIR}/sim2real.sh" parar \
        --solo-entrenamiento --ejecucion "${NOMBRE_EJECUCION}" || true
      return 130
    fi
    if [[ -f "${ARCHIVO_SOLICITUD}" && "${PARAR_AL_SOLICITAR}" == "1" &&
          "${parada_por_fase_enviada}" == "0" ]]; then
      echo "Solicitud manual detectada. Parando el bloque actual para cambiar fase..."
      "${SCRIPT_DIR}/sim2real.sh" parar \
        --solo-entrenamiento --ejecucion "${NOMBRE_EJECUCION}" || true
      parada_por_fase_enviada=1
    fi

    pid="$(sim2real_read_safe_pid_file "${DIRECTORIO_EJECUCION}/entrenamiento.pid" 2>/dev/null || true)"
    launcher_pid="$(sim2real_read_safe_pid_file "${DIRECTORIO_EJECUCION}/lanzador.pid" 2>/dev/null || true)"
    if ! sim2real_training_pid_matches_run \
        "${REPO_ROOT}" "${DIRECTORIO_EJECUCION}" "${pid}" &&
        ! sim2real_launcher_pid_matches_run \
          "${REPO_ROOT}" "${DIRECTORIO_EJECUCION}" "${launcher_pid}"; then
      return 0
    fi
    sleep 0.5
  done
}

fase_lista_para_avanzar() {
  local fase="$1"
  python3 - "${DIRECTORIO_EJECUCION}/recompensas.csv" "${fase}" <<'PY'
import csv
import math
import sys
from pathlib import Path

ruta_csv = Path(sys.argv[1])
fase = int(sys.argv[2])
if fase >= 3 or not ruta_csv.exists():
    raise SystemExit(1)

filas = []
with ruta_csv.open(newline="", encoding="utf-8") as archivo:
    for fila in csv.DictReader(archivo):
        if (
            fila.get("source") == "eval"
            and fila.get("fase_curriculum_recompensa") == str(fase)
        ):
            filas.append(fila)
filas = filas[-6:]
if len(filas) < 3:
    print("Criterio: aun hay pocas evaluaciones recientes.")
    raise SystemExit(1)

def media_campo(nombre, predeterminado=math.nan):
    valores = []
    for fila in filas:
        try:
            valor = float(fila.get(nombre, ""))
        except ValueError:
            continue
        if math.isfinite(valor):
            valores.append(valor)
    return sum(valores) / len(valores) if valores else predeterminado

pose = media_campo("state_pose_imitacion_reward")
error_altura = abs(media_campo("state_error_altura_xml", 999.0))
nivel = media_campo("state_cuerpo_paralelo_reward", media_campo("state_level_gate", 0.0))
soporte = media_campo("state_support_gate", 0.0)
penalizacion_rodilla = media_campo("rodilla_suelo_penalty_ponderado", 0.0)
exceso_altura = media_campo("altura_exceso_penalty_ponderado", 0.0)
contactos = media_campo("state_foot_contacts", 0.0)

if fase == 1:
    lista = (
        pose >= 0.72
        and error_altura <= 0.030
        and nivel >= 0.60
        and penalizacion_rodilla <= 0.30
        and exceso_altura <= 0.30
    )
    motivo = (
        f"fase1 pose={pose:.3f} |err_z|={error_altura:.3f} "
        f"nivel={nivel:.3f} rodilla={penalizacion_rodilla:.3f} exceso_z={exceso_altura:.3f}"
    )
else:
    lista = (
        pose >= 0.62
        and error_altura <= 0.040
        and nivel >= 0.55
        and soporte >= 0.35
        and contactos >= 3.0
        and penalizacion_rodilla <= 0.35
        and exceso_altura <= 0.35
    )
    motivo = (
        f"fase2 pose={pose:.3f} |err_z|={error_altura:.3f} "
        f"nivel={nivel:.3f} soporte={soporte:.3f} contactos={contactos:.2f} "
        f"rodilla={penalizacion_rodilla:.3f} exceso_z={exceso_altura:.3f}"
    )

print("Criterio:", motivo)
raise SystemExit(0 if lista else 1)
PY
}

PASOS_COMPLETADOS=0
PRIMER_BLOQUE=1
BLOQUE_ACTIVO=0
PARADA_RECIBIDA=0
ESTADO_FINAL=""
export SIM2REAL_CURRICULUM_PID="$$"

pid_supervisor_anterior="$(sim2real_read_safe_pid_file "${ARCHIVO_PID_SUPERVISOR}" 2>/dev/null || true)"
if sim2real_curriculum_pid_matches_run \
    "${REPO_ROOT}" "${DIRECTORIO_EJECUCION}" "${pid_supervisor_anterior}"; then
  echo "Ya existe un supervisor curricular activo para esta ejecucion: PID ${pid_supervisor_anterior}." >&2
  exit 1
fi
rm -f "${ARCHIVO_PID_SUPERVISOR}" "${ARCHIVO_PARADA_TOTAL}" "${ARCHIVO_PARADA_ENTRENAMIENTO}"
pid_supervisor_temporal="$(mktemp "${DIRECTORIO_EJECUCION}/.curriculo_automatico.pid.XXXXXX")"
printf '%s\n' "$$" > "${pid_supervisor_temporal}"
mv -f -- "${pid_supervisor_temporal}" "${ARCHIVO_PID_SUPERVISOR}"

solicitar_parada_supervisor() {
  local nombre_senal="$1" temporal
  PARADA_RECIBIDA=1
  if [[ ! -e "${ARCHIVO_PARADA_TOTAL}" && ! -L "${ARCHIVO_PARADA_TOTAL}" ]]; then
    temporal="$(mktemp "${DIRECTORIO_EJECUCION}/.parada_total.XXXXXX")"
    printf '%s\t%s\n' "$(date --iso-8601=seconds)" "${nombre_senal}" > "${temporal}"
    mv -f -- "${temporal}" "${ARCHIVO_PARADA_TOTAL}"
  fi
  escribir_estado "cancelando" "senal ${nombre_senal} recibida"
  exit 130
}

finalizar_supervisor() {
  local codigo_salida=$? pid_guardado
  trap - EXIT TERM INT HUP
  if (( BLOQUE_ACTIVO == 1 )); then
    "${SCRIPT_DIR}/sim2real.sh" parar \
      --solo-entrenamiento --ejecucion "${NOMBRE_EJECUCION}" || true
  fi
  if (( PARADA_RECIBIDA == 1 )) || [[ -f "${ARCHIVO_PARADA_TOTAL}" ]]; then
    escribir_estado "cancelado" "supervisor y entrenamiento detenidos"
  elif [[ "${ESTADO_FINAL}" != "terminado" && "${codigo_salida}" -ne 0 ]]; then
    escribir_estado "error" "el supervisor termino con codigo ${codigo_salida}"
  fi
  pid_guardado="$(sim2real_read_safe_pid_file "${ARCHIVO_PID_SUPERVISOR}" 2>/dev/null || true)"
  if [[ "${pid_guardado}" == "$$" ]]; then
    rm -f "${ARCHIVO_PID_SUPERVISOR}"
  fi
  exit "${codigo_salida}"
}

trap finalizar_supervisor EXIT
trap 'solicitar_parada_supervisor SIGTERM' TERM
trap 'solicitar_parada_supervisor SIGINT' INT
trap 'solicitar_parada_supervisor SIGHUP' HUP

escribir_estado "iniciando"

cat <<EOF
Curriculo automatico SIM2REAL
  nombre_ejecucion: ${NOMBRE_EJECUCION}
  perfil_ppo: ${PERFIL_PPO}
  fase inicial: ${FASE} - $(nombre_fase "${FASE}")
  pasos_totales: ${PASOS_TOTALES}
  pasos_por_bloque: ${PASOS_BLOQUE}
  directorio_ejecucion: ${DIRECTORIO_EJECUCION}

Puedes cambiar fase con:
  scripts/cambiar_fase_sim2real.sh
EOF

while (( PASOS_COMPLETADOS < PASOS_TOTALES )); do
  if [[ -f "${ARCHIVO_PARADA_TOTAL}" ]]; then
    PARADA_RECIBIDA=1
    exit 130
  fi
  if solicitada="$(leer_fase_solicitada)"; then
    FASE="${solicitada}"
    echo "Cambio manual aplicado antes del bloque: fase ${FASE} - $(nombre_fase "${FASE}")"
  fi

  restantes=$(( PASOS_TOTALES - PASOS_COMPLETADOS ))
  pasos_bloque_actual="${PASOS_BLOQUE}"
  if (( restantes < PASOS_BLOQUE )); then
    pasos_bloque_actual="${restantes}"
  fi

  echo ""
  echo "=== Bloque fase ${FASE} - $(nombre_fase "${FASE}") | ${pasos_bloque_actual} pasos ==="
  escribir_estado "entrenando_bloque"
  filas_progreso_antes="$(contar_filas_progreso)"
  rm -f "${ARCHIVO_PARADA_ENTRENAMIENTO}"

  argumentos_entrenamiento=(
    entrenar
    --segundo-plano
    --omitir-prueba-mjx
    --nombre-ejecucion "${NOMBRE_EJECUCION}"
    --perfil-ppo "${PERFIL_PPO}"
    --fase-recompensa "${FASE}"
    --num-timesteps "${pasos_bloque_actual}"
    --anexar-csv
  )
  if (( PRIMER_BLOQUE == 1 && PREPARAR_PRIMERO == 1 )); then
    argumentos_entrenamiento+=(--setup)
  fi
  if (( PRIMER_BLOQUE == 1 && REINICIAR_PRIMERO == 1 )); then
    argumentos_entrenamiento+=(--desde-cero)
  fi

  BLOQUE_ACTIVO=1
  SIM2REAL_CURRICULUM_CHILD=1 \
    "${SCRIPT_DIR}/sim2real.sh" "${argumentos_entrenamiento[@]}"
  PRIMER_BLOQUE=0
  set +e
  esperar_fin_entrenamiento_o_solicitud
  estado_espera=$?
  set -e
  pasos_confirmados_bloque="$(pasos_confirmados_desde_fila "${filas_progreso_antes}")"
  PASOS_COMPLETADOS=$(( PASOS_COMPLETADOS + pasos_confirmados_bloque ))
  BLOQUE_ACTIVO=0
  estado_bloque="$(estado_entrenamiento)"

  if [[ "${estado_espera}" -eq 130 || -f "${ARCHIVO_PARADA_TOTAL}" ]]; then
    PARADA_RECIBIDA=1
    exit 130
  fi

  if solicitada="$(leer_fase_solicitada)"; then
    case "${estado_bloque}" in
      terminado|cancelado) ;;
      *)
        echo "El bloque no termino de forma recuperable: estado ${estado_bloque}." >&2
        exit 1
        ;;
    esac
    FASE="${solicitada}"
    rm -f "${ARCHIVO_PARADA_ENTRENAMIENTO}"
    escribir_estado \
      "bloque_cancelado_por_cambio_de_fase" \
      "${pasos_confirmados_bloque} pasos confirmados antes del cambio"
    echo "Cambio manual aplicado: fase ${FASE} - $(nombre_fase "${FASE}"); ${pasos_confirmados_bloque} pasos confirmados."
    continue
  fi

  if [[ "${estado_bloque}" != "terminado" ]]; then
    echo "El entrenamiento termino con estado ${estado_bloque}; el supervisor no relanzara otro bloque." >&2
    exit 1
  fi
  if (( pasos_confirmados_bloque <= 0 )); then
    echo "El bloque figura como terminado, pero no registro ningun paso positivo; no se contabiliza ni se relanza." >&2
    exit 1
  fi
  rm -f "${ARCHIVO_PARADA_ENTRENAMIENTO}"
  escribir_estado \
    "bloque_terminado" \
    "${pasos_confirmados_bloque} pasos confirmados en el ultimo bloque"

  if fase_lista_para_avanzar "${FASE}"; then
    if (( FASE < 3 )); then
      FASE=$(( FASE + 1 ))
      echo "Criterio cumplido. Subiendo automaticamente a fase ${FASE} - $(nombre_fase "${FASE}")"
    fi
  else
    echo "Criterio no cumplido. Seguimos en fase ${FASE}."
  fi
done

escribir_estado "terminado"
ESTADO_FINAL="terminado"
echo "Curriculo automatico terminado: ${DIRECTORIO_EJECUCION}"
