#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
XML_IDEAL="${REPO_ROOT}/sim2real_mjx/xmls/ROBOT_POSE_IDEAL.xml"

cd "${REPO_ROOT}"

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

if [[ ! -f "${XML_IDEAL}" ]]; then
  echo "No encuentro el XML ideal: ${XML_IDEAL}" >&2
  exit 1
fi

echo ""
echo "Pose inicial de la minisimulacion:"
echo "  1) Actual del checkpoint: aleatoria a cierta altura"
echo "  2) Tumbada en suelo, tipo POSE_SUELO2"
echo "  3) Pose ideal de ROBOT_POSE_IDEAL.xml"
echo "  4) Caida de lado, muy inclinada y aleatoria"
echo "  5) Boca abajo en postura aleatoria"
echo ""

eleccion_postura="$(elegir_de_lista "Elige pose inicial [1-5]: " 5)"
case "${eleccion_postura}" in
  1) postura_inicial="actual" ;;
  2) postura_inicial="suelo2" ;;
  3) postura_inicial="ideal" ;;
  4) postura_inicial="caida_lateral" ;;
  5) postura_inicial="boca_abajo" ;;
esac

argumentos_adicionales=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ruta-xml)
      shift 2
      ;;
    *)
      argumentos_adicionales+=("$1")
      shift
      ;;
  esac
done

echo "Minisimulando ultimo checkpoint con XML fijo: ${XML_IDEAL}"
echo "Pose inicial: ${postura_inicial}"
exec ./scripts/sim2real.sh minisimular \
  --ruta-xml "${XML_IDEAL}" \
  --postura-inicial "${postura_inicial}" \
  "${argumentos_adicionales[@]}"
