#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

FASE_RECOMPENSA="${SIM2REAL_FASE_RECOMPENSA:-}"
PERFIL_PPO="${SIM2REAL_PERFIL_PPO:-}"
PERFIL_PPO_PASADO=0
FASE_RECOMPENSA_PASADA=0

usage() {
  cat <<'EOF'
Uso:
  scripts/lanzar_sim2real.sh [opciones]

Sin opciones abre menus por numeros para perfil PPO y fase curricular.

Opciones utiles:
  --perfil-ppo depuracion|ligero|ligero_rapido|completo
  --fase-recompensa 1|2|3|auto

Fase auto lanza scripts/curriculo_automatico_sim2real.sh:
  fases 1->2->3, 200M pasos totales, cambio automatico por rendimiento.
EOF
}

for arg in "$@"; do
  case "${arg}" in
    --help|-h)
      usage
      exit 0
      ;;
    --perfil-ppo|--perfil-ppo=*)
      PERFIL_PPO_PASADO=1
      ;;
    --fase-recompensa|--fase-recompensa=*)
      FASE_RECOMPENSA_PASADA=1
      ;;
  esac
done

argumento_anterior=""
for arg in "$@"; do
  case "${argumento_anterior}" in
    --perfil-ppo)
      PERFIL_PPO="${arg}"
      argumento_anterior=""
      continue
      ;;
    --fase-recompensa)
      FASE_RECOMPENSA="${arg}"
      argumento_anterior=""
      continue
      ;;
  esac
  case "${arg}" in
    --perfil-ppo|--fase-recompensa)
      argumento_anterior="${arg}"
      ;;
    --perfil-ppo=*)
      PERFIL_PPO="${arg#*=}"
      ;;
    --fase-recompensa=*)
      FASE_RECOMPENSA="${arg#*=}"
      ;;
  esac
done

elegir_perfil_ppo() {
  cat >&2 <<'EOF'
Perfiles PPO:
  1) depuracion     5M pasos, 20 evaluaciones, episodio 1000, red 256-128    | prueba rapida
  2) ligero         100M pasos, 200 evaluaciones, episodio 1500, red 256-256 | recomendado
  3) ligero_rapido  100M pasos, 200 evaluaciones, episodio 1500, red 256-256, 1024 entornos

  completo existe para ejecuciones finales: usa --perfil-ppo completo si lo quieres.
EOF
  local opcion
  read -r -p "Elige perfil PPO [2]: " opcion < /dev/tty
  case "${opcion:-2}" in
    1|depuracion) printf '%s\n' "depuracion" ;;
    2|ligero) printf '%s\n' "ligero" ;;
    3|ligero_rapido) printf '%s\n' "ligero_rapido" ;;
    *)
      echo "Opcion no reconocida: ${opcion}. Usa 1, 2 o 3." >&2
      exit 1
      ;;
  esac
}

elegir_fase_recompensa() {
  cat >&2 <<'EOF'
Fases curriculares de recompensa:
  1) mantener_pose_xml        empieza cerca de la pose ideal y aprende a quedarse ahi
  2) llegar_desde_suelo       empieza peor/en suelo y debe volver a la pose XML
  3) recuperar_desde_caida    recuperacion robusta desde caidas/perturbaciones
  4) auto_3_fases_200M        supervisor: fases 1->2->3, 200M pasos, cambio automatico

  0 existe como base_actual sin curriculo: usa --fase-recompensa 0 si lo necesitas.
EOF
  local opcion
  read -r -p "Elige fase curricular [1]: " opcion < /dev/tty
  case "${opcion:-1}" in
    0|base|base_actual) printf '%s\n' "0" ;;
    1|mantener|pose|pose_xml) printf '%s\n' "1" ;;
    2|suelo|llegar) printf '%s\n' "2" ;;
    3|caida|recuperar) printf '%s\n' "3" ;;
    4|auto|curriculo|auto_3_fases|auto-3-fases) printf '%s\n' "auto" ;;
    *)
      echo "Opcion no reconocida: ${opcion}. Usa 1, 2, 3 o 4." >&2
      exit 1
      ;;
  esac
}

if (( PERFIL_PPO_PASADO == 0 )) && [[ -z "${PERFIL_PPO}" ]]; then
  if [[ -t 0 && -t 1 ]]; then
    PERFIL_PPO="$(elegir_perfil_ppo)"
  else
    PERFIL_PPO="ligero"
  fi
fi

if (( FASE_RECOMPENSA_PASADA == 0 )) && [[ -z "${FASE_RECOMPENSA}" ]]; then
  if [[ -t 0 && -t 1 ]]; then
    FASE_RECOMPENSA="$(elegir_fase_recompensa)"
  else
    FASE_RECOMPENSA="1"
  fi
fi

if [[ "${FASE_RECOMPENSA:-}" == "auto" ]]; then
  comando=(./scripts/curriculo_automatico_sim2real.sh \
    --perfil-ppo "${PERFIL_PPO:-ligero}" \
    --pasos-totales 200000000)
  export SIM2REAL_SHOW_PPO_PROFILES=0
  exec "${comando[@]}"
fi

comando=(./scripts/sim2real.sh entrenar \
  --segundo-plano \
  --setup)

if (( FASE_RECOMPENSA_PASADA == 0 )); then
  comando+=(--fase-recompensa "${FASE_RECOMPENSA:-1}")
fi

if (( PERFIL_PPO_PASADO == 0 )); then
  comando+=(--perfil-ppo "${PERFIL_PPO:-ligero}")
fi

export SIM2REAL_SHOW_PPO_PROFILES=0
exec "${comando[@]}" "$@"
