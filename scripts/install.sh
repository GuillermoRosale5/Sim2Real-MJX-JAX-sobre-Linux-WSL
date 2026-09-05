#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/lib/platform.sh
source "${SCRIPT_DIR}/lib/platform.sh"

ACCELERATOR="${SIM2REAL_ACCELERATOR:-auto}"
INSTALL_SYSTEM_PACKAGES=1
RUN_DOCTOR=1
RUN_SETUP=1

usage() {
  cat <<'EOF'
Uso:
  ./scripts/install.sh [--accelerator compatible|auto|nvidia|amd|intel|cpu]
                       [--skip-system-packages] [--no-setup]

Instala SIM2REAL en Ubuntu nativo o Ubuntu sobre WSL2. `compatible` es el modo
universal del tutorial: utiliza NVIDIA/CUDA o AMD/ROCm cuando estan disponibles
y, si no, instala CPU mostrando un aviso. `auto` conserva el comportamiento
estricto y exige una GPU operativa.
Intel GPU y AMD bajo WSL2 no se aceptan como perfiles GPU. `compatible` continúa
por CPU con un aviso; una solicitud GPU explícita se bloquea.

Opciones tecnicas:
  --skip-system-packages  No repite la instalacion de paquetes APT del sistema.
  --no-setup              Solo paquetes del sistema y permisos; no crea `.venvs/...`.
EOF
  sim2real_print_accelerator_support_summary
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --accelerator)
      if [[ $# -lt 2 || "$2" == -* ]]; then
        echo "Falta el valor de --accelerator. Usa compatible, auto, nvidia, amd, intel o cpu." >&2
        usage >&2
        exit 1
      fi
      ACCELERATOR="$2"
      shift 2
      ;;
    --accelerator=*) ACCELERATOR="${1#*=}"; shift ;;
    --skip-system-packages) INSTALL_SYSTEM_PACKAGES=0; shift ;;
    --no-setup) RUN_SETUP=0; RUN_DOCTOR=0; shift ;;
    --allow-experimental-amd-wsl)
      echo "AMD/JAX bajo WSL2 ya no se habilita: AMD lo declara no validado. Usa Ubuntu nativo con ROCm." >&2
      sim2real_print_accelerator_support_summary >&2
      exit 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Argumento no reconocido: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if (( EUID == 0 )); then
  cat >&2 <<'EOF'
No ejecutes todo el instalador como root ni con `sudo`.
Ejecuta `./scripts/install.sh ...` con tu usuario normal; el propio script usa
sudo solamente para instalar los paquetes APT que lo necesitan.
EOF
  exit 1
fi

sim2real_require_supported_host
sim2real_require_workspace "${REPO_ROOT}"
sim2real_validate_accelerator_request "${ACCELERATOR}"

if ! sim2real_acquire_runtime_lock exclusive "${REPO_ROOT}" nonblocking; then
  ACTIVE_TRAINING="$(sim2real_active_training "${REPO_ROOT}" 2>/dev/null || true)"
  if [[ -n "${ACTIVE_TRAINING}" ]]; then
    IFS=$'\t' read -r ACTIVE_RUN ACTIVE_PID <<< "${ACTIVE_TRAINING}"
    echo "Hay un entrenamiento activo (PID ${ACTIVE_PID}, run ${ACTIVE_RUN}). Detenlo con ./scripts/parar_sim2real.sh antes de instalar o cambiar el perfil." >&2
  else
    echo "Otro comando Sim2Real MJX-JAX esta usando o instalando el entorno. Cierralo y vuelve a ejecutar el instalador." >&2
  fi
  exit 1
fi

ENVIRONMENTS_ROOT="$(sim2real_environments_root "${REPO_ROOT}")"
mkdir -p -- "${ENVIRONMENTS_ROOT}"
[[ -d "${ENVIRONMENTS_ROOT}" && ! -L "${ENVIRONMENTS_ROOT}" ]] || {
  echo "La raiz de entornos no es un directorio seguro: ${ENVIRONMENTS_ROOT}" >&2
  exit 1
}

sim2real_restore_profile_from_transaction() {
  local transaction_dir="$1" target_accelerator="$2"
  local previous_profile="" current_profile="" profile_file profile_dir
  profile_file="$(sim2real_profile_file "${REPO_ROOT}")"
  profile_dir="$(dirname -- "${profile_file}")"
  current_profile="$(sim2real_saved_accelerator "${REPO_ROOT}" 2>/dev/null || true)"
  if [[ -f "${transaction_dir}/previous-profile" &&
        ! -L "${transaction_dir}/previous-profile" ]]; then
    IFS= read -r previous_profile < "${transaction_dir}/previous-profile" || true
    sim2real_validate_accelerator "${previous_profile}" >/dev/null 2>&1 &&
      [[ "${previous_profile}" != "auto" ]] || {
        echo "Perfil anterior invalido en ${transaction_dir}; se conserva la transaccion." >&2
        return 1
      }
    if [[ -z "${current_profile}" || "${current_profile}" == "${target_accelerator}" ||
          "${current_profile}" == "${previous_profile}" ]]; then
      sim2real_save_accelerator "${previous_profile}" "${REPO_ROOT}"
    else
      echo "Se conserva el perfil activo ${current_profile}; es posterior o externo a la transaccion ${target_accelerator}."
    fi
  elif [[ -f "${transaction_dir}/no-previous-profile" &&
          ! -L "${transaction_dir}/no-previous-profile" ]]; then
    if [[ "${current_profile}" == "${target_accelerator}" ]]; then
      [[ ! -L "${profile_dir}" && ! -L "${profile_file}" ]] || {
        echo "No se restaura el perfil porque su ruta contiene un enlace simbolico." >&2
        return 1
      }
      rm -f -- "${profile_file}"
    elif [[ -n "${current_profile}" ]]; then
      echo "Se conserva el perfil activo ${current_profile}; es posterior o externo a la transaccion ${target_accelerator}."
    fi
  else
    echo "Falta el estado del perfil anterior en ${transaction_dir}; no se adivina." >&2
    return 1
  fi
}

sim2real_remove_partial_environment() {
  local environment_dir="$1"
  if [[ -L "${environment_dir}" ]]; then
    echo "No se elimina un entorno parcial que es enlace simbolico: ${environment_dir}" >&2
    return 1
  fi
  if [[ -e "${environment_dir}" ]]; then
    rm -rf -- "${environment_dir}" || {
      echo "No se pudo retirar el entorno parcial: ${environment_dir}" >&2
      return 1
    }
  fi
}

sim2real_recover_profile_transactions() {
  local accelerator="$1" environment_slug environment_dir
  local transaction_dir base_name backup_dir
  local -a interrupted=() committed=()
  environment_slug="$(sim2real_environment_slug "${accelerator}")" || return 1
  environment_dir="$(sim2real_environment_dir "${accelerator}" "${REPO_ROOT}")" || return 1
  [[ "$(realpath -m "${environment_dir}")" == \
     "$(realpath -e "${ENVIRONMENTS_ROOT}")/${environment_slug}" ]] || {
    echo "La ruta del entorno escapa de .venvs: ${environment_dir}" >&2
    return 1
  }
  mapfile -d '' -t interrupted < <(
    find "${ENVIRONMENTS_ROOT}" -mindepth 1 -maxdepth 1 \
      -name ".install-${environment_slug}.*" -print0
  )
  mapfile -d '' -t committed < <(
    find "${ENVIRONMENTS_ROOT}" -mindepth 1 -maxdepth 1 \
      -name ".committed-${environment_slug}.*" -print0
  )
  if (( ${#interrupted[@]} > 1 )); then
    echo "Hay varias instalaciones interrumpidas de ${accelerator}; se conservan para revision:" >&2
    printf '  %s\n' "${interrupted[@]}" >&2
    return 1
  fi

  for transaction_dir in "${committed[@]}"; do
    base_name="$(basename -- "${transaction_dir}")"
    [[ "$(dirname -- "${transaction_dir}")" == "${ENVIRONMENTS_ROOT}" &&
       "${base_name}" == ".committed-${environment_slug}."* &&
       -d "${transaction_dir}" && ! -L "${transaction_dir}" &&
       -f "${transaction_dir}/commit-complete" &&
       ! -L "${transaction_dir}/commit-complete" ]] || {
      echo "Transaccion completada con estructura insegura: ${transaction_dir}" >&2
      return 1
    }
    [[ -d "${environment_dir}" && ! -L "${environment_dir}" ]] || {
      echo "Se conserva ${transaction_dir}: el entorno validado ya no esta en ${environment_dir}." >&2
      return 1
    }
    rm -rf -- "${transaction_dir}" || return 1
    echo "Limpieza recuperada de una instalacion ya validada: ${transaction_dir}"
  done

  for transaction_dir in "${interrupted[@]}"; do
    base_name="$(basename -- "${transaction_dir}")"
    [[ "$(dirname -- "${transaction_dir}")" == "${ENVIRONMENTS_ROOT}" &&
       "${base_name}" == ".install-${environment_slug}."* &&
       -d "${transaction_dir}" && ! -L "${transaction_dir}" ]] || {
      echo "Transaccion interrumpida con estructura insegura: ${transaction_dir}" >&2
      return 1
    }
    if [[ ! -f "${transaction_dir}/initialized" ]]; then
      rm -rf -- "${transaction_dir}" || return 1
      echo "Se retiro una transaccion vacia interrumpida: ${transaction_dir}"
      continue
    fi
    [[ ! -L "${transaction_dir}/initialized" ]] || return 1
    if [[ -f "${transaction_dir}/commit-complete" &&
          ! -L "${transaction_dir}/commit-complete" ]]; then
      [[ -d "${environment_dir}" && ! -L "${environment_dir}" ]] || {
        echo "Se conserva ${transaction_dir}: el commit existe, pero falta ${environment_dir}." >&2
        return 1
      }
      rm -rf -- "${transaction_dir}" || return 1
      echo "Limpieza recuperada de una instalacion validada: ${transaction_dir}"
      continue
    fi

    backup_dir="${transaction_dir}/previous"
    if [[ -e "${backup_dir}" || -L "${backup_dir}" ]]; then
      [[ -d "${backup_dir}" && ! -L "${backup_dir}" ]] || {
        echo "La copia anterior no es un directorio seguro: ${backup_dir}" >&2
        return 1
      }
      sim2real_remove_partial_environment "${environment_dir}" || return 1
      mv -T -- "${backup_dir}" "${environment_dir}" || {
        echo "No se pudo restaurar ${backup_dir}; no se elimina la transaccion." >&2
        return 1
      }
      echo "Entorno ${accelerator} anterior recuperado tras una interrupcion: ${environment_dir}"
    elif [[ -f "${transaction_dir}/had-previous-environment" &&
            ! -L "${transaction_dir}/had-previous-environment" ]]; then
      [[ -d "${environment_dir}" && ! -L "${environment_dir}" ]] || {
        echo "La transaccion indica un entorno anterior, pero no aparece ni en destino ni en backup: ${transaction_dir}" >&2
        return 1
      }
    else
      sim2real_remove_partial_environment "${environment_dir}" || return 1
    fi
    sim2real_restore_profile_from_transaction "${transaction_dir}" "${accelerator}" || return 1
    rm -rf -- "${transaction_dir}" || return 1
    echo "Instalacion ${accelerator} interrumpida recuperada antes de continuar."
  done
}

sim2real_recover_install_transactions() {
  local accelerator
  for accelerator in nvidia amd cpu; do
    sim2real_recover_profile_transactions "${accelerator}" || return 1
  done
}

# La recuperación es global y no depende de la red, uv, el controlador ni del
# perfil que se vaya a solicitar ahora. Así ningún entorno anterior queda
# oculto si algo falla
# en las comprobaciones posteriores.
sim2real_recover_install_transactions

ACTIVE_TRAINING="$(sim2real_active_training "${REPO_ROOT}" 2>/dev/null || true)"
if [[ -n "${ACTIVE_TRAINING}" ]]; then
  IFS=$'\t' read -r ACTIVE_RUN ACTIVE_PID <<< "${ACTIVE_TRAINING}"
  echo "Hay un entrenamiento activo (PID ${ACTIVE_PID}, run ${ACTIVE_RUN}). Detenlo con ./scripts/parar_sim2real.sh antes de instalar o cambiar el perfil." >&2
  exit 1
fi

PREFLIGHT_STATUS=0
if [[ "${ACCELERATOR}" == "compatible" ]]; then
  if RESOLVED_ACCELERATOR="$(sim2real_detect_compatible_accelerator)"; then
    :
  else
    PREFLIGHT_STATUS=$?
    sim2real_print_accelerator_support_summary >&2
    exit "${PREFLIGHT_STATUS}"
  fi
elif [[ "${ACCELERATOR}" == "auto" ]]; then
  # Una instalación automática vuelve a comprobar el equipo y exige una GPU real.
  if RESOLVED_ACCELERATOR="$(sim2real_detect_accelerator)"; then
    :
  else
    PREFLIGHT_STATUS=$?
    sim2real_print_accelerator_support_summary >&2
    exit "${PREFLIGHT_STATUS}"
  fi
else
  if RESOLVED_ACCELERATOR="$(sim2real_resolve_accelerator "${ACCELERATOR}" "${REPO_ROOT}")"; then
    :
  else
    PREFLIGHT_STATUS=$?
    sim2real_print_accelerator_support_summary >&2
    exit "${PREFLIGHT_STATUS}"
  fi
fi
HOST_KIND="$(sim2real_detect_host)"

if [[ "${HOST_KIND}" == "wsl" ]]; then
  PLATFORM_CONTEXT="Ubuntu sobre WSL2"
else
  HOST_VIRTUALIZATION="$(sim2real_detect_virtualization)"
  case "${HOST_VIRTUALIZATION}" in
    none) PLATFORM_CONTEXT="Ubuntu nativo" ;;
    oracle) PLATFORM_CONTEXT="Ubuntu en VM (VirtualBox/Oracle)" ;;
    microsoft) PLATFORM_CONTEXT="Ubuntu en VM (Hyper-V/Microsoft)" ;;
    *) PLATFORM_CONTEXT="Ubuntu en VM (${HOST_VIRTUALIZATION})" ;;
  esac
fi

echo "Plataforma: $(sim2real_os_pretty_name) / ${PLATFORM_CONTEXT}"
echo "Acelerador solicitado: ${ACCELERATOR}"
echo "Acelerador resuelto: ${RESOLVED_ACCELERATOR}"
echo "Nivel de soporte: $(sim2real_support_level "${RESOLVED_ACCELERATOR}" "${HOST_KIND}")"

if sim2real_check_profile_host "${RESOLVED_ACCELERATOR}" "${HOST_KIND}" "${REPO_ROOT}"; then
  :
else
  PREFLIGHT_STATUS=$?
  sim2real_print_accelerator_support_summary >&2
  exit "${PREFLIGHT_STATUS}"
fi
ENVIRONMENT_DIR="$(sim2real_environment_dir "${RESOLVED_ACCELERATOR}" "${REPO_ROOT}")"
ENVIRONMENT_PYTHON="$(sim2real_python_executable "${RESOLVED_ACCELERATOR}" "${REPO_ROOT}")"

if (( INSTALL_SYSTEM_PACKAGES == 1 )); then
  command -v sudo >/dev/null 2>&1 || {
    echo "No encuentro sudo. Instala los paquetes base como administrador o usa --skip-system-packages." >&2
    exit 1
  }
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates curl git build-essential python3 python3-venv python3-pip \
    pkg-config pciutils util-linux libgl1 libegl1 libglfw3 libglew2.2
fi

chmod +x "${REPO_ROOT}"/scripts/*.sh "${REPO_ROOT}"/scripts/lib/*.sh
if (( RUN_SETUP == 0 )); then
  echo "Paquetes base y permisos preparados. Setup Python omitido por --no-setup."
  exit 0
fi

export PATH="${HOME}/.local/bin:${PATH}"
UV_REQUIRED_VERSION="0.11.8"
UV_CURRENT_VERSION="$(uv --version 2>/dev/null | awk '{print $2}' || true)"
if [[ "${UV_CURRENT_VERSION}" != "${UV_REQUIRED_VERSION}" ]]; then
  echo "Instalando uv ${UV_REQUIRED_VERSION} (encontrado: ${UV_CURRENT_VERSION:-ninguno})..."
  export UV_NO_MODIFY_PATH=1
  curl -LsSf "https://astral.sh/uv/${UV_REQUIRED_VERSION}/install.sh" | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  hash -r
fi
UV_CURRENT_VERSION="$(uv --version 2>/dev/null | awk '{print $2}' || true)"
[[ "${UV_CURRENT_VERSION}" == "${UV_REQUIRED_VERSION}" ]] || {
  echo "Se requiere uv ${UV_REQUIRED_VERSION}; version activa: ${UV_CURRENT_VERSION:-no disponible}." >&2
  exit 1
}

cd "${REPO_ROOT}"
UV_ARGS=(sync --frozen)
while IFS= read -r extra_arg; do
  [[ -n "${extra_arg}" ]] && UV_ARGS+=("${extra_arg}")
done < <(sim2real_uv_extra_args "${RESOLVED_ACCELERATOR}" || true)

if [[ ! -f "${REPO_ROOT}/uv.lock" ]]; then
  echo "Falta uv.lock; generandolo desde pyproject.toml..."
  uv lock
else
  uv lock --check || {
    echo "uv.lock no coincide con pyproject.toml. No se actualiza implicitamente en una instalacion reproducible." >&2
    exit 1
  }
fi

ENVIRONMENTS_ROOT="$(sim2real_environments_root "${REPO_ROOT}")"
ENVIRONMENT_SLUG="$(sim2real_environment_slug "${RESOLVED_ACCELERATOR}")"
mkdir -p "${ENVIRONMENTS_ROOT}"
[[ -d "${ENVIRONMENTS_ROOT}" && ! -L "${ENVIRONMENTS_ROOT}" ]] || {
  echo "La raiz de entornos no es un directorio seguro: ${ENVIRONMENTS_ROOT}" >&2
  exit 1
}
[[ "$(realpath -m "${ENVIRONMENT_DIR}")" == "$(realpath -e "${ENVIRONMENTS_ROOT}")/${ENVIRONMENT_SLUG}" ]] || {
  echo "La ruta del entorno escapa de .venvs: ${ENVIRONMENT_DIR}" >&2
  exit 1
}
[[ ! -L "${ENVIRONMENT_DIR}" ]] || {
  echo "El entorno no puede ser un enlace simbolico: ${ENVIRONMENT_DIR}" >&2
  exit 1
}

INSTALL_TRANSACTION_DIR="$(mktemp -d "${ENVIRONMENTS_ROOT}/.install-${ENVIRONMENT_SLUG}.XXXXXX")"
INSTALL_BACKUP_DIR="${INSTALL_TRANSACTION_DIR}/previous"
INSTALL_TRANSACTION_ACTIVE=1
INSTALL_EXISTING_MOVED=0
INSTALL_NEW_ENV_STARTED=0
INSTALL_TRANSACTION_SUCCEEDED=0
PREVIOUS_PROFILE="$(sim2real_saved_accelerator "${REPO_ROOT}" 2>/dev/null || true)"
if [[ -n "${PREVIOUS_PROFILE}" ]]; then
  printf '%s\n' "${PREVIOUS_PROFILE}" > "${INSTALL_TRANSACTION_DIR}/previous-profile"
else
  : > "${INSTALL_TRANSACTION_DIR}/no-previous-profile"
fi
if [[ -e "${ENVIRONMENT_DIR}" ]]; then
  : > "${INSTALL_TRANSACTION_DIR}/had-previous-environment"
fi
: > "${INSTALL_TRANSACTION_DIR}/initialized"

sim2real_install_transaction_cleanup() {
  local status=$? rollback_failed=0
  trap - EXIT
  trap - HUP INT TERM
  if (( INSTALL_TRANSACTION_ACTIVE == 1 )); then
    if (( INSTALL_TRANSACTION_SUCCEEDED == 0 )); then
      echo "La instalacion del perfil ${RESOLVED_ACCELERATOR} fallo; restaurando el entorno anterior." >&2
      if (( INSTALL_NEW_ENV_STARTED == 1 )); then
        sim2real_remove_partial_environment "${ENVIRONMENT_DIR}" || rollback_failed=1
      fi
      if (( INSTALL_EXISTING_MOVED == 1 )); then
        if [[ -e "${ENVIRONMENT_DIR}" || -L "${ENVIRONMENT_DIR}" ]]; then
          echo "No se restaura encima de una ruta ocupada. La copia anterior sigue en ${INSTALL_BACKUP_DIR}." >&2
          rollback_failed=1
        elif ! mv -T -- "${INSTALL_BACKUP_DIR}" "${ENVIRONMENT_DIR}"; then
          echo "No se pudo restaurar el entorno. La copia anterior se conserva en ${INSTALL_BACKUP_DIR}." >&2
          rollback_failed=1
        else
          echo "Entorno anterior restaurado: ${ENVIRONMENT_DIR}" >&2
        fi
      fi
      sim2real_restore_profile_from_transaction \
        "${INSTALL_TRANSACTION_DIR}" "${RESOLVED_ACCELERATOR}" || rollback_failed=1
    fi
    if (( rollback_failed == 0 )); then
      rm -rf -- "${INSTALL_TRANSACTION_DIR}" || {
        echo "No se pudo limpiar ${INSTALL_TRANSACTION_DIR}; se conserva para recuperarlo en la siguiente instalacion." >&2
        rollback_failed=1
      }
    fi
  fi
  if (( rollback_failed == 1 )); then
    exit 90
  fi
  exit "${status}"
}
trap sim2real_install_transaction_cleanup EXIT

if [[ -e "${ENVIRONMENT_DIR}" ]]; then
  [[ -d "${ENVIRONMENT_DIR}" && ! -L "${ENVIRONMENT_DIR}" ]] || {
    echo "El entorno existente no es un directorio seguro: ${ENVIRONMENT_DIR}" >&2
    exit 1
  }
  mv -T -- "${ENVIRONMENT_DIR}" "${INSTALL_BACKUP_DIR}"
  INSTALL_EXISTING_MOVED=1
fi
INSTALL_NEW_ENV_STARTED=1
UV_PROJECT_ENVIRONMENT="${ENVIRONMENT_DIR}" uv "${UV_ARGS[@]}"

if [[ "${RESOLVED_ACCELERATOR}" == "amd" ]]; then
  # JAX 0.6.2 para ROCm 7 necesita el jaxlib parcheado publicado por ROCm.
  # No se usa jax[rocm], porque resolvería una combinación distinta o no validada.
  uv pip install --python "${ENVIRONMENT_PYTHON}" --reinstall --no-deps \
    --requirement "${REPO_ROOT}/requirements/amd-rocm70-py312.txt"
fi

echo "Entorno instalado en ${ENVIRONMENT_DIR}"
if (( RUN_DOCTOR == 1 )); then
  SIM2REAL_ACCELERATOR="${RESOLVED_ACCELERATOR}" \
    "${SCRIPT_DIR}/doctor.sh" --accelerator "${RESOLVED_ACCELERATOR}"
  # El doctor ya ha terminado: el guardado del perfil y el cambio de estado de
  # la transacción forman una confirmación corta que no se deja a medias por señales.
  trap '' HUP INT TERM
  sim2real_save_accelerator "${RESOLVED_ACCELERATOR}" "${REPO_ROOT}"
  : > "${INSTALL_TRANSACTION_DIR}/commit-complete"
  INSTALL_COMMITTED_DIR="${ENVIRONMENTS_ROOT}/.committed-${ENVIRONMENT_SLUG}.${INSTALL_TRANSACTION_DIR##*.}"
  [[ ! -e "${INSTALL_COMMITTED_DIR}" && ! -L "${INSTALL_COMMITTED_DIR}" ]]
  mv -T -- "${INSTALL_TRANSACTION_DIR}" "${INSTALL_COMMITTED_DIR}"
  INSTALL_TRANSACTION_DIR="${INSTALL_COMMITTED_DIR}"
  INSTALL_BACKUP_DIR="${INSTALL_TRANSACTION_DIR}/previous"
  INSTALL_TRANSACTION_SUCCEEDED=1
  trap - HUP INT TERM
  echo "Perfil activo: ${RESOLVED_ACCELERATOR} (${ENVIRONMENT_DIR})"
else
  echo "Perfil no activado porque falta la validacion obligatoria." >&2
fi
