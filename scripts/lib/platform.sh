#!/usr/bin/env bash
# Funciones compartidas de detección de plataforma y acelerador.
# Este archivo se importa desde otros scripts; no debe activar `set -e`.

sim2real_repo_root() {
  local source_dir
  source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  cd -- "${source_dir}/../.." && pwd
}

sim2real_detect_host() {
  if [[ "$(uname -s 2>/dev/null || true)" != "Linux" ]]; then
    printf '%s\n' "unsupported"
  elif [[ -n "${WSL_INTEROP:-}" ]] || grep -qiE '(microsoft|wsl)' /proc/sys/kernel/osrelease /proc/version 2>/dev/null; then
    printf '%s\n' "wsl"
  else
    printf '%s\n' "ubuntu"
  fi
}

sim2real_detect_wsl_version() {
  local release
  if [[ "$(sim2real_detect_host)" != "wsl" ]]; then
    printf '%s\n' "none"
    return 0
  fi
  release="$(uname -r 2>/dev/null || true)"
  if [[ "${release,,}" == *microsoft-standard* || "${release,,}" == *wsl2* ]]; then
    printf '%s\n' "2"
  else
    # Los kernels de WSL1 incluyen Microsoft/WSL, pero no el identificador
    # microsoft-standard-WSL2. Es preferible rechazar un equipo ambiguo a
    # prometer una pila GPU que WSL1 no puede proporcionar.
    printf '%s\n' "1"
  fi
}

sim2real_detect_virtualization() {
  local virtualization="" dmi_hint="" dmi_file
  if [[ "$(sim2real_detect_host)" == "wsl" ]]; then
    printf '%s\n' "wsl"
    return 0
  fi
  if command -v systemd-detect-virt >/dev/null 2>&1; then
    virtualization="$(systemd-detect-virt --vm 2>/dev/null || true)"
    virtualization="${virtualization%%$'\n'*}"
    virtualization="${virtualization//[[:space:]]/}"
    if [[ -n "${virtualization}" && "${virtualization,,}" != "none" ]]; then
      printf '%s\n' "${virtualization,,}"
      return 0
    fi
  fi
  for dmi_file in \
    /sys/class/dmi/id/sys_vendor \
    /sys/class/dmi/id/product_name \
    /sys/class/dmi/id/board_vendor; do
    [[ -r "${dmi_file}" ]] || continue
    dmi_hint+=" $(<"${dmi_file}")"
  done
  case "${dmi_hint,,}" in
    *virtualbox*|*innotek*) printf '%s\n' "oracle" ;;
    *vmware*) printf '%s\n' "vmware" ;;
    *qemu*) printf '%s\n' "qemu" ;;
    *kvm*) printf '%s\n' "kvm" ;;
    *microsoft*virtual*) printf '%s\n' "microsoft" ;;
    *parallels*) printf '%s\n' "parallels" ;;
    *xen*) printf '%s\n' "xen" ;;
    *) printf '%s\n' "none" ;;
  esac
}

sim2real_os_id() {
  if [[ -r /etc/os-release ]]; then
    (
      # shellcheck disable=SC1091
      source /etc/os-release
      printf '%s\n' "${ID:-unknown}"
    )
  else
    printf '%s\n' "unknown"
  fi
}

sim2real_os_pretty_name() {
  if [[ -r /etc/os-release ]]; then
    (
      # shellcheck disable=SC1091
      source /etc/os-release
      printf '%s\n' "${PRETTY_NAME:-Linux}"
    )
  else
    printf '%s\n' "Linux"
  fi
}

sim2real_os_version_id() {
  if [[ -r /etc/os-release ]]; then
    (
      # shellcheck disable=SC1091
      source /etc/os-release
      printf '%s\n' "${VERSION_ID:-unknown}"
    )
  else
    printf '%s\n' "unknown"
  fi
}

sim2real_require_supported_host() {
  local host os_id os_version architecture wsl_version
  host="$(sim2real_detect_host)"
  os_id="$(sim2real_os_id)"
  os_version="$(sim2real_os_version_id)"
  architecture="$(uname -m 2>/dev/null || true)"
  if [[ "${host}" == "unsupported" ]]; then
    echo "Este repositorio solo se ejecuta en Ubuntu Linux nativo o Ubuntu sobre WSL2." >&2
    return 1
  fi
  if [[ "${os_id}" != "ubuntu" ]]; then
    echo "Distribucion no soportada: ${os_id}. Se requiere Ubuntu nativo o Ubuntu sobre WSL2." >&2
    return 1
  fi
  if [[ "${architecture}" != "x86_64" ]]; then
    echo "Arquitectura no soportada: ${architecture:-desconocida}. SIM2REAL requiere Ubuntu x86_64 (Intel/AMD de 64 bits)." >&2
    return 1
  fi
  case "${os_version}" in
    22.04|24.04) ;;
    *)
      echo "Version de Ubuntu no soportada: ${os_version}. SIM2REAL admite Ubuntu 22.04 y 24.04." >&2
      return 1
      ;;
  esac
  if [[ "${host}" == "wsl" ]]; then
    wsl_version="$(sim2real_detect_wsl_version)"
    if [[ "${wsl_version}" != "2" ]]; then
      cat >&2 <<'EOF'
WSL1 no es compatible con esta pila MJX/JAX. Convierte la distribucion desde
PowerShell con `wsl --set-version <Distro> 2` y vuelve a ejecutar el instalador.
EOF
      return 1
    fi
  fi
}

sim2real_require_workspace() {
  local repo_root host resolved filesystem_type
  repo_root="${1:-$(sim2real_repo_root)}"
  host="$(sim2real_detect_host)"
  resolved="$(realpath "${repo_root}")"
  if [[ "${host}" != "wsl" ]]; then
    return 0
  fi
  filesystem_type="$(sim2real_workspace_filesystem_type "${resolved}")"
  if sim2real_is_windows_filesystem_type "${filesystem_type}" \
      || [[ -z "${filesystem_type}" && "${resolved}" == /mnt/* ]]; then
    cat >&2 <<EOF
El repositorio esta en un disco Windows montado en WSL: ${resolved}
Filesystem detectado: ${filesystem_type:-desconocido}
Esta version necesita el filesystem Linux de WSL (por ejemplo ~/robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL)
para evitar penalizaciones de E/S y problemas de permisos.
EOF
    return 1
  fi
}

sim2real_workspace_filesystem_type() {
  local target="$1" filesystem_type
  command -v findmnt >/dev/null 2>&1 || return 0
  filesystem_type="$(findmnt --target "${target}" --noheadings --output FSTYPE 2>/dev/null | head -n 1 || true)"
  filesystem_type="${filesystem_type//[[:space:]]/}"
  printf '%s\n' "${filesystem_type,,}"
}

sim2real_is_windows_filesystem_type() {
  case "${1,,}" in
    drvfs|9p|v9fs|virtiofs) return 0 ;;
    *) return 1 ;;
  esac
}

sim2real_validate_accelerator() {
  case "${1:-}" in
    auto|nvidia|amd|intel|cpu) return 0 ;;
    *)
      echo "Acelerador no reconocido: ${1:-<vacio>}. Usa auto, nvidia, amd, intel o cpu." >&2
      return 1
      ;;
  esac
}

sim2real_validate_accelerator_request() {
  case "${1:-}" in
    compatible) return 0 ;;
    auto|nvidia|amd|intel|cpu) return 0 ;;
    *)
      echo "Acelerador no reconocido: ${1:-<vacio>}. Usa compatible, auto, nvidia, amd, intel o cpu." >&2
      return 1
      ;;
  esac
}

sim2real_environment_slug() {
  case "${1:-}" in
    nvidia) printf '%s\n' "nvidia-cuda12" ;;
    amd) printf '%s\n' "amd-rocm70" ;;
    cpu) printf '%s\n' "cpu" ;;
    intel) printf '%s\n' "intel-oneapi" ;;
    *)
      echo "No existe un entorno para el acelerador: ${1:-<vacio>}." >&2
      return 1
      ;;
  esac
}

sim2real_environments_root() {
  local repo_root="${1:-$(sim2real_repo_root)}"
  printf '%s\n' "${repo_root}/.venvs"
}

sim2real_environment_dir() {
  local accelerator="$1" repo_root="${2:-$(sim2real_repo_root)}" slug
  slug="$(sim2real_environment_slug "${accelerator}")" || return 1
  printf '%s/%s\n' "$(sim2real_environments_root "${repo_root}")" "${slug}"
}

sim2real_python_executable() {
  local accelerator="$1" repo_root="${2:-$(sim2real_repo_root)}"
  printf '%s/bin/python\n' "$(sim2real_environment_dir "${accelerator}" "${repo_root}")"
}

sim2real_runtime_lock_file() {
  local repo_root="${1:-$(sim2real_repo_root)}" state_dir lock_file
  state_dir="${repo_root}/.sim2real"
  lock_file="${state_dir}/runtime.lock"
  [[ ! -L "${state_dir}" && ! -L "${lock_file}" ]] || {
    echo "El bloqueo local de SIM2REAL no puede ser un enlace simbolico." >&2
    return 1
  }
  mkdir -p -- "${state_dir}"
  [[ -d "${state_dir}" && ! -L "${state_dir}" ]] || return 1
  if [[ -e "${lock_file}" || -L "${lock_file}" ]]; then
    [[ -f "${lock_file}" && ! -L "${lock_file}" ]] || {
      echo "El bloqueo local no es un archivo regular seguro: ${lock_file}" >&2
      return 1
    }
  else
    (
      umask 077
      set -o noclobber
      : > "${lock_file}"
    ) 2>/dev/null || true
  fi
  [[ -f "${lock_file}" && ! -L "${lock_file}" ]] || return 1
  chmod 600 -- "${lock_file}"
  printf '%s\n' "${lock_file}"
}

sim2real_acquire_runtime_lock() {
  local mode="$1" repo_root="${2:-$(sim2real_repo_root)}" wait_mode="${3:-wait}"
  local lock_file expected_identity opened_identity current_identity
  case "${mode}" in
    shared|exclusive) ;;
    *) echo "Modo de bloqueo SIM2REAL no valido: ${mode:-<vacio>}." >&2; return 1 ;;
  esac
  case "${wait_mode}" in
    wait|nonblocking) ;;
    *) echo "Espera de bloqueo SIM2REAL no valida: ${wait_mode:-<vacia>}." >&2; return 1 ;;
  esac
  command -v flock >/dev/null 2>&1 || {
    echo "Falta flock (paquete util-linux); no se puede proteger el entorno activo." >&2
    return 1
  }
  lock_file="$(sim2real_runtime_lock_file "${repo_root}")" || return 1
  if [[ -n "${SIM2REAL_RUNTIME_LOCK_FD:-}" ]]; then
    [[ "${SIM2REAL_RUNTIME_LOCK_FD}" =~ ^[0-9]+$ &&
       -e "/proc/$$/fd/${SIM2REAL_RUNTIME_LOCK_FD}" ]] || {
      echo "El descriptor heredado del bloqueo SIM2REAL no es valido." >&2
      return 1
    }
    expected_identity="$(stat -Lc '%d:%i' -- "${lock_file}")" || return 1
    opened_identity="$(stat -Lc '%d:%i' -- "/proc/$$/fd/${SIM2REAL_RUNTIME_LOCK_FD}")" || return 1
    [[ "${opened_identity}" == "${expected_identity}" ]] || {
      echo "El descriptor heredado no pertenece a ${lock_file}." >&2
      return 1
    }
    [[ "${SIM2REAL_RUNTIME_LOCK_MODE:-}" == "${mode}" ||
       "${SIM2REAL_RUNTIME_LOCK_MODE:-}" == "exclusive" ]] || {
      echo "No se puede elevar un bloqueo SIM2REAL compartido a exclusivo." >&2
      return 1
    }
    if [[ "${SIM2REAL_RUNTIME_LOCK_MODE}" == "exclusive" ]]; then
      flock --exclusive --nonblock "${SIM2REAL_RUNTIME_LOCK_FD}" || return 1
    else
      flock --shared --nonblock "${SIM2REAL_RUNTIME_LOCK_FD}" || return 1
    fi
    return 0
  fi
  expected_identity="$(stat -Lc '%d:%i' -- "${lock_file}")" || return 1
  exec {SIM2REAL_RUNTIME_LOCK_FD}<>"${lock_file}"
  opened_identity="$(stat -Lc '%d:%i' -- "/proc/$$/fd/${SIM2REAL_RUNTIME_LOCK_FD}")" || {
    exec {SIM2REAL_RUNTIME_LOCK_FD}>&-
    unset SIM2REAL_RUNTIME_LOCK_FD SIM2REAL_RUNTIME_LOCK_MODE
    return 1
  }
  current_identity="$(stat -Lc '%d:%i' -- "${lock_file}")" || {
    exec {SIM2REAL_RUNTIME_LOCK_FD}>&-
    unset SIM2REAL_RUNTIME_LOCK_FD SIM2REAL_RUNTIME_LOCK_MODE
    return 1
  }
  if [[ "${opened_identity}" != "${expected_identity}" ||
        "${current_identity}" != "${expected_identity}" ]]; then
    exec {SIM2REAL_RUNTIME_LOCK_FD}>&-
    unset SIM2REAL_RUNTIME_LOCK_FD SIM2REAL_RUNTIME_LOCK_MODE
    echo "El archivo de bloqueo cambio mientras se abria; se aborta por seguridad." >&2
    return 1
  fi
  if [[ "${mode}" == "exclusive" ]]; then
    if [[ "${wait_mode}" == "nonblocking" ]]; then
      flock --exclusive --nonblock "${SIM2REAL_RUNTIME_LOCK_FD}" || {
        exec {SIM2REAL_RUNTIME_LOCK_FD}>&-
        unset SIM2REAL_RUNTIME_LOCK_FD SIM2REAL_RUNTIME_LOCK_MODE
        return 1
      }
    else
      flock --exclusive "${SIM2REAL_RUNTIME_LOCK_FD}"
    fi
  elif [[ "${wait_mode}" == "nonblocking" ]]; then
    flock --shared --nonblock "${SIM2REAL_RUNTIME_LOCK_FD}" || {
      exec {SIM2REAL_RUNTIME_LOCK_FD}>&-
      unset SIM2REAL_RUNTIME_LOCK_FD SIM2REAL_RUNTIME_LOCK_MODE
      return 1
    }
  else
    flock --shared "${SIM2REAL_RUNTIME_LOCK_FD}"
  fi
  SIM2REAL_RUNTIME_LOCK_MODE="${mode}"
  export SIM2REAL_RUNTIME_LOCK_FD SIM2REAL_RUNTIME_LOCK_MODE
}

sim2real_run_with_runtime_lock() {
  local repo_root="$1" lock_file
  shift
  command -v flock >/dev/null 2>&1 || {
    echo "Falta flock (paquete util-linux); no se puede proteger el entorno activo." >&2
    return 1
  }
  lock_file="$(sim2real_runtime_lock_file "${repo_root}")" || return 1
  flock --shared "${lock_file}" "$@"
}

sim2real_validate_run_name() {
  local run_name="${1:-}"
  [[ "${run_name}" != "." && "${run_name}" != ".." && "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]
}

sim2real_logs_root() {
  local repo_root="${1:-$(sim2real_repo_root)}" resolved_repo logs_dir resolved_logs
  resolved_repo="$(realpath "${repo_root}")"
  logs_dir="${repo_root}/logs_sim2real_mjx"
  [[ ! -L "${logs_dir}" ]] || {
    echo "La carpeta de logs no puede ser un enlace simbolico: ${logs_dir}" >&2
    return 1
  }
  mkdir -p "${logs_dir}"
  resolved_logs="$(realpath "${logs_dir}")"
  [[ "${resolved_logs}" == "${resolved_repo}/logs_sim2real_mjx" ]] || {
    echo "La carpeta de logs escapa del repositorio: ${resolved_logs}" >&2
    return 1
  }
  printf '%s\n' "${resolved_logs}"
}

sim2real_safe_run_dir() {
  local repo_root="$1" run_name="$2" logs_root candidate resolved
  sim2real_validate_run_name "${run_name}" || {
    echo "Nombre de run no valido: ${run_name:-<vacio>}. No se admiten rutas." >&2
    return 1
  }
  logs_root="$(sim2real_logs_root "${repo_root}")" || return 1
  candidate="${logs_root}/${run_name}"
  [[ ! -L "${candidate}" ]] || {
    echo "La run no puede ser un enlace simbolico: ${candidate}" >&2
    return 1
  }
  resolved="$(realpath -m "${candidate}")"
  [[ "$(dirname "${resolved}")" == "${logs_root}" ]] || {
    echo "La run escapa de ${logs_root}: ${resolved}" >&2
    return 1
  }
  printf '%s\n' "${resolved}"
}

sim2real_safe_existing_run() {
  local repo_root="$1" candidate="$2" logs_root resolved
  logs_root="$(sim2real_logs_root "${repo_root}")" || return 1
  [[ -d "${candidate}" && ! -L "${candidate}" ]] || return 1
  resolved="$(realpath "${candidate}")" || return 1
  [[ "$(dirname "${resolved}")" == "${logs_root}" ]] || return 1
  sim2real_validate_run_name "$(basename "${resolved}")" || return 1
  printf '%s\n' "${resolved}"
}

sim2real_pid_alive() {
  local pid="${1:-}"
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" >/dev/null 2>&1
}

sim2real_read_safe_pid_file() {
  local pid_file="$1" pid
  [[ -f "${pid_file}" && ! -L "${pid_file}" ]] || return 1
  IFS= read -r pid < "${pid_file}" || return 1
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s\n' "${pid}"
}

sim2real_process_argv() {
  local pid="$1"
  sim2real_pid_alive "${pid}" || return 1
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  tr '\0' '\n' < "/proc/${pid}/cmdline"
}

sim2real_training_pid_matches_run() {
  local repo_root="$1" run_dir="$2" pid="$3"
  local safe_run logs_root pid_file stored_pid expected_script expected_python expected_name profile
  local has_logdir=0 has_run_name=0 has_expected_python=0 index
  local -a argv=()

  sim2real_pid_alive "${pid}" || return 1
  safe_run="$(sim2real_safe_existing_run "${repo_root}" "${run_dir}")" || return 1
  logs_root="$(sim2real_logs_root "${repo_root}")" || return 1
  pid_file="${safe_run}/entrenamiento.pid"
  [[ -f "${pid_file}" && ! -L "${pid_file}" ]] || return 1
  IFS= read -r stored_pid < "${pid_file}" || return 1
  [[ "${stored_pid}" == "${pid}" ]] || return 1
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  mapfile -d '' -t argv < "/proc/${pid}/cmdline" || true

  expected_script="${repo_root}/sim2real_mjx/entrenar_ppo_mjx.py"
  expected_name="$(basename "${safe_run}")"
  [[ "${#argv[@]}" -ge 2 ]] || return 1
  for profile in nvidia amd cpu; do
    expected_python="$(sim2real_python_executable "${profile}" "${repo_root}")" || return 1
    [[ "${argv[0]}" == "${expected_python}" ]] && has_expected_python=1
  done
  # Compatibilidad de migración: reconoce un proceso iniciado por v1.1.0 para
  # poder detenerlo, pero ningún comando nuevo selecciona esta `.venv`.
  [[ "${argv[0]}" == "${repo_root}/.venv/bin/python" ]] && has_expected_python=1
  [[ "${has_expected_python}" == "1" ]] || return 1
  [[ "${argv[1]}" == "${expected_script}" ]] || return 1

  for (( index = 2; index < ${#argv[@]}; index++ )); do
    if [[ "${argv[index]}" == "--logdir" && $(( index + 1 )) -lt ${#argv[@]} && "${argv[index + 1]}" == "${logs_root}" ]]; then
      has_logdir=1
    elif [[ "${argv[index]}" == "--run_name" && $(( index + 1 )) -lt ${#argv[@]} && "${argv[index + 1]}" == "${expected_name}" ]]; then
      has_run_name=1
    fi
  done
  [[ "${has_logdir}" == "1" && "${has_run_name}" == "1" ]]
}

sim2real_launcher_pid_matches_run() {
  local repo_root="$1" run_dir="$2" pid="$3"
  local safe_run logs_root pid_file stored_pid expected_script expected_python
  local expected_name argument profile
  local has_flock=0 has_script=0 has_python=0 has_logdir=0 has_run_name=0
  local previous=""
  local -a argv=()

  sim2real_pid_alive "${pid}" || return 1
  safe_run="$(sim2real_safe_existing_run "${repo_root}" "${run_dir}")" || return 1
  logs_root="$(sim2real_logs_root "${repo_root}")" || return 1
  pid_file="${safe_run}/lanzador.pid"
  stored_pid="$(sim2real_read_safe_pid_file "${pid_file}")" || return 1
  [[ "${stored_pid}" == "${pid}" ]] || return 1
  mapfile -t argv < <(sim2real_process_argv "${pid}") || return 1

  expected_script="${repo_root}/sim2real_mjx/entrenar_ppo_mjx.py"
  expected_name="$(basename "${safe_run}")"
  for argument in "${argv[@]}"; do
    [[ "$(basename -- "${argument}")" == "flock" ]] && has_flock=1
    [[ "${argument}" == "${expected_script}" ]] && has_script=1
    for profile in nvidia amd cpu; do
      expected_python="$(sim2real_python_executable "${profile}" "${repo_root}")" || return 1
      [[ "${argument}" == "${expected_python}" ]] && has_python=1
    done
    [[ "${argument}" == "${repo_root}/.venv/bin/python" ]] && has_python=1
    if [[ "${previous}" == "--logdir" && "${argument}" == "${logs_root}" ]]; then
      has_logdir=1
    elif [[ "${previous}" == "--run_name" && "${argument}" == "${expected_name}" ]]; then
      has_run_name=1
    fi
    previous="${argument}"
  done
  [[ "${has_flock}" == "1" && "${has_script}" == "1" &&
     "${has_python}" == "1" && "${has_logdir}" == "1" &&
     "${has_run_name}" == "1" ]]
}

sim2real_curriculum_pid_matches_run() {
  local repo_root="$1" run_dir="$2" pid="$3"
  local safe_run pid_file stored_pid expected_script argument process_cwd resolved_argument
  local has_script=0
  local -a argv=()

  sim2real_pid_alive "${pid}" || return 1
  safe_run="$(sim2real_safe_existing_run "${repo_root}" "${run_dir}")" || return 1
  pid_file="${safe_run}/curriculo_automatico.pid"
  stored_pid="$(sim2real_read_safe_pid_file "${pid_file}")" || return 1
  [[ "${stored_pid}" == "${pid}" ]] || return 1
  mapfile -t argv < <(sim2real_process_argv "${pid}") || return 1
  expected_script="${repo_root}/scripts/curriculo_automatico_sim2real.sh"
  process_cwd="$(readlink -f -- "/proc/${pid}/cwd" 2>/dev/null || true)"
  for argument in "${argv[@]}"; do
    if [[ "${argument}" == "${expected_script}" ]]; then
      has_script=1
    elif [[ -n "${process_cwd}" && "${argument}" == */* ]]; then
      if [[ "${argument}" == /* ]]; then
        resolved_argument="$(realpath -e -- "${argument}" 2>/dev/null || true)"
      else
        resolved_argument="$(realpath -e -- "${process_cwd}/${argument}" 2>/dev/null || true)"
      fi
      [[ "${resolved_argument}" == "${expected_script}" ]] && has_script=1
    fi
  done
  [[ "${has_script}" == "1" ]]
}

sim2real_active_training() {
  local repo_root="${1:-$(sim2real_repo_root)}" logs_root run_dir pid
  logs_root="$(sim2real_logs_root "${repo_root}")" || return 1
  for run_dir in "${logs_root}"/*; do
    [[ -d "${run_dir}" && ! -L "${run_dir}" && -f "${run_dir}/entrenamiento.pid" ]] || continue
    IFS= read -r pid < "${run_dir}/entrenamiento.pid" || continue
    if sim2real_training_pid_matches_run "${repo_root}" "${run_dir}" "${pid}"; then
      printf '%s\t%s\n' "${run_dir}" "${pid}"
      return 0
    fi
  done
  return 1
}

sim2real_thermal_guard_pid_matches_run() {
  local repo_root="$1" run_dir="$2" guard_pid="$3" train_pid="$4"
  local safe_run pid_file stored_pid expected_script
  local has_script=0 has_train_pid=0 has_run_dir=0 index
  local -a argv=()

  sim2real_pid_alive "${guard_pid}" || return 1
  [[ "${train_pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  safe_run="$(sim2real_safe_existing_run "${repo_root}" "${run_dir}")" || return 1
  pid_file="${safe_run}/proteccion_termica.pid"
  [[ -f "${pid_file}" && ! -L "${pid_file}" ]] || return 1
  IFS= read -r stored_pid < "${pid_file}" || return 1
  [[ "${stored_pid}" == "${guard_pid}" ]] || return 1
  [[ -r "/proc/${guard_pid}/cmdline" ]] || return 1
  mapfile -d '' -t argv < "/proc/${guard_pid}/cmdline" || true

  expected_script="${repo_root}/scripts/proteccion_termica.sh"
  for (( index = 0; index < ${#argv[@]}; index++ )); do
    [[ "${argv[index]}" == "${expected_script}" ]] && has_script=1
    if [[ "${argv[index]}" == "--pid" && $(( index + 1 )) -lt ${#argv[@]} && "${argv[index + 1]}" == "${train_pid}" ]]; then
      has_train_pid=1
    elif [[ "${argv[index]}" == "--run-dir" && $(( index + 1 )) -lt ${#argv[@]} && "${argv[index + 1]}" == "${safe_run}" ]]; then
      has_run_dir=1
    fi
  done
  [[ "${has_script}" == "1" && "${has_train_pid}" == "1" && "${has_run_dir}" == "1" ]]
}

sim2real_signal_training_process() {
  local repo_root="$1" run_dir="$2" pid="$3" signal_name="$4"
  [[ "${signal_name}" == "TERM" || "${signal_name}" == "KILL" ]] || return 1
  if ! sim2real_training_pid_matches_run "${repo_root}" "${run_dir}" "${pid}"; then
    echo "Se rechaza SIG${signal_name}: PID ${pid:-<vacio>} no corresponde al entrenamiento de esta run." >&2
    return 1
  fi
  kill -s "${signal_name}" "${pid}"
}

sim2real_signal_launcher_process() {
  local repo_root="$1" run_dir="$2" pid="$3" signal_name="$4"
  [[ "${signal_name}" == "TERM" || "${signal_name}" == "KILL" ]] || return 1
  if ! sim2real_launcher_pid_matches_run "${repo_root}" "${run_dir}" "${pid}"; then
    echo "Se rechaza SIG${signal_name}: PID ${pid:-<vacio>} no corresponde al lanzador de esta run." >&2
    return 1
  fi
  kill -s "${signal_name}" "${pid}"
}

sim2real_signal_curriculum_process() {
  local repo_root="$1" run_dir="$2" pid="$3" signal_name="$4"
  [[ "${signal_name}" == "TERM" || "${signal_name}" == "KILL" ]] || return 1
  if ! sim2real_curriculum_pid_matches_run "${repo_root}" "${run_dir}" "${pid}"; then
    echo "Se rechaza SIG${signal_name}: PID ${pid:-<vacio>} no corresponde al supervisor curricular de esta run." >&2
    return 1
  fi
  kill -s "${signal_name}" "${pid}"
}

sim2real_signal_thermal_guard() {
  local repo_root="$1" run_dir="$2" guard_pid="$3" train_pid="$4" signal_name="$5"
  [[ "${signal_name}" == "TERM" || "${signal_name}" == "KILL" ]] || return 1
  if ! sim2real_thermal_guard_pid_matches_run "${repo_root}" "${run_dir}" "${guard_pid}" "${train_pid}"; then
    echo "Se rechaza SIG${signal_name}: PID ${guard_pid:-<vacio>} no corresponde al guard termico de esta run." >&2
    return 1
  fi
  kill -s "${signal_name}" "${guard_pid}"
}

sim2real_escribir_ultima_ejecucion() {
  local repo_root="$1" run_dir="$2" logs_root safe_run pointer temp_file
  logs_root="$(sim2real_logs_root "${repo_root}")" || return 1
  safe_run="$(sim2real_safe_existing_run "${repo_root}" "${run_dir}")" || {
    echo "No se guarda un puntero a una run insegura: ${run_dir}" >&2
    return 1
  }
  pointer="${logs_root}/ultima_ejecucion.txt"
  [[ ! -L "${pointer}" ]] || {
    echo "El puntero ultima_ejecucion.txt no puede ser un enlace simbolico." >&2
    return 1
  }
  temp_file="$(mktemp "${logs_root}/.ultima_run.XXXXXX")"
  printf '%s\n' "${safe_run}" > "${temp_file}"
  mv -f -- "${temp_file}" "${pointer}"
}

sim2real_lspci() {
  if command -v lspci >/dev/null 2>&1; then
    lspci 2>/dev/null || true
  fi
}

sim2real_nvidia_smi_command() {
  if [[ "$(sim2real_detect_host)" == "wsl" && -x /usr/lib/wsl/lib/nvidia-smi ]]; then
    printf '%s\n' "/usr/lib/wsl/lib/nvidia-smi"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    command -v nvidia-smi
  else
    return 1
  fi
}

sim2real_has_pci_display_vendor() {
  local expected_vendor="${1,,}" device vendor class
  for device in /sys/bus/pci/devices/*; do
    [[ -r "${device}/vendor" && -r "${device}/class" ]] || continue
    vendor="$(<"${device}/vendor")"
    class="$(<"${device}/class")"
    if [[ "${vendor,,}" == "${expected_vendor}" && "${class,,}" == 0x03* ]]; then
      return 0
    fi
  done
  return 1
}

sim2real_has_nvidia_hardware() {
  local nvidia_smi
  if nvidia_smi="$(sim2real_nvidia_smi_command 2>/dev/null)" &&
      "${nvidia_smi}" -L >/dev/null 2>&1; then
    return 0
  fi
  sim2real_has_pci_display_vendor "0x10de" && return 0
  sim2real_lspci |
    grep -iE 'NVIDIA.*(VGA|3D|Display)|(VGA|3D|Display).*NVIDIA' >/dev/null
}

sim2real_has_amd_hardware() {
  [[ -e /dev/kfd ]] && return 0
  sim2real_has_pci_display_vendor "0x1002" && return 0
  sim2real_lspci |
    grep -iE '(AMD|ATI).*(VGA|3D|Display)|(VGA|3D|Display).*(AMD|ATI)' >/dev/null
}

sim2real_has_intel_hardware() {
  sim2real_has_pci_display_vendor "0x8086" && return 0
  sim2real_lspci |
    grep -iE 'Intel.*(VGA|3D|Display)|(VGA|3D|Display).*Intel' >/dev/null
}

sim2real_rocminfo_command() {
  if command -v rocminfo >/dev/null 2>&1; then
    command -v rocminfo
  elif [[ -x /opt/rocm/bin/rocminfo ]]; then
    printf '%s\n' "/opt/rocm/bin/rocminfo"
  else
    return 1
  fi
}

sim2real_rocminfo_works() {
  local rocminfo_command output
  rocminfo_command="$(sim2real_rocminfo_command)" || return 1
  output="$(timeout 20s "${rocminfo_command}" 2>/dev/null)" || return 1
  grep -qE 'Name:[[:space:]]+gfx[0-9]+' <<< "${output}"
}

sim2real_has_rocm_runtime() {
  [[ -e /dev/kfd && -r /dev/kfd && -w /dev/kfd ]] && sim2real_rocminfo_works
}

sim2real_nvidia_driver_version() {
  local nvidia_smi output version
  nvidia_smi="$(sim2real_nvidia_smi_command)" || return 1
  version="$(
    "${nvidia_smi}" --query-gpu=driver_version --format=csv,noheader 2>/dev/null |
      sed -n '1p' |
      tr -d '[:space:]' || true
  )"
  if [[ ! "${version}" =~ ^[0-9]+([.][0-9]+)*$ ]]; then
    output="$("${nvidia_smi}" 2>/dev/null || true)"
    version="$(
      printf '%s\n' "${output}" |
        sed -nE 's/.*Driver Version:[[:space:]]*([0-9]+([.][0-9]+)*).*/\1/p' |
        sed -n '1p'
    )"
  fi
  [[ "${version}" =~ ^[0-9]+([.][0-9]+)*$ ]] || return 1
  printf '%s\n' "${version}"
}

sim2real_nvidia_driver_version_supported() {
  local version="${1:-}" major
  [[ "${version}" =~ ^[0-9]+([.][0-9]+)*$ ]] || return 1
  major="${version%%.*}"
  (( 10#${major} >= 525 ))
}

sim2real_rocm_version() {
  local version_file output
  if [[ -n "${SIM2REAL_ROCM_VERSION:-}" ]]; then
    printf '%s\n' "${SIM2REAL_ROCM_VERSION}" | grep -oE '^[0-9]+\.[0-9]+\.[0-9]+$'
    return
  fi
  for version_file in /opt/rocm/.info/version /opt/rocm/.info/version-dev; do
    if [[ -r "${version_file}" ]]; then
      head -n 1 "${version_file}" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1
      return 0
    fi
  done
  if command -v hipconfig >/dev/null 2>&1; then
    output="$(hipconfig --version 2>/dev/null || true)"
    printf '%s\n' "${output}" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1
  fi
}

sim2real_rocm_version_supported() {
  case "${1:-}" in
    7.0.0|7.0.1|7.0.2) return 0 ;;
    *) return 1 ;;
  esac
}

sim2real_detect_accelerator() {
  # `auto` significa GPU real o error. CPU solo se elige de forma explícita.
  # Se cuentan únicamente backends GPU que ya están operativos y son compatibles
  # con esta versión; una iGPU no elegible no crea una falsa ambigüedad.
  local host nvidia_smi rocm_version ready_count=0 selected=""
  host="$(sim2real_detect_host)"
  if nvidia_smi="$(sim2real_nvidia_smi_command 2>/dev/null)" &&
      "${nvidia_smi}" -L >/dev/null 2>&1 &&
      sim2real_nvidia_driver_version_supported "$(sim2real_nvidia_driver_version 2>/dev/null || true)"; then
    selected="nvidia"
    ready_count=$((ready_count + 1))
  fi
  rocm_version="$(sim2real_rocm_version 2>/dev/null || true)"
  if [[ "${host}" != "wsl" && "$(sim2real_os_version_id)" == "24.04" ]] \
      && sim2real_has_amd_hardware \
      && sim2real_has_rocm_runtime \
      && sim2real_rocm_version_supported "${rocm_version}"; then
    selected="amd"
    ready_count=$((ready_count + 1))
  fi
  if (( ready_count == 1 )); then
    printf '%s\n' "${selected}"
    return 0
  fi
  if (( ready_count > 1 )); then
    echo "Deteccion GPU ambigua: NVIDIA/CUDA y AMD/ROCm estan operativos. Repite con --accelerator nvidia o --accelerator amd." >&2
    return 1
  fi

  if [[ "${host}" == "wsl" ]] && sim2real_has_amd_hardware; then
    echo "Se detecta AMD bajo WSL2, pero AMD no habilita ni valida JAX sobre ROCDXG. Usa Ubuntu nativo con ROCm compatible; no se usara CPU en silencio." >&2
  elif sim2real_has_nvidia_hardware; then
    echo "Se detecta NVIDIA, pero el driver no esta operativo o no cumple la version minima 525. Repara nvidia-smi y repite; no se usara CPU en silencio." >&2
  elif sim2real_has_amd_hardware; then
    echo "Se detecta AMD, pero ROCm 7.0.0-7.0.2, /dev/kfd, sus permisos o rocminfo no estan preparados. Completa el runtime AMD y repite; no se usara CPU en silencio." >&2
  elif sim2real_has_intel_hardware; then
    echo "Se detecta Intel GPU, pero el plugin Intel disponible no es compatible con la baseline JAX 0.6.2 de SIM2REAL. No se usara CPU en silencio." >&2
  else
    echo "No se ha encontrado ninguna GPU compatible y operativa. Usa --accelerator cpu solo para desarrollo o conecta una GPU admitida." >&2
  fi
  return 1
}

sim2real_detect_compatible_accelerator() {
  # Modo de instalación universal: conserva la GPU siempre que exista una ruta
  # admitida y operativa. Si no, elige CPU de forma visible, nunca fingiendo que
  # sigue existiendo aceleración GPU.
  local host nvidia_smi rocm_version reason
  host="$(sim2real_detect_host)"

  if nvidia_smi="$(sim2real_nvidia_smi_command 2>/dev/null)" &&
      "${nvidia_smi}" -L >/dev/null 2>&1 &&
      sim2real_nvidia_driver_version_supported \
        "$(sim2real_nvidia_driver_version 2>/dev/null || true)"; then
    rocm_version="$(sim2real_rocm_version 2>/dev/null || true)"
    if [[ "${host}" != "wsl" && "$(sim2real_os_version_id)" == "24.04" ]] &&
        sim2real_has_amd_hardware &&
        sim2real_has_rocm_runtime &&
        sim2real_rocm_version_supported "${rocm_version}"; then
      echo "Aviso: NVIDIA y AMD estan operativas; el modo compatible prioriza NVIDIA. Usa --accelerator amd si quieres forzar ROCm." >&2
    fi
    printf '%s\n' "nvidia"
    return 0
  fi

  rocm_version="$(sim2real_rocm_version 2>/dev/null || true)"
  if [[ "${host}" != "wsl" && "$(sim2real_os_version_id)" == "24.04" ]] &&
      sim2real_has_amd_hardware &&
      sim2real_has_rocm_runtime &&
      sim2real_rocm_version_supported "${rocm_version}"; then
    printf '%s\n' "amd"
    return 0
  fi

  if [[ "${host}" == "wsl" ]] && sim2real_has_amd_hardware; then
    reason="se detecta AMD bajo WSL2, donde esta version no admite JAX GPU"
  elif sim2real_has_nvidia_hardware; then
    reason="se detecta NVIDIA, pero nvidia-smi o el driver >= 525 no estan operativos"
  elif sim2real_has_amd_hardware; then
    reason="se detecta AMD, pero Ubuntu 24.04 y ROCm 7.0.0-7.0.2 no estan listos"
  elif sim2real_has_intel_hardware; then
    reason="se detecta Intel, cuya GPU no esta admitida por la baseline JAX actual"
  else
    reason="no se ha detectado una GPU compatible y operativa"
  fi

  cat >&2 <<EOF

AVISO: ${reason}.
El modo compatible selecciona CPU automaticamente. SIM2REAL podra usarse
para desarrollo y pruebas pequenas, pero no tendra aceleracion GPU ni sera
adecuado para entrenamiento PPO masivo.
EOF
  sim2real_print_accelerator_support_summary >&2
  printf '%s\n' "cpu"
}

sim2real_profile_file() {
  local repo_root="${1:-$(sim2real_repo_root)}"
  printf '%s\n' "${repo_root}/.sim2real/accelerator"
}

sim2real_saved_accelerator() {
  local profile_file saved
  profile_file="$(sim2real_profile_file "${1:-}")"
  if [[ -r "${profile_file}" ]]; then
    read -r saved < "${profile_file}" || true
    if sim2real_validate_accelerator "${saved}" >/dev/null 2>&1 && [[ "${saved}" != "auto" ]]; then
      printf '%s\n' "${saved}"
      return 0
    fi
  fi
  return 1
}

sim2real_resolve_accelerator() {
  local requested="${1:-auto}" repo_root="${2:-$(sim2real_repo_root)}" saved
  sim2real_validate_accelerator "${requested}" || return 1
  if [[ "${requested}" != "auto" ]]; then
    printf '%s\n' "${requested}"
    return 0
  fi
  if [[ -n "${SIM2REAL_ACCELERATOR:-}" && "${SIM2REAL_ACCELERATOR}" != "auto" ]]; then
    sim2real_validate_accelerator "${SIM2REAL_ACCELERATOR}" || return 1
    printf '%s\n' "${SIM2REAL_ACCELERATOR}"
    return 0
  fi
  if saved="$(sim2real_saved_accelerator "${repo_root}" 2>/dev/null)"; then
    printf '%s\n' "${saved}"
  else
    sim2real_detect_accelerator
  fi
}

sim2real_save_accelerator() {
  local accelerator="$1" repo_root="${2:-$(sim2real_repo_root)}" profile_file profile_dir temporary
  sim2real_validate_accelerator "${accelerator}" || return 1
  [[ "${accelerator}" != "auto" ]] || {
    echo "No se guarda el valor auto; primero debe resolverse." >&2
    return 1
  }
  profile_file="$(sim2real_profile_file "${repo_root}")"
  profile_dir="$(dirname "${profile_file}")"
  [[ ! -L "${profile_dir}" && ! -L "${profile_file}" ]] || {
    echo "El estado local del acelerador no puede ser un enlace simbolico." >&2
    return 1
  }
  mkdir -p "${profile_dir}"
  temporary="$(mktemp "${profile_dir}/.accelerator.XXXXXX")"
  printf '%s\n' "${accelerator}" > "${temporary}"
  mv -f -- "${temporary}" "${profile_file}"
}

sim2real_support_level() {
  local accelerator="$1" host="${2:-$(sim2real_detect_host)}"
  case "${accelerator}" in
    nvidia)
      if [[ "${host}" == "wsl" ]]; then
        printf '%s\n' "validado en el proyecto con Ubuntu 24.04 sobre WSL2 (WSL figura como experimental en JAX)"
      else
        printf '%s\n' "implementado, sin validacion fisica en este proyecto"
      fi
      ;;
    cpu) printf '%s\n' "validado para instalacion y pruebas pequenas; entrenamiento muy lento" ;;
    amd)
      if [[ "${host}" == "wsl" ]]; then
        printf '%s\n' "bloqueado: AMD no habilita ni valida JAX sobre WSL"
      else
        printf '%s\n' "experimental e implementado, sin validacion fisica en este proyecto (requiere ROCm 7.0.0-7.0.2)"
      fi
      ;;
    intel) printf '%s\n' "no-compatible con baseline JAX 0.6.2" ;;
    *) printf '%s\n' "desconocido" ;;
  esac
}

sim2real_print_accelerator_support_summary() {
  cat <<'EOF'
Compatibilidad GPU de esta version:

MODO UNIVERSAL RECOMENDADO:
  --accelerator compatible detecta la mejor ruta disponible. En Ubuntu nativo,
  el bootstrap tambien puede preparar el driver de una NVIDIA visible por PCI.
  Fuera de ese caso, si no hay una GPU admitida y operativa, selecciona CPU
  automaticamente con un aviso visible. No presenta CPU como aceleracion GPU.

VALIDADO DE PRINCIPIO A FIN EN EL PROYECTO:
  NVIDIA sobre WSL2 con Ubuntu 24.04: usa --accelerator nvidia.
  CPU sobre Ubuntu 22.04/24.04 nativo o WSL2: validado para instalacion,
    desarrollo y pruebas pequenas, no para entrenamiento masivo.

IMPLEMENTADO, PERO SIN VALIDACION FISICA EN ESTE PROYECTO:
  NVIDIA sobre WSL2 con Ubuntu 22.04 o sobre Ubuntu 22.04/24.04 nativo:
    usa --accelerator nvidia.
  AMD sobre Ubuntu 24.04 nativo: ruta experimental; con hardware admitido y
    ROCm 7.0.0-7.0.2 ya operativo, usa --accelerator amd.

NO SOPORTADOS CON GPU:
  Usuarios AMD en WSL2: usa --accelerator cpu; actualmente la GPU AMD
    no esta admitida.
  Usuarios Intel: usa --accelerator cpu; actualmente la GPU Intel
    no esta admitida.

El perfil cpu no utiliza la GPU ni sustituye silenciosamente un perfil GPU
solicitado.
EOF
}

sim2real_check_profile_host() {
  local accelerator="$1" host="${2:-$(sim2real_detect_host)}" repo_root="${3:-$(sim2real_repo_root)}"
  case "${accelerator}" in
    nvidia)
      local driver_version nvidia_smi virtualization
      nvidia_smi="$(sim2real_nvidia_smi_command 2>/dev/null || true)"
      if [[ -z "${nvidia_smi}" ]] || ! "${nvidia_smi}" -L >/dev/null 2>&1; then
        if [[ "${host}" == "wsl" ]]; then
          echo "Perfil NVIDIA: nvidia-smi no ve la GPU dentro de WSL. Instala/actualiza el driver NVIDIA en Windows, ejecuta 'wsl --shutdown' y no uses ubuntu-drivers dentro de WSL." >&2
        else
          virtualization="$(sim2real_detect_virtualization)"
          if [[ "${virtualization}" == "oracle" ]]; then
            echo "Perfil NVIDIA: Ubuntu esta dentro de VirtualBox y la VM no expone una GPU NVIDIA PCI. Usa CPU o un host Ubuntu/WSL2 donde la GPU sea visible." >&2
          elif [[ "${virtualization}" != "none" ]]; then
            echo "Perfil NVIDIA: la maquina virtual (${virtualization}) no expone una GPU NVIDIA/passthrough utilizable. El hipervisor debe proporcionarla antes de instalar SIM2REAL." >&2
          else
            echo "Perfil NVIDIA: nvidia-smi no ve la GPU. Instala un driver NVIDIA compatible (CUDA 12 requiere driver >= 525)." >&2
          fi
        fi
        return 1
      fi
      driver_version="$(sim2real_nvidia_driver_version || true)"
      if [[ -z "${driver_version}" ]]; then
        if [[ "${host}" == "wsl" ]]; then
          echo "Perfil NVIDIA: WSL ve la GPU, pero no puede determinar la version del driver NVIDIA de Windows. Actualizalo en Windows y ejecuta 'wsl --shutdown'; no uses ubuntu-drivers." >&2
        else
          echo "Perfil NVIDIA: no puedo determinar la version del driver mediante nvidia-smi." >&2
        fi
        return 1
      fi
      if ! sim2real_nvidia_driver_version_supported "${driver_version}"; then
        if [[ "${host}" == "wsl" ]]; then
          echo "Perfil NVIDIA: driver de Windows ${driver_version} demasiado antiguo. Actualizalo en Windows, ejecuta 'wsl --shutdown' y no uses ubuntu-drivers; JAX CUDA 12 requiere >= 525." >&2
        else
          echo "Perfil NVIDIA: driver ${driver_version} demasiado antiguo. JAX con CUDA 12 requiere driver >= 525." >&2
        fi
        return 1
      fi
      ;;
    amd)
      if [[ "${host}" == "wsl" ]]; then
        cat >&2 <<'EOF'
El perfil AMD GPU esta bloqueado bajo WSL2. ROCm puede exponer algunas GPU
mediante ROCDXG, pero AMD declara que JAX no esta habilitado ni validado en
esta ruta. Usa Ubuntu nativo con una GPU incluida en la matriz ROCm.
No se continuara con CPU ni se creara un entorno aparentemente valido.
EOF
        return 1
      fi
      if [[ "$(sim2real_os_version_id)" != "24.04" ]]; then
        echo "Perfil AMD: los wheels ROCm fijados usan Python 3.12 y esta ruta solo se admite en Ubuntu 24.04. Ubuntu 22.04 conserva NVIDIA y CPU, pero no este perfil AMD." >&2
        return 1
      fi
      if ! sim2real_has_amd_hardware; then
        echo "Perfil AMD: no se ha detectado una GPU AMD." >&2
        return 1
      fi
      if [[ ! -e /dev/kfd ]]; then
        echo "Perfil AMD: falta /dev/kfd. Instala y carga correctamente el driver/runtime ROCm del host." >&2
        return 1
      fi
      if [[ ! -r /dev/kfd || ! -w /dev/kfd ]]; then
        echo "Perfil AMD: el usuario no tiene permisos de lectura/escritura sobre /dev/kfd. Revisa los grupos render y video y vuelve a iniciar sesion." >&2
        return 1
      fi
      if ! sim2real_rocminfo_command >/dev/null 2>&1; then
        echo "Perfil AMD: no encuentro rocminfo en PATH ni en /opt/rocm/bin." >&2
        return 1
      fi
      if ! sim2real_rocminfo_works; then
        echo "Perfil AMD: rocminfo existe, pero no puede inicializar el runtime o enumerar la GPU." >&2
        return 1
      fi
      local rocm_version
      rocm_version="$(sim2real_rocm_version || true)"
      if [[ -z "${rocm_version}" ]]; then
        echo "Perfil AMD: no puedo determinar la version ROCm. Define SIM2REAL_ROCM_VERSION=7.0.x solo si la has verificado." >&2
        return 1
      fi
      if ! sim2real_rocm_version_supported "${rocm_version}"; then
        echo "Perfil AMD: ROCm ${rocm_version} no coincide con la matriz 7.0.0-7.0.2 validada para los wheels fijados." >&2
        return 1
      fi
      ;;
    intel)
      cat >&2 <<'EOF'
El perfil Intel GPU no se instala en esta version.
El plugin oficial Intel OpenXLA publicado exige una serie de JAX distinta de
la baseline validada (JAX/JAXLIB 0.6.2), ademas de hardware Data Center y oneAPI.
Usa --accelerator cpu en equipos Intel. No se fingira aceleracion GPU.
EOF
      return 1
      ;;
    cpu) return 0 ;;
  esac
}

sim2real_uv_extra_args() {
  case "$1" in
    nvidia) printf '%s\n' "--extra" "nvidia" ;;
    amd) return 0 ;;
    cpu) return 0 ;;
    intel) return 1 ;;
  esac
}

sim2real_configure_accelerator_env() {
  local accelerator="$1"
  export SIM2REAL_ACCELERATOR_RESOLVED="${accelerator}"
  if [[ "${accelerator}" == "nvidia" && "$(sim2real_detect_host)" == "wsl" &&
      -d /usr/lib/wsl/lib ]]; then
    case ":${PATH}:" in
      *":/usr/lib/wsl/lib:"*) ;;
      *) export PATH="/usr/lib/wsl/lib:${PATH}" ;;
    esac
  fi
  # Solo se fuerza el perfil CPU explícito. NVIDIA y AMD quedan en la detección
  # automática de JAX;
  # nunca ocultamos un acelerador de fabricante con JAX_PLATFORMS=cpu.
  if [[ "${accelerator}" == "cpu" ]]; then
    export JAX_PLATFORMS="cpu"
    export JAX_PLATFORM_NAME="cpu"
  else
    unset JAX_PLATFORMS || true
    unset JAX_PLATFORM_NAME || true
  fi
}

sim2real_expected_jax_backend() {
  case "$1" in
    nvidia|amd) printf '%s\n' "gpu" ;;
    intel) printf '%s\n' "xpu" ;;
    cpu) printf '%s\n' "cpu" ;;
  esac
}

sim2real_nvidia_metrics_csv() {
  local nvidia_smi
  nvidia_smi="$(sim2real_nvidia_smi_command)" || return 1
  "${nvidia_smi}" --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null | sed -n '1p'
}

sim2real_amd_metrics_csv() {
  local device vendor_file hwmon temp_file temp util used total
  for device in /sys/class/drm/card*/device; do
    [[ -r "${device}/vendor" ]] || continue
    vendor_file="$(<"${device}/vendor")"
    [[ "${vendor_file}" == "0x1002" ]] || continue
    temp=""
    for hwmon in "${device}"/hwmon/hwmon*; do
      temp_file="${hwmon}/temp1_input"
      if [[ -r "${temp_file}" ]]; then
        temp="$(( $(<"${temp_file}") / 1000 ))"
        break
      fi
    done
    util=""
    used=""
    total=""
    [[ -r "${device}/gpu_busy_percent" ]] && util="$(<"${device}/gpu_busy_percent")"
    [[ -r "${device}/mem_info_vram_used" ]] && used="$(<"${device}/mem_info_vram_used")"
    [[ -r "${device}/mem_info_vram_total" ]] && total="$(<"${device}/mem_info_vram_total")"
    [[ -n "${used}" ]] && used="$(( used / 1024 / 1024 ))"
    [[ -n "${total}" ]] && total="$(( total / 1024 / 1024 ))"
    [[ -n "${temp}" ]] || return 1
    printf '%s,%s,%s,%s\n' "${temp}" "${util:-}" "${used:-}" "${total:-}"
    return 0
  done
  return 1
}

sim2real_gpu_metrics_csv() {
  local accelerator="${1:-$(sim2real_resolve_accelerator auto)}"
  case "${accelerator}" in
    nvidia) sim2real_nvidia_metrics_csv ;;
    amd) sim2real_amd_metrics_csv ;;
    *) return 1 ;;
  esac
}

sim2real_gpu_summary() {
  local accelerator="${1:-$(sim2real_resolve_accelerator auto)}"
  local line temp util used total name nvidia_smi
  case "${accelerator}" in
    nvidia)
      nvidia_smi="$(sim2real_nvidia_smi_command 2>/dev/null || true)"
      if [[ -n "${nvidia_smi}" ]]; then
        name="$(
          "${nvidia_smi}" --query-gpu=name --format=csv,noheader 2>/dev/null |
            sed -n '1p' || true
        )"
      fi
      ;;
    amd) name="AMD/ROCm" ;;
    cpu) printf '%s\n' "CPU (sin acelerador GPU seleccionado)"; return 0 ;;
    intel) printf '%s\n' "Intel detectada, backend no compatible"; return 0 ;;
  esac
  if line="$(sim2real_gpu_metrics_csv "${accelerator}" 2>/dev/null)"; then
    IFS=',' read -r temp util used total <<< "${line}"
    printf '%s | util %s%% | VRAM %s/%s MiB | %s C\n' \
      "${name:-${accelerator}}" "${util:-n/a}" "${used:-n/a}" "${total:-n/a}" "${temp:-n/a}"
  else
    printf '%s\n' "${name:-${accelerator}} | metricas no disponibles"
  fi
}
