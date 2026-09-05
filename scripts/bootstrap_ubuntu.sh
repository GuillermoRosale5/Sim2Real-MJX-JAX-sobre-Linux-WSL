#!/usr/bin/env bash

# Bootstrap autocontenido para una instalacion nueva en Ubuntu nativo o WSL2.
# Se puede descargar desde raw.githubusercontent.com y ejecutar antes de clonar.

SIM2REAL_BOOTSTRAP_DEFAULT_REPO_URL="https://github.com/GuillermoRosale5/Sim2Real-MJX-JAX-sobre-Linux-WSL.git"
SIM2REAL_BOOTSTRAP_MARKER="sim2real-mjx-jax-linux-wsl-v1"
SIM2REAL_BOOTSTRAP_TEMP_CLONE=""

sim2real_bootstrap_print_accelerator_support_summary() {
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

sim2real_bootstrap_usage() {
  cat <<'EOF'
Uso:
  bash bootstrap_ubuntu.sh --accelerator compatible|auto|nvidia|amd|intel|cpu
                           [opciones]

Opciones:
  --accelerator PERFIL   Obligatorio. `compatible` hace la seleccion automatica;
                         `auto` exige GPU y falla si solo puede usar CPU.
  --install-path RUTA    Por defecto: ~/robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL
  --repo-url URL         Solo https://github.com/<owner>/Sim2Real-MJX-JAX-sobre-Linux-WSL[.git]
  -h, --help             Muestra esta ayuda.

Este bootstrap admite Ubuntu 22.04/24.04 x86_64 nativo o sobre WSL2.
No lo ejecutes con sudo. En WSL2 nunca instala un driver Linux: NVIDIA debe
estar expuesta por el driver de Windows. `auto` exige una GPU compatible y
operativa. `compatible` puede elegir CPU automaticamente, siempre con un aviso.
EOF
  sim2real_bootstrap_print_accelerator_support_summary
}

sim2real_bootstrap_step() {
  printf '\n==> %s\n' "$*"
}

sim2real_bootstrap_die() {
  local exit_code="$1"
  shift
  printf 'ERROR: %s\n' "$*" >&2
  exit "${exit_code}"
}

sim2real_bootstrap_require_option_value() {
  local option="$1" remaining="$2" value="${3:-}"
  if (( remaining < 2 )) || [[ -z "${value}" || "${value}" == -* ]]; then
    sim2real_bootstrap_die 2 "Falta el valor de ${option}."
  fi
}

sim2real_bootstrap_validate_repo_url() {
  local repo_url="${1:-}"
  [[ "${repo_url}" =~ ^https://github\.com/[A-Za-z0-9][A-Za-z0-9._-]*/Sim2Real-MJX-JAX-sobre-Linux-WSL(\.git)?$ ]]
}

sim2real_bootstrap_normalize_repo_url() {
  local repo_url="${1:-}"
  sim2real_bootstrap_validate_repo_url "${repo_url}" || return 1
  printf '%s\n' "${repo_url%.git}"
}

sim2real_bootstrap_is_wsl() {
  [[ -n "${WSL_INTEROP:-}" ]] ||
    grep -qiE '(microsoft|wsl)' /proc/sys/kernel/osrelease /proc/version 2>/dev/null
}

sim2real_bootstrap_detect_wsl_version() {
  local release
  sim2real_bootstrap_is_wsl || {
    printf '%s\n' "none"
    return 0
  }
  release="$(uname -r 2>/dev/null || true)"
  if [[ "${release,,}" == *microsoft-standard* || "${release,,}" == *wsl2* ]]; then
    printf '%s\n' "2"
  else
    printf '%s\n' "1"
  fi
}

sim2real_bootstrap_host_kind() {
  if sim2real_bootstrap_is_wsl; then
    printf '%s\n' "wsl"
  else
    printf '%s\n' "native"
  fi
}

sim2real_bootstrap_add_wsl_driver_path() {
  local wsl_driver_dir="/usr/lib/wsl/lib"
  [[ -d "${wsl_driver_dir}" ]] || return 0
  case ":${PATH}:" in
    *":${wsl_driver_dir}:"*) ;;
    *) export PATH="${wsl_driver_dir}:${PATH}" ;;
  esac
}

sim2real_bootstrap_os_id() {
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

sim2real_bootstrap_os_version() {
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

sim2real_bootstrap_validate_host() {
  local os_id os_version architecture wsl_version

  if (( EUID == 0 )); then
    sim2real_bootstrap_die 10 \
      "No ejecutes el bootstrap con sudo/root. Usa tu usuario normal; el script pedira sudo solo para APT y, en Ubuntu nativo, ubuntu-drivers."
  fi
  [[ "$(uname -s 2>/dev/null || true)" == "Linux" ]] ||
    sim2real_bootstrap_die 10 "Este bootstrap necesita Linux."
  os_id="$(sim2real_bootstrap_os_id)"
  os_version="$(sim2real_bootstrap_os_version)"
  architecture="$(uname -m 2>/dev/null || true)"
  [[ "${os_id}" == "ubuntu" ]] ||
    sim2real_bootstrap_die 10 \
      "Distribucion no soportada: ${os_id}. Se requiere Ubuntu nativo o Ubuntu sobre WSL2."
  case "${os_version}" in
    22.04|24.04) ;;
    *)
      sim2real_bootstrap_die 10 \
        "Version de Ubuntu no soportada: ${os_version}. Se requiere Ubuntu 22.04 o 24.04."
      ;;
  esac
  [[ "${architecture}" == "x86_64" ]] ||
    sim2real_bootstrap_die 10 \
      "Arquitectura no soportada: ${architecture:-desconocida}. Se requiere x86_64."

  if sim2real_bootstrap_is_wsl; then
    wsl_version="$(sim2real_bootstrap_detect_wsl_version)"
    [[ "${wsl_version}" == "2" ]] ||
      sim2real_bootstrap_die 10 \
        "WSL1 no es compatible con MJX/JAX GPU. Convierte esta distribucion a WSL2 desde PowerShell y repite el bootstrap."
  fi

  [[ -n "${HOME:-}" && -d "${HOME}" ]] ||
    sim2real_bootstrap_die 10 "HOME no apunta a un directorio valido."
  [[ ! -L "${HOME}" ]] ||
    sim2real_bootstrap_die 10 "HOME no puede ser un enlace simbolico para este bootstrap."
  command -v sudo >/dev/null 2>&1 ||
    sim2real_bootstrap_die 10 "No encuentro sudo. La cuenta debe poder administrar paquetes APT."
  command -v apt-get >/dev/null 2>&1 ||
    sim2real_bootstrap_die 10 "No encuentro apt-get en este Ubuntu."
}

sim2real_bootstrap_resolve_install_path() {
  local requested="${1:-}" home_real expanded lexical resolved relative cursor component
  local -a components=()

  home_real="$(realpath -e -- "${HOME}")" ||
    sim2real_bootstrap_die 10 "No se puede resolver HOME: ${HOME}."
  if [[ -z "${requested}" ]]; then
    expanded="${home_real}/robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL"
  else
    case "${requested}" in
      '~') expanded="${home_real}" ;;
      \~/*) expanded="${home_real}/${requested#\~/}" ;;
      /*) expanded="${requested}" ;;
      *)
        sim2real_bootstrap_die 2 \
          "--install-path debe ser una ruta absoluta o comenzar por ~/."
        ;;
    esac
  fi

  lexical="$(realpath -ms -- "${expanded}")" ||
    sim2real_bootstrap_die 10 "No se puede normalizar la ruta de instalacion."
  case "${lexical}" in
    "${home_real}"/*) ;;
    *)
      sim2real_bootstrap_die 10 \
        "La instalacion debe quedar dentro de HOME (${home_real}); ruta recibida: ${lexical}."
      ;;
  esac

  relative="${lexical#"${home_real}"/}"
  IFS='/' read -r -a components <<< "${relative}"
  cursor="${home_real}"
  for component in "${components[@]}"; do
    [[ -n "${component}" ]] || continue
    cursor="${cursor}/${component}"
    [[ ! -L "${cursor}" ]] ||
      sim2real_bootstrap_die 10 \
        "La ruta de instalacion contiene un enlace simbolico: ${cursor}."
  done

  resolved="$(realpath -m -- "${lexical}")" ||
    sim2real_bootstrap_die 10 "No se puede resolver la ruta de instalacion."
  [[ "${resolved}" == "${lexical}" ]] ||
    sim2real_bootstrap_die 10 \
      "La ruta de instalacion atraviesa un enlace simbolico: ${lexical}."
  printf '%s\n' "${lexical}"
}

sim2real_bootstrap_wsl_filesystem_type() {
  local target="$1" probe filesystem_type
  command -v findmnt >/dev/null 2>&1 || return 0
  probe="${target}"
  while [[ ! -e "${probe}" && "${probe}" != "/" ]]; do
    probe="$(dirname -- "${probe}")"
  done
  filesystem_type="$(
    findmnt --target "${probe}" --noheadings --output FSTYPE 2>/dev/null |
      sed -n '1p' || true
  )"
  filesystem_type="${filesystem_type//[[:space:]]/}"
  printf '%s\n' "${filesystem_type,,}"
}

sim2real_bootstrap_validate_wsl_install_path() {
  local install_path="$1" filesystem_type
  filesystem_type="$(sim2real_bootstrap_wsl_filesystem_type "${install_path}")"
  case "${filesystem_type}" in
    drvfs|9p|v9fs|virtiofs)
      sim2real_bootstrap_die 10 \
        "En WSL2 la instalacion debe vivir en el filesystem Linux, no en uno compartido con Windows (${filesystem_type}): ${install_path}"
      ;;
  esac
  if [[ -z "${filesystem_type}" && "${install_path}" == /mnt/* ]]; then
    sim2real_bootstrap_die 10 \
      "En WSL2 la instalacion no puede quedar bajo /mnt; usa una ruta Linux como ~/robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL."
  fi
}

sim2real_bootstrap_install_base_packages() {
  local host_kind="${1:-native}"
  local -a packages=(ca-certificates curl git pciutils util-linux)

  case "${host_kind}" in
    native|wsl) ;;
    *) sim2real_bootstrap_die 10 "Tipo de host interno no valido: ${host_kind}." ;;
  esac

  sim2real_bootstrap_step "Instalando herramientas base de Ubuntu (${host_kind})"
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"
}

sim2real_bootstrap_install_nvidia_management_packages() {
  [[ "$(sim2real_bootstrap_host_kind)" == "native" ]] ||
    sim2real_bootstrap_die 40 \
      "Proteccion WSL2: no se instalaran ubuntu-drivers-common ni mokutil. El driver NVIDIA debe instalarse en Windows."
  sim2real_bootstrap_step "Instalando la gestion de drivers NVIDIA de Ubuntu nativo"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ubuntu-drivers-common mokutil
}

sim2real_bootstrap_has_pci_display_vendor() {
  local expected_vendor="${1,,}" device vendor class numeric_vendor
  for device in /sys/bus/pci/devices/*; do
    [[ -r "${device}/vendor" && -r "${device}/class" ]] || continue
    vendor="$(<"${device}/vendor")"
    class="$(<"${device}/class")"
    if [[ "${vendor,,}" == "${expected_vendor}" && "${class,,}" == 0x03* ]]; then
      return 0
    fi
  done

  command -v lspci >/dev/null 2>&1 || return 1
  numeric_vendor="${expected_vendor#0x}"
  lspci -Dn 2>/dev/null |
    grep -iE "[[:space:]]03[[:xdigit:]]{2}:[[:space:]]${numeric_vendor}:" >/dev/null
}

sim2real_bootstrap_report_pci_graphics() {
  local devices
  command -v lspci >/dev/null 2>&1 || {
    echo "Controladores graficos PCI detectados: no se puede consultar lspci." >&2
    return 0
  }
  devices="$(
    LC_ALL=C lspci -Dnn 2>/dev/null |
      awk '/\[03[[:xdigit:]][[:xdigit:]]\]:/ || /\[10de:/ {print "  - " $0}'
  )"
  if [[ -n "${devices}" ]]; then
    printf 'Controladores graficos PCI detectados:\n%s\n' "${devices}" >&2
  else
    echo "Controladores graficos PCI detectados: ninguno de clase Display (03xx)." >&2
  fi
}

sim2real_bootstrap_rocminfo_command() {
  if command -v rocminfo >/dev/null 2>&1; then
    command -v rocminfo
  elif [[ -x /opt/rocm/bin/rocminfo ]]; then
    printf '%s\n' "/opt/rocm/bin/rocminfo"
  else
    return 1
  fi
}

sim2real_bootstrap_rocm_version() {
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

sim2real_bootstrap_amd_runtime_ready() {
  local rocminfo_command rocm_version output
  [[ "$(sim2real_bootstrap_os_version)" == "24.04" ]] || return 1
  [[ -e /dev/kfd && -r /dev/kfd && -w /dev/kfd ]] || return 1
  rocminfo_command="$(sim2real_bootstrap_rocminfo_command)" || return 1
  output="$(timeout 20s "${rocminfo_command}" 2>/dev/null)" || return 1
  grep -qE 'Name:[[:space:]]+gfx[0-9]+' <<< "${output}" || return 1
  rocm_version="$(sim2real_bootstrap_rocm_version 2>/dev/null || true)"
  case "${rocm_version}" in
    7.0.0|7.0.1|7.0.2) return 0 ;;
    *) return 1 ;;
  esac
}

sim2real_bootstrap_validate_amd_native() {
  [[ "$(sim2real_bootstrap_os_version)" == "24.04" ]] ||
    sim2real_bootstrap_die 40 \
      "El perfil AMD fijado usa Python 3.12 y solo se admite en Ubuntu 24.04. En Ubuntu 22.04 usa NVIDIA o CPU explicita."
  sim2real_bootstrap_has_pci_display_vendor "0x1002" || {
    sim2real_bootstrap_report_pci_graphics
    sim2real_bootstrap_die 40 \
      "Se solicito AMD, pero Ubuntu no expone una GPU AMD PCI de clase grafica. No se continuara silenciosamente con CPU."
  }
  sim2real_bootstrap_amd_runtime_ready ||
    sim2real_bootstrap_die 40 \
      "La GPU AMD aparece, pero ROCm no esta listo. Comprueba primero que el modelo, Ubuntu y kernel figuran en la matriz ROCm 7.0.2; instala ROCm 7.0.0-7.0.2, concede acceso a /dev/kfd, reinicia y verifica que rocminfo enumera un agente gfx. Por seguridad este bootstrap no modifica amdgpu/DKMS a ciegas."
}

sim2real_bootstrap_detect_virtualization() {
  local virtualization="" dmi_hint="" dmi_file

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

sim2real_bootstrap_virtualization_label() {
  case "${1:-none}" in
    oracle) printf '%s\n' "VirtualBox/Oracle" ;;
    microsoft) printf '%s\n' "Hyper-V/Microsoft" ;;
    none) printf '%s\n' "ninguna" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

sim2real_bootstrap_platform_description() {
  local host_kind="$1" virtualization
  if [[ "${host_kind}" == "wsl" ]]; then
    printf '%s\n' "Ubuntu sobre WSL2"
    return 0
  fi
  virtualization="$(sim2real_bootstrap_detect_virtualization)"
  if [[ "${virtualization}" == "none" ]]; then
    printf '%s\n' "Ubuntu nativo"
  else
    printf 'Ubuntu en VM (%s)\n' \
      "$(sim2real_bootstrap_virtualization_label "${virtualization}")"
  fi
}

sim2real_bootstrap_compatible_cpu() {
  local reason="$1"
  cat >&2 <<EOF

AVISO: no hay una ruta GPU de SIM2REAL utilizable en este momento.
Motivo: ${reason}
El modo compatible seleccionara CPU automaticamente. La instalacion podra
usarse para desarrollo y pruebas pequenas, pero no tendra aceleracion GPU ni
sera adecuada para entrenamiento PPO masivo.
EOF
  sim2real_bootstrap_print_accelerator_support_summary >&2
  printf '%s\n' "cpu"
}

sim2real_bootstrap_resolve_compatible_accelerator() {
  local host_kind="$1" virtualization

  # NVIDIA operativa es la ruta con mayor nivel de validacion del proyecto.
  if sim2real_bootstrap_nvidia_driver_ready; then
    if [[ "${host_kind}" == "native" &&
          "$(sim2real_bootstrap_os_version)" == "24.04" ]] &&
        sim2real_bootstrap_has_pci_display_vendor "0x1002" &&
        sim2real_bootstrap_amd_runtime_ready; then
      echo "Aviso: NVIDIA y AMD estan operativas; el modo compatible prioriza NVIDIA. Usa --accelerator amd si quieres forzar ROCm." >&2
    fi
    printf '%s\n' "nvidia"
    return 0
  fi

  if [[ "${host_kind}" == "native" &&
        "$(sim2real_bootstrap_os_version)" == "24.04" ]] &&
      sim2real_bootstrap_has_pci_display_vendor "0x1002" &&
      sim2real_bootstrap_amd_runtime_ready; then
    printf '%s\n' "amd"
    return 0
  fi

  # En Ubuntu nativo el bootstrap puede preparar el driver NVIDIA oficial si
  # la GPU existe por PCI, aunque nvidia-smi aun no funcione.
  if [[ "${host_kind}" == "native" ]] &&
      sim2real_bootstrap_has_pci_display_vendor "0x10de"; then
    printf '%s\n' "nvidia"
    return 0
  fi

  if [[ "${host_kind}" == "wsl" ]]; then
    sim2real_bootstrap_compatible_cpu \
      "WSL2 no expone una NVIDIA operativa; AMD e Intel no tienen una ruta GPU admitida por esta version."
  elif sim2real_bootstrap_has_pci_display_vendor "0x1002"; then
    sim2real_bootstrap_compatible_cpu \
      "se detecta AMD, pero esta ruta exige Ubuntu 24.04 y ROCm 7.0.0-7.0.2 ya operativo."
  elif sim2real_bootstrap_has_pci_display_vendor "0x8086"; then
    sim2real_bootstrap_compatible_cpu \
      "se detecta Intel, cuya GPU no esta admitida por la baseline JAX actual."
  else
    virtualization="$(sim2real_bootstrap_detect_virtualization)"
    if [[ "${virtualization}" != "none" ]]; then
      sim2real_bootstrap_compatible_cpu \
        "la maquina virtual (${virtualization}) no expone una GPU compatible y operativa."
    else
      sim2real_bootstrap_compatible_cpu \
        "no se ha detectado una GPU compatible y operativa."
    fi
  fi
}

sim2real_bootstrap_resolve_accelerator() {
  local requested="$1" host_kind="${2:-native}" virtualization
  case "${host_kind}" in
    native|wsl) ;;
    *) sim2real_bootstrap_die 10 "Tipo de host interno no valido: ${host_kind}." ;;
  esac
  case "${requested}" in
    compatible)
      sim2real_bootstrap_resolve_compatible_accelerator "${host_kind}"
      ;;
    nvidia)
      if [[ "${host_kind}" == "native" ]] &&
          ! sim2real_bootstrap_has_pci_display_vendor "0x10de"; then
        sim2real_bootstrap_report_pci_graphics
        virtualization="$(sim2real_bootstrap_detect_virtualization)"
        if [[ "${virtualization}" == "oracle" ]]; then
          sim2real_bootstrap_die 40 \
            "Ubuntu se ejecuta dentro de VirtualBox y la maquina virtual no expone una GPU NVIDIA PCI. El bootstrap no puede crear ese acceso ni instalar un driver que lo sustituya. Usa --accelerator cpu o ejecuta SIM2REAL en Ubuntu nativo/WSL2 con la GPU visible."
        elif [[ "${virtualization}" != "none" ]]; then
          sim2real_bootstrap_die 40 \
            "Ubuntu se ejecuta en una maquina virtual (${virtualization}) sin una GPU NVIDIA PCI expuesta. El hipervisor debe exponer previamente una GPU mediante una tecnologia compatible; este bootstrap no puede hacerlo. Usa --accelerator cpu o un host con GPU visible."
        fi
        sim2real_bootstrap_die 40 \
          "Se solicito NVIDIA, pero Ubuntu no expone una GPU NVIDIA PCI de clase grafica. No se continuara silenciosamente con CPU."
      fi
      printf '%s\n' "nvidia"
      ;;
    amd)
      if [[ "${host_kind}" == "wsl" ]]; then
        sim2real_bootstrap_die 40 \
          "AMD/JAX esta bloqueado bajo WSL2: AMD no lo habilita ni valida sobre ROCDXG. Usa Ubuntu nativo con una GPU y una version ROCm incluidas en su matriz."
      fi
      sim2real_bootstrap_validate_amd_native
      printf '%s\n' "amd"
      ;;
    intel)
      if [[ "${host_kind}" == "wsl" ]]; then
        sim2real_bootstrap_die 40 \
          "Intel GPU no esta soportada por JAX bajo WSL2. Usa Ubuntu nativo para experimentar o --accelerator cpu solo para desarrollo."
      fi
      sim2real_bootstrap_die 40 \
        "El plugin Intel disponible exige otra serie de JAX/Flax y no puede ejecutar de forma segura la baseline SIM2REAL actual. El perfil queda bloqueado hasta validarlo en hardware Intel compatible."
      ;;
    cpu)
      printf '%s\n' "cpu"
      ;;
    auto)
      if [[ "${host_kind}" == "wsl" ]]; then
        if sim2real_bootstrap_nvidia_driver_ready; then
          printf '%s\n' "nvidia"
        else
          sim2real_bootstrap_die 40 \
            "WSL2 no ve una ruta GPU JAX admitida y operativa. NVIDIA necesita un driver Windows >= 525; AMD/JAX no esta validado por AMD e Intel/WSL no esta soportado. No se usara CPU en silencio."
        fi
      elif sim2real_bootstrap_nvidia_driver_ready &&
          sim2real_bootstrap_has_pci_display_vendor "0x1002" &&
          sim2real_bootstrap_amd_runtime_ready; then
        sim2real_bootstrap_die 40 \
          "NVIDIA/CUDA y AMD/ROCm estan operativas a la vez. Elige de forma explicita --accelerator nvidia o --accelerator amd para no decidir por ti."
      elif sim2real_bootstrap_nvidia_driver_ready; then
        printf '%s\n' "nvidia"
      elif sim2real_bootstrap_has_pci_display_vendor "0x1002" &&
          sim2real_bootstrap_amd_runtime_ready; then
        printf '%s\n' "amd"
      elif sim2real_bootstrap_has_pci_display_vendor "0x10de"; then
        # En una instalacion desde cero el bootstrap puede preparar este driver
        # despues de elegir NVIDIA. install.sh, que no toca drivers, solo cuenta
        # backends que ya estan operativos.
        printf '%s\n' "nvidia"
      elif sim2real_bootstrap_has_pci_display_vendor "0x1002"; then
        sim2real_bootstrap_validate_amd_native
        printf '%s\n' "amd"
      elif sim2real_bootstrap_has_pci_display_vendor "0x8086"; then
        sim2real_bootstrap_die 40 \
          "Se detecta Intel GPU, pero su plugin exige otra baseline JAX/Flax y aun no esta validado para SIM2REAL. No se usara CPU en silencio."
      else
        virtualization="$(sim2real_bootstrap_detect_virtualization)"
        if [[ "${virtualization}" == "oracle" ]]; then
          sim2real_bootstrap_die 40 \
            "Ubuntu se ejecuta dentro de VirtualBox sin una GPU PCI utilizable. VirtualBox no proporciona esta ruta de computo; usa Ubuntu nativo o WSL2 con GPU visible. CPU solo se activa con --accelerator cpu."
        elif [[ "${virtualization}" != "none" ]]; then
          sim2real_bootstrap_die 40 \
            "La maquina virtual (${virtualization}) no expone una GPU compatible y operativa. Configura passthrough real o usa otro host; no se usara CPU en silencio."
        else
          sim2real_bootstrap_die 40 \
            "No se detecta una GPU compatible. Conecta/prepara NVIDIA o AMD admitida; usa --accelerator cpu solo para desarrollo."
        fi
      fi
      ;;
    *)
      sim2real_bootstrap_die 2 \
        "Acelerador no valido: ${requested:-<vacio>}. Usa compatible, auto, nvidia, amd, intel o cpu."
      ;;
  esac
}

sim2real_bootstrap_nvidia_smi_command() {
  if sim2real_bootstrap_is_wsl && [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
    printf '%s\n' "/usr/lib/wsl/lib/nvidia-smi"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    command -v nvidia-smi
  else
    return 1
  fi
}

sim2real_bootstrap_nvidia_driver_version() {
  local nvidia_smi output version
  nvidia_smi="$(sim2real_bootstrap_nvidia_smi_command)" || return 1
  "${nvidia_smi}" -L >/dev/null 2>&1 || return 1
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

sim2real_bootstrap_nvidia_driver_version_supported() {
  local version="${1:-}" major
  [[ "${version}" =~ ^[0-9]+([.][0-9]+)*$ ]] || return 1
  major="${version%%.*}"
  (( 10#${major} >= 525 ))
}

sim2real_bootstrap_nvidia_driver_ready() {
  local version
  version="$(sim2real_bootstrap_nvidia_driver_version)" || return 1
  sim2real_bootstrap_nvidia_driver_version_supported "${version}"
}

sim2real_bootstrap_state_directory() {
  local state_base
  if [[ -n "${XDG_STATE_HOME:-}" && "${XDG_STATE_HOME}" == /* ]]; then
    state_base="${XDG_STATE_HOME}"
  else
    state_base="${HOME}/.local/state"
  fi
  printf '%s\n' "${state_base}/sim2real-bootstrap"
}

sim2real_bootstrap_driver_state_file() {
  printf '%s/driver-install-boot-id\n' "$(sim2real_bootstrap_state_directory)"
}

sim2real_bootstrap_current_boot_id() {
  [[ -r /proc/sys/kernel/random/boot_id ]] || return 1
  tr -d '[:space:]' </proc/sys/kernel/random/boot_id
}

sim2real_bootstrap_write_driver_state() {
  local state_dir state_file boot_id temporary
  state_dir="$(sim2real_bootstrap_state_directory)"
  state_file="$(sim2real_bootstrap_driver_state_file)"
  boot_id="$(sim2real_bootstrap_current_boot_id)" ||
    sim2real_bootstrap_die 40 "No se puede leer el identificador del arranque actual."
  [[ ! -L "${state_dir}" && ! -L "${state_file}" ]] ||
    sim2real_bootstrap_die 40 "El estado del bootstrap no puede ser un enlace simbolico."
  mkdir -p -- "${state_dir}"
  chmod 700 -- "${state_dir}"
  temporary="$(mktemp "${state_dir}/.driver-state.XXXXXX")"
  printf '%s\n' "${boot_id}" >"${temporary}"
  mv -f -- "${temporary}" "${state_file}"
}

sim2real_bootstrap_clear_driver_state() {
  local state_file
  state_file="$(sim2real_bootstrap_driver_state_file)"
  if [[ -e "${state_file}" || -L "${state_file}" ]]; then
    [[ -f "${state_file}" && ! -L "${state_file}" ]] ||
      sim2real_bootstrap_die 40 "El estado NVIDIA no es un archivo regular seguro: ${state_file}."
    rm -f -- "${state_file}"
  fi
}

sim2real_bootstrap_request_reboot() {
  cat >&2 <<'EOF'

REINICIO NECESARIO
El bootstrap no reinicia el equipo automaticamente.

  1. Ejecuta: sudo reboot
  2. Si Secure Boot muestra MokManager, completa "Enroll MOK".
  3. Abre otra terminal y repite exactamente el mismo comando del bootstrap.
EOF
  exit 20
}

sim2real_bootstrap_recommended_nvidia_package() {
  local output
  local -a packages=()
  if ! output="$(LC_ALL=C ubuntu-drivers devices 2>&1)"; then
    printf '%s\n' "${output}" >&2
    return 1
  fi
  mapfile -t packages < <(
    printf '%s\n' "${output}" |
      awk '
        /recommended/ {
          for (i = 1; i <= NF; i++) {
            if ($i ~ /^nvidia-driver-[0-9]+/) {
              gsub(/[,;]$/, "", $i)
              print $i
            }
          }
        }
      ' |
      LC_ALL=C sort -u
  )
  [[ "${#packages[@]}" == "1" ]] || {
    printf '%s\n' "${output}" >&2
    return 1
  }
  printf '%s\n' "${packages[0]}"
}

sim2real_bootstrap_prepare_nvidia() {
  local state_file saved_boot_id current_boot_id driver_version package package_version
  [[ "$(sim2real_bootstrap_host_kind)" == "native" ]] ||
    sim2real_bootstrap_die 40 \
      "Proteccion WSL2: ubuntu-drivers solo puede ejecutarse en Ubuntu nativo. Instala el driver NVIDIA en Windows."
  state_file="$(sim2real_bootstrap_driver_state_file)"

  if sim2real_bootstrap_nvidia_driver_ready; then
    driver_version="$(sim2real_bootstrap_nvidia_driver_version)"
    sim2real_bootstrap_clear_driver_state
    echo "Driver NVIDIA operativo: ${driver_version}"
    return 0
  fi

  current_boot_id="$(sim2real_bootstrap_current_boot_id)" ||
    sim2real_bootstrap_die 40 "No se puede leer el identificador del arranque actual."
  if [[ -e "${state_file}" || -L "${state_file}" ]]; then
    [[ -f "${state_file}" && ! -L "${state_file}" ]] ||
      sim2real_bootstrap_die 40 "El estado NVIDIA no es un archivo regular seguro: ${state_file}."
    IFS= read -r saved_boot_id <"${state_file}" || true
    [[ "${saved_boot_id}" =~ ^[0-9a-fA-F-]{36}$ ]] ||
      sim2real_bootstrap_die 40 \
        "El estado NVIDIA esta dañado: ${state_file}. Revisalo manualmente."
    if [[ "${saved_boot_id}" == "${current_boot_id}" ]]; then
      echo "El driver NVIDIA se preparo durante este mismo arranque." >&2
      sim2real_bootstrap_request_reboot
    fi
    sim2real_bootstrap_die 40 \
      "El equipo ya se reinicio, pero NVIDIA sigue sin un driver >= 525 operativo. Revisa Secure Boot/MOK, el modulo NVIDIA y nvidia-smi antes de repetir. Estado: ${state_file}"
  fi

  if [[ -f /run/reboot-required || -f /var/run/reboot-required ]]; then
    echo "Ubuntu ya indica que hay un reinicio pendiente antes de preparar NVIDIA." >&2
    sim2real_bootstrap_request_reboot
  fi

  package="$(sim2real_bootstrap_recommended_nvidia_package)" ||
    sim2real_bootstrap_die 40 \
      "ubuntu-drivers no ha identificado exactamente un driver NVIDIA recomendado. Comprueba los repositorios oficiales restricted y el soporte de la GPU."
  [[ "${package}" =~ ^nvidia-driver-[0-9]+([a-z0-9-]*)?$ ]] ||
    sim2real_bootstrap_die 40 "Nombre de paquete NVIDIA inesperado: ${package}."
  package_version="${package#nvidia-driver-}"
  package_version="${package_version%%-*}"
  sim2real_bootstrap_nvidia_driver_version_supported "${package_version}" ||
    sim2real_bootstrap_die 40 \
      "La GPU recomienda ${package}, anterior a la rama 525 requerida por JAX CUDA 12. Usa --accelerator cpu."

  sim2real_bootstrap_step "Instalando el driver oficial recomendado por Ubuntu: ${package}"
  echo "No se instala CUDA Toolkit global; JAX aportara sus librerias CUDA en .venvs/nvidia-cuda12."
  sudo ubuntu-drivers install
  sim2real_bootstrap_write_driver_state
  sim2real_bootstrap_request_reboot
}

sim2real_bootstrap_validate_nvidia_wsl() {
  local driver_version nvidia_smi

  nvidia_smi="$(sim2real_bootstrap_nvidia_smi_command 2>/dev/null || true)"
  if [[ -z "${nvidia_smi}" ]] || ! "${nvidia_smi}" -L >/dev/null 2>&1; then
    sim2real_bootstrap_die 40 \
      "WSL2 no puede usar nvidia-smi. Instala o actualiza en Windows un driver NVIDIA con soporte WSL, ejecuta 'wsl --shutdown' desde PowerShell y repite. No instales ubuntu-drivers, CUDA Toolkit ni un driver NVIDIA Linux dentro de WSL."
  fi

  driver_version="$(sim2real_bootstrap_nvidia_driver_version 2>/dev/null || true)"
  if [[ -z "${driver_version}" ]]; then
    sim2real_bootstrap_die 40 \
      "WSL2 ve la GPU NVIDIA mediante nvidia-smi, pero no puede determinar la version del driver de Windows. Actualiza el driver en Windows, ejecuta 'wsl --shutdown' desde PowerShell y repite; no uses ubuntu-drivers dentro de WSL."
  fi
  if sim2real_bootstrap_nvidia_driver_version_supported "${driver_version}"; then
    echo "Driver NVIDIA de Windows visible en WSL2: ${driver_version}"
    echo "El backend JAX GPU se comprobara despues de instalar el entorno."
    return 0
  fi
  sim2real_bootstrap_die 40 \
    "WSL2 ve el driver NVIDIA de Windows ${driver_version}, pero JAX CUDA 12 requiere >= 525. Actualiza el driver en Windows, ejecuta 'wsl --shutdown' desde PowerShell y repite. No instales un driver NVIDIA Linux dentro de WSL."
}

sim2real_bootstrap_validate_checkout() {
  local checkout="$1" expected_repo_url="$2" marker origin branch dirty top_level
  [[ -d "${checkout}" && ! -L "${checkout}" ]] ||
    sim2real_bootstrap_die 30 "Checkout ausente o inseguro: ${checkout}."
  [[ -d "${checkout}/.git" && ! -L "${checkout}/.git" ]] ||
    sim2real_bootstrap_die 30 "La ruta no contiene un clon Git normal de SIM2REAL: ${checkout}."

  top_level="$(git -C "${checkout}" rev-parse --show-toplevel 2>/dev/null)" ||
    sim2real_bootstrap_die 30 "Git no reconoce el repositorio: ${checkout}."
  [[ "$(realpath -e -- "${top_level}")" == "$(realpath -e -- "${checkout}")" ]] ||
    sim2real_bootstrap_die 30 "La ruta no es la raiz del repositorio Git esperado."

  [[ -f "${checkout}/.sim2real-repository" && ! -L "${checkout}/.sim2real-repository" ]] ||
    sim2real_bootstrap_die 30 "Falta el marcador de SIM2REAL en ${checkout}."
  IFS= read -r marker <"${checkout}/.sim2real-repository" || true
  [[ "${marker}" == "${SIM2REAL_BOOTSTRAP_MARKER}" ]] ||
    sim2real_bootstrap_die 30 "El marcador del repositorio no corresponde a SIM2REAL Linux/WSL."
  [[ -f "${checkout}/pyproject.toml" && ! -L "${checkout}/pyproject.toml" ]] ||
    sim2real_bootstrap_die 30 "Falta pyproject.toml en el repositorio."
  grep -Eq '^name[[:space:]]*=[[:space:]]*"sim2real-mjx-jax"[[:space:]]*$' \
    "${checkout}/pyproject.toml" ||
    sim2real_bootstrap_die 30 "pyproject.toml no identifica el proyecto SIM2REAL."
  [[ -f "${checkout}/scripts/install.sh" && ! -L "${checkout}/scripts/install.sh" && -x "${checkout}/scripts/install.sh" ]] ||
    sim2real_bootstrap_die 30 "Falta scripts/install.sh o no es un ejecutable regular."

  origin="$(git -C "${checkout}" remote get-url origin 2>/dev/null)" ||
    sim2real_bootstrap_die 30 "El repositorio no tiene un remote origin."
  [[ "$(sim2real_bootstrap_normalize_repo_url "${origin}" 2>/dev/null || true)" == \
      "$(sim2real_bootstrap_normalize_repo_url "${expected_repo_url}")" ]] ||
    sim2real_bootstrap_die 30 \
      "origin no coincide con --repo-url. Actual: ${origin}; esperado: ${expected_repo_url}. No se modifica automaticamente."
  branch="$(git -C "${checkout}" symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
  [[ "${branch}" == "main" ]] ||
    sim2real_bootstrap_die 30 \
      "El repositorio debe estar en la rama main (actual: ${branch:-detached})."
  dirty="$(git -C "${checkout}" status --porcelain=v1 --untracked-files=all)" ||
    sim2real_bootstrap_die 30 "No se puede comprobar el estado de Git."
  [[ -z "${dirty}" ]] ||
    sim2real_bootstrap_die 30 \
      "Hay cambios o archivos locales sin versionar en ${checkout}. No se sobrescribe nada."
}

sim2real_bootstrap_notice_temporary_clone() {
  if [[ -n "${SIM2REAL_BOOTSTRAP_TEMP_CLONE}" && -d "${SIM2REAL_BOOTSTRAP_TEMP_CLONE}" ]]; then
    echo "Aviso: se conserva la copia temporal para diagnostico: ${SIM2REAL_BOOTSTRAP_TEMP_CLONE}" >&2
  fi
}

sim2real_bootstrap_sync_repository() {
  local install_path="$1" repo_url="$2" parent temp_root checkout
  parent="$(dirname -- "${install_path}")"
  mkdir -p -- "${parent}"
  [[ -d "${parent}" && ! -L "${parent}" && -w "${parent}" ]] ||
    sim2real_bootstrap_die 30 "La carpeta padre no es segura o escribible: ${parent}."

  if [[ -e "${install_path}" || -L "${install_path}" ]]; then
    [[ -d "${install_path}" && ! -L "${install_path}" ]] ||
      sim2real_bootstrap_die 30 \
        "La ruta ya existe y no es un directorio normal: ${install_path}."
    sim2real_bootstrap_validate_checkout "${install_path}" "${repo_url}"
    sim2real_bootstrap_step "Actualizando SIM2REAL mediante fast-forward"
    git -c core.hooksPath=/dev/null -C "${install_path}" fetch \
      --prune --no-tags origin \
      +refs/heads/main:refs/remotes/origin/main
    git -C "${install_path}" merge-base --is-ancestor \
      HEAD refs/remotes/origin/main ||
      sim2real_bootstrap_die 30 \
        "La rama local contiene commits propios o ha divergido de origin/main. No se modifica."
    git -c core.hooksPath=/dev/null -C "${install_path}" merge \
      --ff-only refs/remotes/origin/main
    sim2real_bootstrap_validate_checkout "${install_path}" "${repo_url}"
    return 0
  fi

  sim2real_bootstrap_step "Clonando SIM2REAL en una carpeta temporal segura"
  temp_root="$(mktemp -d "${parent}/.sim2real-bootstrap-clone.XXXXXX")"
  SIM2REAL_BOOTSTRAP_TEMP_CLONE="${temp_root}"
  checkout="${temp_root}/checkout"
  git -c core.hooksPath=/dev/null clone \
    --branch main --single-branch --origin origin -- \
    "${repo_url}" "${checkout}"
  sim2real_bootstrap_validate_checkout "${checkout}" "${repo_url}"
  [[ ! -e "${install_path}" && ! -L "${install_path}" ]] ||
    sim2real_bootstrap_die 30 \
      "La ruta de destino aparecio durante el clonado; se conserva la copia temporal y no se sobrescribe."
  mv -Tn -- "${checkout}" "${install_path}"
  [[ ! -e "${checkout}" ]] ||
    sim2real_bootstrap_die 30 \
      "No se pudo mover el clon sin sobrescribir la ruta de destino."
  rmdir -- "${temp_root}"
  SIM2REAL_BOOTSTRAP_TEMP_CLONE=""
  sim2real_bootstrap_validate_checkout "${install_path}" "${repo_url}"
}

sim2real_bootstrap_run_acceptance() {
  local install_path="$1" accelerator="$2"
  sim2real_bootstrap_step "Instalando SIM2REAL con perfil ${accelerator}"
  (
    cd -- "${install_path}"
    ./scripts/install.sh --accelerator "${accelerator}"

    sim2real_bootstrap_step "Repitiendo una prueba fisica MJX independiente"
    ./scripts/sim2real.sh test-mjx --steps 10

    sim2real_bootstrap_step "Cargando el modelo preentrenado sin abrir ventana"
    ./scripts/sim2real.sh visualizar-modelo-preentrenado --solo-comprobar
  )
}

sim2real_bootstrap_main() (
  set -Eeuo pipefail
  umask 022
  trap sim2real_bootstrap_notice_temporary_clone EXIT

  local requested_accelerator="" install_path_argument=""
  local repo_url="${SIM2REAL_BOOTSTRAP_DEFAULT_REPO_URL}"
  local host_kind install_path resolved_accelerator preflight_status

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --accelerator)
        sim2real_bootstrap_require_option_value "$1" "$#" "${2:-}"
        requested_accelerator="$2"
        shift 2
        ;;
      --accelerator=*)
        requested_accelerator="${1#*=}"
        shift
        ;;
      --install-path)
        sim2real_bootstrap_require_option_value "$1" "$#" "${2:-}"
        install_path_argument="$2"
        shift 2
        ;;
      --install-path=*)
        install_path_argument="${1#*=}"
        [[ -n "${install_path_argument}" ]] ||
          sim2real_bootstrap_die 2 "Falta el valor de --install-path."
        shift
        ;;
      --repo-url)
        sim2real_bootstrap_require_option_value "$1" "$#" "${2:-}"
        repo_url="$2"
        shift 2
        ;;
      --repo-url=*)
        repo_url="${1#*=}"
        shift
        ;;
      -h|--help)
        sim2real_bootstrap_usage
        return 0
        ;;
      *)
        sim2real_bootstrap_usage >&2
        sim2real_bootstrap_die 2 "Argumento no reconocido: $1."
        ;;
    esac
  done

  [[ -n "${requested_accelerator}" ]] || {
    sim2real_bootstrap_usage >&2
    sim2real_bootstrap_die 2 \
      "Debes elegir --accelerator compatible, auto, nvidia, amd, intel o cpu."
  }
  case "${requested_accelerator}" in
    compatible|auto|nvidia|amd|intel|cpu) ;;
    *)
      sim2real_bootstrap_die 2 \
        "Acelerador no valido: ${requested_accelerator:-<vacio>}. Usa compatible, auto, nvidia, amd, intel o cpu."
      ;;
  esac
  sim2real_bootstrap_validate_repo_url "${repo_url}" ||
    sim2real_bootstrap_die 2 \
      "--repo-url debe tener la forma https://github.com/<owner>/Sim2Real-MJX-JAX-sobre-Linux-WSL[.git]."

  sim2real_bootstrap_validate_host
  host_kind="$(sim2real_bootstrap_host_kind)"
  if [[ "${host_kind}" == "wsl" ]]; then
    sim2real_bootstrap_add_wsl_driver_path
  fi
  install_path="$(sim2real_bootstrap_resolve_install_path "${install_path_argument}")"
  if [[ "${host_kind}" == "wsl" ]]; then
    sim2real_bootstrap_validate_wsl_install_path "${install_path}"
  fi
  echo "Plataforma detectada: $(sim2real_bootstrap_platform_description "${host_kind}")"
  echo "Acelerador solicitado: ${requested_accelerator}"
  echo "Ruta de instalacion: ${install_path}"

  # El hardware y la combinacion sistema/acelerador se comprueban antes de
  # ejecutar APT o tocar el repositorio. Una solicitud GPU incompatible falla;
  # `compatible` puede continuar por CPU, pero siempre muestra el aviso.
  if resolved_accelerator="$(
    sim2real_bootstrap_resolve_accelerator "${requested_accelerator}" "${host_kind}"
  )"; then
    :
  else
    preflight_status=$?
    sim2real_bootstrap_print_accelerator_support_summary >&2
    return "${preflight_status}"
  fi
  echo "Acelerador resuelto: ${resolved_accelerator}"

  if [[ "${resolved_accelerator}" == "nvidia" && "${host_kind}" == "wsl" ]]; then
    sim2real_bootstrap_step "Comprobando el driver NVIDIA de Windows desde WSL2"
    if ( sim2real_bootstrap_validate_nvidia_wsl ); then
      :
    else
      preflight_status=$?
      sim2real_bootstrap_print_accelerator_support_summary >&2
      return "${preflight_status}"
    fi
  fi

  sim2real_bootstrap_install_base_packages "${host_kind}"

  if [[ "${resolved_accelerator}" == "nvidia" && "${host_kind}" != "wsl" ]]; then
    sim2real_bootstrap_step "Comprobando el driver NVIDIA del host nativo"
    sim2real_bootstrap_install_nvidia_management_packages
    sim2real_bootstrap_prepare_nvidia
  elif [[ "${resolved_accelerator}" == "amd" ]]; then
    echo "Perfil AMD: por seguridad el bootstrap no instala ni sustituye amdgpu/ROCm."
    echo "Se exige Ubuntu nativo, una GPU incluida por AMD y ROCm 7.0.0-7.0.2 ya operativo; el instalador lo comprobara antes de crear el entorno."
  else
    echo "Perfil CPU: el bootstrap no instala ni modifica drivers de GPU."
  fi

  sim2real_bootstrap_sync_repository "${install_path}" "${repo_url}"
  sim2real_bootstrap_run_acceptance "${install_path}" "${resolved_accelerator}"

  cat <<EOF

Instalacion terminada y verificada.
Repositorio: ${install_path}
Acelerador: ${resolved_accelerator}
EOF
)

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  sim2real_bootstrap_main "$@"
fi
