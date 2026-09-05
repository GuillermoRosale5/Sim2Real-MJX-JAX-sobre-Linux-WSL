# Instalación de Sim2Real MJX-JAX sobre Linux/WSL

Este repositorio instala el entorno de simulación en una única carpeta de
Linux. El mismo código funciona en Ubuntu nativo y en Ubuntu dentro de WSL2;
el instalador detecta cuál de los dos sistemas estamos utilizando y prepara el
perfil compatible.

> **La instalación para Windows con WSL2 aparece primero.**
>
> Si vas a instalar el proyecto directamente en **Ubuntu nativo**, baja hasta
> [Instalación en Ubuntu nativo](#instalación-en-ubuntu-nativo).

## Compatibilidad antes de instalar

| Equipo | Perfil utilizado | Situación actual |
|---|---|---|
| NVIDIA en WSL2 | `nvidia` | GPU mediante CUDA 12. Es la ruta GPU validada de principio a fin en este proyecto, sobre Ubuntu 24.04. |
| NVIDIA en Ubuntu 22.04 o 24.04 nativo | `nvidia` | Ruta implementada, pero no validada físicamente en este proyecto. El instalador comprueba el controlador y puede preparar el paquete oficial de Ubuntu. |
| AMD en Ubuntu 24.04 nativo | `amd` | Ruta experimental implementada, pero no validada físicamente en este proyecto. Requiere ROCm 7.0.0–7.0.2 ya operativo. |
| AMD en WSL2 | `cpu` | La GPU AMD no está admitida en esta configuración. |
| Intel en Ubuntu o WSL2 | `cpu` | La GPU Intel no está admitida en esta configuración. |
| Equipo sin GPU admitida | `cpu` | Sirve para instalar, desarrollar y hacer pruebas pequeñas; no es adecuado para entrenamiento masivo. |

El bloque recomendado usa `--accelerator compatible`. Este modo selecciona
NVIDIA cuando CUDA está disponible, AMD en Ubuntu nativo cuando ROCm ya
funciona y CPU cuando no encuentra una GPU admitida. Si utiliza CPU lo indica
de forma visible; nunca la presenta como aceleración GPU.

## Instalación desde Windows con WSL2

### Autoinstalador completo desde PowerShell

En un equipo Windows nuevo no hace falta descargar primero el repositorio ni
instalar Git, Python, CUDA o las librerías del proyecto a mano. Abre
**PowerShell como administrador** y pega este bloque completo:

```powershell
$sim2realInstaller = Join-Path ([IO.Path]::GetTempPath()) ("sim2real-install-" + [guid]::NewGuid() + ".ps1")
try {
  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
  Invoke-WebRequest -UseBasicParsing `
    -Uri 'https://raw.githubusercontent.com/GuillermoRosale5/Sim2Real-MJX-JAX-sobre-Linux-WSL/main/scripts/install_windows.ps1' `
    -OutFile $sim2realInstaller
  $sim2realHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $sim2realInstaller).Hash.ToLowerInvariant()
  if ($sim2realHash -ne '6400d9686c75af0ad2dae026129545f62de068ab450272b51506707c82ffaa70') {
    throw "El instalador descargado no coincide con el SHA-256 publicado."
  }
  powershell -NoProfile -ExecutionPolicy Bypass -File $sim2realInstaller
  if ($LASTEXITCODE -ne 0) { throw "La instalación terminó con código $LASTEXITCODE." }
}
finally {
  Remove-Item -LiteralPath $sim2realInstaller -Force -ErrorAction SilentlyContinue
}
```

El instalador muestra en directo cada orden que se ejecuta dentro de Ubuntu.
Si WSL o Ubuntu todavía no existen, los prepara y pide reiniciar o crear el
usuario Linux cuando sea necesario. Después de ese primer arranque se vuelve a
pegar exactamente el mismo bloque y la instalación continúa sola. La copia de
trabajo queda en `~/robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL`, dentro del
sistema de archivos de Linux.

El SHA-256 pertenece al archivo publicado actualmente en `main`. Si el
repositorio cambia, utiliza el bloque que aparezca en el README más reciente.

### Preparar Ubuntu dentro de Windows

Los siguientes pasos explican manualmente lo que comprueba el autoinstalador.
Solo hacen falta si Windows pide completar la activación inicial de WSL o si
queremos revisar la configuración.

Abre **PowerShell como administrador** y ejecuta:

```powershell
wsl --install -d Ubuntu-24.04
```

Reinicia Windows si lo solicita. Después abre `Ubuntu-24.04` desde el menú
Inicio y crea el usuario y la contraseña de Linux. Cuando escribas la
contraseña no aparecerán caracteres en la terminal; es normal.

Vuelve a PowerShell y comprueba que Ubuntu utiliza WSL2:

```powershell
wsl --list --verbose
```

La columna `VERSION` debe mostrar `2`. Si muestra `1`, conviértelo con:

```powershell
wsl --set-version Ubuntu-24.04 2
```

### Comprobar NVIDIA en WSL2

Este paso solo corresponde a equipos NVIDIA. Abre la terminal de Ubuntu y
ejecuta:

```bash
nvidia-smi
```

Si no funciona, instala o actualiza en Windows el controlador NVIDIA con
soporte para WSL. Después ejecuta `wsl --shutdown` desde PowerShell, vuelve a
abrir Ubuntu y repite `nvidia-smi`. No instales un controlador NVIDIA de Linux,
CUDA Toolkit ni `ubuntu-drivers` dentro de WSL2.

Los equipos AMD o Intel en WSL2 pueden continuar: el instalador seleccionará el
perfil CPU y mostrará su limitación.

### Ejecutar el autoinstalador en WSL2

Abre `Ubuntu-24.04` con tu usuario normal. Copia y pega **el bloque completo**,
desde el primer `(` hasta el último `)`:

```bash
(
  set -e
  sudo -v
  if ! command -v curl >/dev/null 2>&1 ||
      [[ ! -s /etc/ssl/certs/ca-certificates.crt ]]; then
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl
  fi

  sim2real_bootstrap_file="$(mktemp)"
  trap 'rm -f -- "$sim2real_bootstrap_file"' EXIT
  curl --proto '=https' --tlsv1.2 --fail --show-error --location \
    --output "$sim2real_bootstrap_file" \
    'https://raw.githubusercontent.com/GuillermoRosale5/Sim2Real-MJX-JAX-sobre-Linux-WSL/main/scripts/bootstrap_ubuntu.sh'
  printf '%s  %s\n' \
    'ee5c79a8c566ecdc5ec087ddb75ca833dfca538ed00c16c2e87d6458d17d3974' \
    "$sim2real_bootstrap_file" | sha256sum --check -
  bash -n "$sim2real_bootstrap_file"
  bash "$sim2real_bootstrap_file" --accelerator compatible
)
```

El checksum corresponde al `bootstrap_ubuntu.sh` de la revisión publicada en
`main`. Si GitHub contiene una revisión posterior, utiliza siempre el bloque
completo que aparezca en su README actualizado.

El proceso instala las herramientas del sistema, crea la carpeta
`~/robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL`, prepara un entorno aislado para
el acelerador detectado y ejecuta las comprobaciones de instalación. No hace
falta crear manualmente la carpeta del proyecto ni clonar antes el repositorio.

## Instalación en Ubuntu nativo

Esta sección corresponde a un equipo que arranca Ubuntu directamente, no a una
máquina virtual. VirtualBox no expone una GPU NVIDIA o AMD real para CUDA o
ROCm; allí solo se puede validar el perfil CPU.

La instalación admite Ubuntu 22.04 LTS y Ubuntu 24.04 LTS sobre `x86_64`.
Compruébalo con:

```bash
uname -m
. /etc/os-release && printf '%s %s\n' "$ID" "$VERSION_ID"
```

La primera orden debe mostrar `x86_64`; la segunda, `ubuntu 22.04` o
`ubuntu 24.04`.

### NVIDIA en Ubuntu nativo

El instalador comprueba la GPU y el controlador. Si debe instalar el
controlador oficial de Ubuntu, puede terminar mostrando `REINICIO NECESARIO`.
En ese caso:

1. Ejecuta `sudo reboot`.
2. Completa el alta de MOK si Secure Boot la solicita.
3. Vuelve a copiar el mismo bloque de instalación.

No es necesario instalar un CUDA Toolkit global: las bibliotecas utilizadas
por JAX quedan dentro del entorno del proyecto.

### AMD en Ubuntu nativo

La ruta AMD requiere Ubuntu 24.04 y una GPU incluida en la matriz de
compatibilidad de ROCm. Antes de instalar el proyecto deben funcionar:

```bash
test -r /dev/kfd && test -w /dev/kfd && echo "/dev/kfd: permisos correctos"
rocminfo
```

El instalador acepta ROCm 7.0.0, 7.0.1 y 7.0.2. No instala ROCm ni modifica su
controlador. Si estas comprobaciones fallan, prepara primero ROCm siguiendo la
documentación oficial de AMD.

### Ejecutar el autoinstalador en Ubuntu nativo

Copia en una terminal el mismo bloque completo utilizado en WSL2:

```bash
(
  set -e
  sudo -v
  if ! command -v curl >/dev/null 2>&1 ||
      [[ ! -s /etc/ssl/certs/ca-certificates.crt ]]; then
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl
  fi

  sim2real_bootstrap_file="$(mktemp)"
  trap 'rm -f -- "$sim2real_bootstrap_file"' EXIT
  curl --proto '=https' --tlsv1.2 --fail --show-error --location \
    --output "$sim2real_bootstrap_file" \
    'https://raw.githubusercontent.com/GuillermoRosale5/Sim2Real-MJX-JAX-sobre-Linux-WSL/main/scripts/bootstrap_ubuntu.sh'
  printf '%s  %s\n' \
    'ee5c79a8c566ecdc5ec087ddb75ca833dfca538ed00c16c2e87d6458d17d3974' \
    "$sim2real_bootstrap_file" | sha256sum --check -
  bash -n "$sim2real_bootstrap_file"
  bash "$sim2real_bootstrap_file" --accelerator compatible
)
```

Utiliza un usuario normal con permiso para ejecutar `sudo`; no lances el bloque
mediante `sudo bash`.

## Instalación manual de respaldo

Si necesitas revisar cada paso o trabajas detrás de un proxy, puedes clonar e
instalar manualmente:

```bash
sudo apt-get update
sudo apt-get install -y git
mkdir -p ~/robotica
git clone https://github.com/GuillermoRosale5/Sim2Real-MJX-JAX-sobre-Linux-WSL.git ~/robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL
cd ~/robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL
./scripts/install.sh --accelerator compatible
```

El instalador es repetible: conserva los entornos válidos y los resultados
locales. Si una instalación se interrumpe, entra en la carpeta y ejecuta de
nuevo `./scripts/install.sh --accelerator compatible`.

## Problemas durante la instalación

- **La terminal vuelve al prompt después de `Reading package lists...`:** esa
  orden ha terminado; continúa con el bloque completo del autoinstalador. No
  pegues una segunda orden en la misma línea que `sudo apt-get update`.
- **El repositorio está en `/mnt/c`:** vuelve a instalarlo bajo `~/robotica`.
  WSL2 debe ejecutar el entorno desde el sistema de archivos de Linux.
- **`nvidia-smi` falla en WSL2:** actualiza el controlador NVIDIA de Windows y
  ejecuta `wsl --shutdown`. No instales un controlador Linux dentro de WSL2.
- **El instalador selecciona CPU:** no ha encontrado una ruta GPU admitida. El
  mensaje anterior indica si se trata de AMD en WSL2, Intel, una máquina
  virtual o un controlador que todavía no funciona.
- **ROCm no puede acceder a `/dev/kfd`:** revisa los grupos `render` y `video`,
  cierra la sesión y comprueba la instalación oficial de ROCm.
- **La instalación solicita reiniciar Ubuntu nativo:** reinicia, completa MOK
  si aparece y ejecuta otra vez el mismo bloque.

La explicación técnica, los comandos de trabajo y la estructura del proyecto
se conservan en la carpeta [`docs`](docs/).
