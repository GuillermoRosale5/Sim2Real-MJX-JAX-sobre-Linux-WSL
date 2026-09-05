# Instalación y resolución de problemas

Esta guía permite preparar Sim2Real MJX-JAX desde cero en Ubuntu nativo o en Ubuntu
dentro de WSL2. El código de simulación es el mismo en los dos casos. Lo que
cambia es la forma en la que el sistema operativo expone la GPU y su
controlador.

## Compatibilidad actual

El instalador trabaja con estos perfiles:

| Equipo | Perfil | Estado |
|---|---|---|
| NVIDIA en Ubuntu nativo | `nvidia` | Implementado, pero sin validación física en este proyecto. El instalador puede preparar el controlador oficial de Ubuntu. |
| NVIDIA en WSL2 | `nvidia` | Validado de principio a fin con Ubuntu 24.04 cuando `nvidia-smi` funciona. El controlador se instala en Windows. |
| AMD en Ubuntu 24.04 nativo | `amd` | Experimental e implementado, pero sin validación física en este proyecto. Requiere una GPU admitida y ROCm 7.0.0–7.0.2 ya operativo. |
| AMD en WSL2 | `cpu` | La GPU AMD no está admitida en esta versión. |
| Intel en Ubuntu o WSL2 | `cpu` | La GPU Intel no está admitida en esta versión. |

El bloque de instalación recomendado utiliza `--accelerator compatible`. Este
modo elige automáticamente NVIDIA, AMD nativa preparada o CPU. Cuando termina
en CPU lo comunica de forma visible; nunca presenta la CPU como si fuera una
GPU.

El perfil CPU sirve para desarrollar y realizar pruebas pequeñas. No es una
ruta realista para un entrenamiento PPO masivo.

## Requisitos previos

La instalación está preparada para Ubuntu 22.04 LTS y Ubuntu 24.04 LTS en
equipos `x86_64`, tanto en nativo como en WSL2. Antes de empezar se puede
comprobar con:

```bash
uname -m
. /etc/os-release && printf '%s %s\n' "$ID" "$VERSION_ID"
```

La primera orden debe mostrar `x86_64`. La segunda debe mostrar `ubuntu 22.04`
o `ubuntu 24.04`. WSL1, ARM, otras distribuciones y versiones intermedias de
Ubuntu quedan fuera del sistema probado.

También hacen falta conexión a Internet y espacio suficiente para el entorno
virtual. Sim2Real MJX-JAX fija Python 3.12 y `uv` descarga esa versión cuando no está
disponible en el sistema.

En WSL2 el repositorio debe quedar dentro del sistema de archivos de Linux, por
ejemplo bajo `/home/usuario/robotica`. No debe instalarse en `/mnt/c`, en otro
disco de Windows montado ni en una ruta compartida equivalente. Windows puede
editar la misma copia mediante Remote WSL o `\\wsl.localhost`; no hace falta
mantener dos carpetas sincronizadas.

## Instalación automática en Ubuntu nativo o WSL2

Este es el procedimiento principal. Se ejecuta desde una terminal de Ubuntu con
un usuario normal que tenga permiso para usar `sudo`. No debe lanzarse mediante
`sudo bash`.

Copia el bloque completo, desde el primer `(` hasta el último `)`, sin modificar
el acelerador. Cuando aparezca `[sudo] password`, escribe la contraseña del
usuario de Ubuntu y pulsa Intro. La terminal no muestra asteriscos mientras se
escribe.

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

El enlace descarga el instalador publicado en `main`. El valor SHA-256 debe
coincidir exactamente con el archivo de esa revisión. Utiliza siempre el bloque
completo que figure en el README actual: primero descarga el programa, después
lo verifica y finalmente lo ejecuta; no utiliza la forma insegura `curl | bash`.

El instalador realiza de forma automática estas operaciones:

- distingue Ubuntu nativo de WSL2 y rechaza WSL1;
- comprueba la arquitectura, la versión de Ubuntu y la ubicación del proyecto;
- detecta el hardware antes de instalar las dependencias internas;
- elige un perfil compatible y crea su entorno aislado;
- clona el repositorio en `~/robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL`, o lo actualiza si
  ya existe, está limpio y continúa en la rama `main`;
- ejecuta el diagnóstico, una prueba corta de MJX y la carga sin ventana de la
  red preentrenada.

No borra cambios locales ni sustituye un repositorio desconocido. Si encuentra
una copia modificada, otra rama o un remoto incompatible, se detiene para no
perder trabajo.

Los entornos quedan separados por fabricante:

```text
.venvs/nvidia-cuda12   NVIDIA + CUDA 12
.venvs/amd-rocm70      AMD + ROCm 7.0
.venvs/cpu             CPU
```

No deben mezclarse paquetes CUDA y ROCm dentro del mismo entorno ni copiar una
`.venv` entre ordenadores.

## NVIDIA en Ubuntu nativo

En Ubuntu nativo el instalador comprueba la GPU por PCI. Si el controlador ya
funciona, continúa directamente. Si falta, utiliza `ubuntu-drivers` para
preparar el paquete oficial recomendado por Ubuntu.

La instalación del controlador puede necesitar un reinicio. En ese caso el
programa termina con el código `20` y muestra `REINICIO NECESARIO`. El propio
instalador no reinicia el equipo:

1. Ejecuta `sudo reboot`.
2. Si aparece MokManager, selecciona **Enroll MOK** e introduce la contraseña
   temporal creada durante la instalación del controlador.
3. Vuelve a iniciar sesión y repite el mismo bloque completo.

El segundo intento continúa donde corresponde. Si después del reinicio
`nvidia-smi` sigue sin funcionar, se detiene con el código `40` para revisar el
controlador, el kernel, Secure Boot o el alta de MOK. No entra en un bucle de
reinstalación.

Para JAX con CUDA 12 se necesita un controlador NVIDIA de la serie 525 o
posterior. No hace falta instalar un CUDA Toolkit global: las bibliotecas CUDA y
cuDNN utilizadas por JAX se instalan dentro del entorno de Sim2Real MJX-JAX.

## NVIDIA en WSL2

En WSL2 el controlador NVIDIA pertenece a Windows. Sim2Real MJX-JAX no instala
`ubuntu-drivers`, `mokutil`, DKMS ni otro controlador Linux dentro de Ubuntu.

Antes de ejecutar el instalador debe funcionar:

```bash
nvidia-smi
```

Si no funciona o muestra una versión anterior a 525, actualiza en Windows el
[controlador NVIDIA con soporte para WSL](https://learn.microsoft.com/es-es/windows/ai/directml/gpu-cuda-in-wsl),
ejecuta lo siguiente desde PowerShell y vuelve a abrir Ubuntu:

```powershell
wsl --shutdown
```

Después se repite el mismo bloque de instalación. WSL2 no utiliza MokManager ni
necesita instalar un módulo NVIDIA dentro de la distribución.

## Preparación inicial de WSL2

En un Windows que todavía no tenga WSL, abre PowerShell como administrador y
ejecuta:

```powershell
wsl --install -d Ubuntu-24.04
```

Reinicia Windows si lo solicita, abre Ubuntu y crea el usuario Linux. A partir
de ese momento todo el procedimiento de Sim2Real MJX-JAX se ejecuta en la terminal de
Ubuntu con el bloque automático anterior.

La versión de cada distribución se consulta desde PowerShell con:

```powershell
wsl --list --verbose
```

Debe aparecer la versión `2`. Si una distribución está en WSL1, se convierte
con `wsl --set-version <Distro> 2`.

### Inicio desde PowerShell

Esta es la entrada recomendada en un Windows nuevo. No necesita una copia
previa del repositorio, Git, Python, CUDA ni las dependencias de MuJoCo. Abre
PowerShell como administrador y pega el bloque completo:

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

El bloque verifica el SHA-256 antes de ejecutar el archivo. El instalador
comprueba o instala la distribución indicada, garantiza que utiliza WSL2,
prepara las herramientas básicas dentro de Ubuntu y crea la copia ejecutable
en `~/robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL`. Después llama al instalador
común `scripts/install.sh` dentro de Linux y muestra su salida en directo.

Si Windows necesita reiniciarse o Ubuntu todavía no tiene un usuario creado,
abrimos la distribución una vez, terminamos esa preparación y repetimos el
mismo bloque. Esta ruta mantiene una sola copia de trabajo dentro de Linux, que
Windows puede abrir mediante `\\wsl.localhost` o con una herramienta compatible
con Remote WSL.

Si el repositorio ya está descargado y limpio, también se puede ejecutar
directamente desde su raíz:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1
```

## AMD en Ubuntu nativo

La ruta AMD es experimental y solo está preparada para Ubuntu 24.04 nativo
`x86_64`. ROCm debe estar instalado antes de Sim2Real MJX-JAX. La GPU, la versión
puntual de Ubuntu y el kernel deben figurar en la
[matriz oficial de ROCm 7.0.2](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.0.2/reference/system-requirements.html).

Después de seguir la
[instalación oficial de ROCm para Ubuntu](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.0.2/install/install-methods/package-manager/package-manager-ubuntu.html),
comprueba:

```bash
test -r /dev/kfd && test -w /dev/kfd && echo "/dev/kfd: permisos OK"
rocminfo
./scripts/install.sh --accelerator amd
```

Sim2Real MJX-JAX acepta ROCm 7.0.0, 7.0.1 y 7.0.2. El instalador comprueba
`/dev/kfd`, sus permisos, `rocminfo` y la presencia de un agente GPU `gfx`.
Después instala la versión de JAX preparada para ROCm y verifica un cálculo JIT
y un paso de MJX.

Que `doctor` termine correctamente demuestra que el entorno funciona en esa
ejecución, pero no convierte en oficial una GPU que no aparezca en la matriz de
AMD. Esta ruta aún no se ha validado sobre una GPU AMD física del proyecto.

AMD sobre WSL2 no está admitido. El modo `compatible` utiliza CPU con un aviso;
`auto` y `--accelerator amd` terminan con error.

## Equipos Intel y uso de CPU

La GPU Intel no se activa en esta versión. La extensión OpenXLA de Intel
considerada no es compatible con las versiones de JAX, Flax, Brax y MJX fijadas
por Sim2Real MJX-JAX.

El bloque automático selecciona CPU y explica el motivo. También puede forzarse
de forma explícita con:

```bash
./scripts/install.sh --accelerator cpu
```

Este perfil permite comprobar la instalación, cargar el modelo, compilar los
XML y realizar simulaciones pequeñas. Un entrenamiento PPO completo sería
extremadamente lento.

## Selección manual del acelerador

El modo recomendado para una primera instalación es `compatible`. Los demás
modos quedan disponibles para una necesidad concreta:

- `--accelerator nvidia` exige una NVIDIA operativa y nunca cae a CPU;
- `--accelerator amd` exige AMD, Ubuntu 24.04 nativo y ROCm preparado;
- `--accelerator cpu` fuerza CPU aunque exista una GPU compatible;
- `--accelerator auto` exige alguna GPU admitida y termina con error si solo
  puede utilizar CPU.

Si NVIDIA y AMD están operativas al mismo tiempo, `auto` solicita que se elija
una de ellas. Tras la instalación, los comandos normales reutilizan el último
perfil que superó todas las comprobaciones.

Las opciones avanzadas del archivo local son:

```bash
./scripts/bootstrap_ubuntu.sh --accelerator nvidia \
  --install-path ~/robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL \
  --repo-url https://github.com/GuillermoRosale5/Sim2Real-MJX-JAX-sobre-Linux-WSL.git
```

`--install-path` debe apuntar a una ruta dentro del directorio personal.
`--repo-url` permite utilizar una copia derivada en GitHub que conserve el nombre
`Sim2Real-MJX-JAX-sobre-Linux-WSL`. Ninguna de las dos opciones hace falta en una
instalación normal.

## Instalación manual de respaldo

La ruta automática es la recomendada. Si hace falta revisar cada paso o trabajar
detrás de un proxy, se puede clonar el repositorio y ejecutar el instalador
interno:

```bash
sudo apt update
sudo apt install -y git
mkdir -p ~/robotica
git clone https://github.com/GuillermoRosale5/Sim2Real-MJX-JAX-sobre-Linux-WSL.git ~/robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL
cd ~/robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL
./scripts/install.sh --accelerator compatible
```

Si la carpeta ya existe no se repite `git clone`. Basta con entrar en ella y
ejecutar `./scripts/install.sh --accelerator compatible`.

El instalador interno es idempotente: crea o repara únicamente el perfil
elegido, conserva los demás entornos y restaura el estado anterior si una
instalación falla o se interrumpe. No modifica controladores propietarios, no
instala ROCm y no borra los resultados de los entrenamientos.

## Comprobación de la instalación

Al terminar se pueden repetir las comprobaciones principales:

```bash
./scripts/doctor.sh
./scripts/sim2real.sh test-mjx --steps 150
./scripts/sim2real.sh visualizar-modelo-preentrenado --solo-comprobar
```

`doctor` debe mostrar `gpu` para los perfiles NVIDIA y AMD, y `cpu` para el
perfil CPU. Un perfil GPU que termine utilizando CPU se considera incorrecto y
produce un error.

La última orden comprueba la red incluida sin abrir una ventana. Si termina
correctamente, se puede visualizar con:

```bash
./scripts/visualizar_modelo_preentrenado.sh
```

Esta visualización utiliza siempre la red versionada de fase 2 y paso
45.932.544. No consulta `logs_sim2real_mjx/ultima_ejecucion.txt`, por lo que un
entrenamiento local incompleto no sustituye los pesos de referencia.

## Validación de Ubuntu nativo

WSL2 ejecuta el mismo código, pero no valida el controlador Linux, el kernel,
Secure Boot ni MOK. Para comprobar de forma realista Ubuntu nativo sin crear una
partición en el disco interno existen estas opciones:

- **Live USB:** permite comprobar el arranque y los periféricos, pero no incluye
  el controlador propietario NVIDIA y no certifica JAX/CUDA.
- **Instalación completa en un SSD externo:** cuenta como Ubuntu nativo y
  permite instalar controladores, reiniciar, completar MOK, crear los entornos
  y guardar resultados. Es la opción más realista sin tocar el disco interno.
- **VirtualBox 7.2:** sirve para probar CPU y los scripts. Su adaptador gráfico
  virtual no expone la GPU PCI como CUDA o ROCm.
- **WSL2:** valida el funcionamiento diario Windows/WSL y, con NVIDIA, la
  comunicación con el controlador de Windows.

La ruta NVIDIA en WSL2 se ha validado de extremo a extremo en el equipo del
proyecto. La ruta NVIDIA nativa está implementada y cubierta por pruebas, pero
queda pendiente su validación completa sobre una GPU física con Ubuntu nativo.
La instalación CPU se comprueba automáticamente en Ubuntu 22.04 y 24.04. AMD
nativo continúa marcado como experimental hasta probarlo sobre hardware real.

## Cambio de perfil

Antes de cambiar de acelerador hay que detener el entrenamiento y cerrar
visores, gráficas o diagnósticos que sigan utilizando el entorno:

```bash
./scripts/parar_sim2real.sh
./scripts/install.sh --accelerator cpu --skip-system-packages
./scripts/doctor.sh
```

El perfil nuevo se guarda en su propia carpeta. Cambiar a CPU no borra
`.venvs/nvidia-cuda12` ni `.venvs/amd-rocm70`. El instalador utiliza un bloqueo
exclusivo para evitar que un entorno se modifique mientras otro proceso lo está
usando.

Si un cierre brusco interrumpe la instalación, basta con repetir el comando. La
transacción pendiente se recupera antes de volver a empezar.

## Problemas frecuentes

### El repositorio está dentro de `/mnt/c`

Vuelve a clonarlo bajo `~/robotica`. El instalador no mueve la carpeta por su
cuenta para no decidir dónde deben guardarse los datos.

### `nvidia-smi` no funciona en WSL2

Actualiza el controlador NVIDIA de Windows, ejecuta `wsl --shutdown` desde
PowerShell y abre Ubuntu otra vez. No instales un controlador NVIDIA Linux
dentro de WSL.

### VirtualBox muestra una tarjeta gráfica y NVIDIA falla

VMSVGA y VBoxSVGA son adaptadores virtuales. No proporcionan CUDA. Utiliza CPU
para probar la instalación o ejecuta Sim2Real MJX-JAX en WSL2 o Ubuntu nativo con la
GPU visible.

### ROCm no puede acceder a `/dev/kfd`

Revisa los grupos `render` y `video`, cierra la sesión y sigue la instalación
oficial de ROCm para el modelo exacto de GPU.

### `uv sync --frozen` indica que el bloqueo está desactualizado

No edites versiones de forma aislada. `pyproject.toml` y `uv.lock` deben
actualizarse juntos y pasar las pruebas del repositorio.

### El visor no abre desde WSL2

Comprueba WSLg con `echo $DISPLAY` y `echo $WAYLAND_DISPLAY`. El entrenamiento
sin ventana utiliza EGL y el visor utiliza GLFW. Ejecuta primero
`visualizar-modelo-preentrenado --solo-comprobar` para separar un problema de carga de la red de un
problema gráfico.

## Fuentes de plataforma

- [Controladores NVIDIA en Ubuntu](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/)
- [Secure Boot y gestión de MOK](https://documentation.ubuntu.com/security/docs/security-features/platform-protections/secure-boot/)
- [Prueba de Ubuntu mediante Live USB](https://documentation.ubuntu.com/desktop/en/24.04/tutorial/try-ubuntu-desktop/)
- [Instalación de Ubuntu y selección del disco](https://documentation.ubuntu.com/desktop/en/24.04/tutorial/install-ubuntu-desktop/)
- [Instalación de WSL](https://learn.microsoft.com/windows/wsl/install)
- [CUDA en WSL2](https://learn.microsoft.com/es-es/windows/ai/directml/gpu-cuda-in-wsl)
- [Manual de VirtualBox 7.2](https://docs.oracle.com/en/virtualization/virtualbox/7.2/user/EN-VBOX-7-2-USER.pdf)
- [Instalación de JAX](https://docs.jax.dev/en/latest/installation.html)
- [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html)
- [Compatibilidad de ROCm 7.0.2](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.0.2/reference/system-requirements.html)
- [Instalación nativa de ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.0.2/install/install-methods/package-manager/package-manager-ubuntu.html)
- [ROCm JAX v0.6.0](https://github.com/ROCm/rocm-jax/releases/tag/rocm-jax-v0.6.0)
- [Extensión de Intel para OpenXLA](https://github.com/intel/intel-extension-for-openxla)
