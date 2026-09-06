# Aceleradores GPU y compatibilidad

Esta sección explica cuándo y qué parte de Sim2Real MJX-JAX se ejecuta en CPU, 
qué parte puede ejecutarse en GPU y qué combinaciones del sistema operativo que
se tenga como base y su tarjeta tarjeta gráfica admite realmente esta versión.

## La decisión técnica del proyecto

MuJoCo es un motor de simulación física ampliamente utilizado por la comunidad de
Reinforcement Learning en robótica. La aceleración de entrenamiento de aprendizaje
de redes neuronales para R-L emplean estándares sobre CUDA empleando librerías como
PyTorch. Mujoco clásico es compatible con PyTorch, y este es compatible con GPU para
entrenamiento masivo de sus redes neuronales y el algoritmo de calibración de pesos
pero requiere centralizar toda la simulación física en su CPU.

Por ese motivo en este proyecto hemos elegido una alternativa que ofrece mujoco que 
nos permite no depender de la CPU para el cálculo de la física. Se trata de MuJoCo XLA
basado en JAX y compatible con el mismo algoritmo de actualización de pesos PPO que
al mismo tiempo se basa en BRAX, una variante de JAX. JAX y Brax cubren otras capas
del sistema y no deben presentarse como si fueran la misma herramienta.

Como recomendación del autor para otros investigadores que deseen explorar alternativas
a esta, PyTorch sí puede entrenar redes neuronales en GPU. Cuenta con rutas para
NVIDIA/CUDA, AMD/ROCm, Intel/XPU y Apple/MPS, cada una con sus propios límites.
Por tanto, Sim2Real MJX-JAX no utiliza JAX porque PyTorch sea incapaz de trabajar en
GPU.

Con MuJoCo clásico, las llamadas a `mj_step` y `mujoco.rollout` calculan la física en 
CPU. Podemos colocar la red y el PPO de PyTorch en una GPU, pero la simulación y el 
aprendizaje quedan separados entre CPU y GPU. La propia documentación de MuJoCo advierte 
de que las transferencias entre ambos dispositivos pueden convertirse en un cuello de botella.

MJX-JAX expresa la dinámica de MuJoCo mediante JAX/XLA. De esta forma,
el paso físico `mjx.step`, los lotes de entornos y el PPO de Brax pueden
permanecer dentro del mismo ecosistema JAX y sobre el mismo acelerador. JAX
permite compilar y vectorizar el cálculo con `jit`, `vmap` y `pmap`, que es justo
lo que necesita un entrenamiento con miles de entornos en paralelo.

Esta ventaja aparece cuando trabajamos con lotes grandes. MuJoCo señala que
MJX-JAX está especializado en miles o decenas de miles de escenas iguales en
paralelo. Para una sola escena puede ser bastante más lento que MuJoCo clásico,
que está muy optimizado para reducir la latencia en CPU.

## Comparación de las tres rutas principales

| Configuración | Simulación física | Red y PPO | Consecuencia práctica |
|---|---|---|---|
| MuJoCo clásico + PyTorch | CPU | CPU o una GPU admitida por PyTorch | Es una combinación válida, pero poner la red en GPU no traslada la física y puede exigir transferencias CPU–GPU. |
| MJX-JAX + PPO de Brax | CPU, GPU o TPU admitida por JAX | El mismo entorno JAX | Permite mantener física y aprendizaje sobre el acelerador y vectorizar conjuntamente muchos entornos. Es la ruta utilizada por Sim2Real MJX-JAX. |
| MuJoCo Warp + PyTorch o RSL-RL | GPU NVIDIA | GPU NVIDIA | Ofrece un rendimiento muy alto y escala bien en escenas complejas, pero está diseñado específicamente para NVIDIA. |

MuJoCo Warp es una alternativa muy interesante cuando el único objetivo es
obtener el máximo rendimiento sobre NVIDIA. No sustituye el objetivo de este
TFG porque renuncia a la adaptación entre fabricantes. Por ahora Sim2Real MJX-JAX
mantiene `impl=jax`; `--impl warp` no forma parte de los perfiles admitidos.

## JAX no convierte cualquier gráfica en compatible

JAX proporciona una API común para trabajar con matrices y transformar
funciones, pero necesita una implementación compilada para cada plataforma. El
paquete Python `jax` no basta por sí solo: `jaxlib` y los complementos PJRT
conectan el código con CUDA, ROCm, oneAPI u otro entorno de ejecución.

Esto permite conservar casi todo el código científico al cambiar de fabricante,
pero no elimina sus diferencias. Cada ruta conserva sus propios controladores,
bibliotecas, versiones, sistemas operativos y listas de tarjetas admitidas. No
existe un único binario de JAX que pueda ejecutarse indistintamente en cualquier
GPU.

Para este proyecto, MJX-JAX sigue siendo la opción más adaptable dentro del
ecosistema MuJoCo porque permite llevar tanto la física como el PPO al
acelerador y contempla más de un fabricante. Eso no significa que sea una
solución universal ni que todas sus plataformas tengan el mismo grado de
madurez.

## Compatibilidad publicada por los proyectos originales

La tabla siguiente resume la documentación oficial actual de JAX y de los
fabricantes. Sirve como contexto, pero no sustituye la tabla de validación de
Sim2Real MJX-JAX que aparece después.

| Plataforma | NVIDIA | AMD | Intel | CPU |
|---|---|---|---|---|
| Linux `x86_64` nativo | Admitida por JAX mediante CUDA | Admitida mediante el complemento ROCm y solo para hardware incluido por AMD | Complemento experimental de Intel | Admitida |
| WSL2 `x86_64` | JAX la clasifica como experimental; CUDA para WSL2 está soportado por NVIDIA | JAX la clasifica como experimental, pero AMD indica que JAX todavía no está habilitado ni validado bajo WSL2 | No admitida por JAX | Admitida |
| Windows nativo `x86_64` | No admitida por JAX | No admitida por JAX | No admitida por JAX | Disponible; queda fuera del sistema Linux/WSL de este repositorio |

El soporte AMD depende además de la matriz de ROCm. Que una tarjeta sea Radeon
no implica que ROCm la incluya. El soporte Intel de JAX se publica como
experimental y utiliza un complemento, unas versiones y una lista de hardware
propias.

## Estados utilizados por Sim2Real MJX-JAX

Para no confundir una ruta programada con una ruta comprobada, usamos cuatro
estados distintos:

- **Validada:** la instalación, JAX, MJX y una ejecución real se han probado de
  principio a fin en hardware del proyecto.
- **Implementada:** existen instalador, perfil aislado, diagnósticos y pruebas
  automáticas, pero falta una ejecución completa sobre la GPU física indicada.
- **Experimental:** la ruta está implementada, pero depende de una pila menos
  madura o todavía no ha superado la validación física del proyecto.
- **No admitida:** esta versión la rechaza para evitar una instalación que
  parezca correcta pero termine usando CPU o una combinación incompatible.

Un resultado positivo en otro equipo demuestra que esa ejecución concreta
funciona. No permite afirmar que Sim2Real MJX-JAX haya validado todas las tarjetas de
ese fabricante.

## Estado real de esta versión

| Sistema y acelerador | Estado en Sim2Real MJX-JAX | Alcance real |
|---|---|---|
| WSL2 con Ubuntu 22.04/24.04 y NVIDIA | **Validada en WSL2 con Ubuntu 24.04** | La ruta se completó de principio a fin en el equipo NVIDIA del proyecto. Ubuntu 22.04 está admitido y cubierto por las comprobaciones, pero no se presenta como una segunda validación física. |
| Ubuntu 22.04/24.04 nativo y NVIDIA | **Implementada** | Usa la misma pila CUDA fijada y cuenta con diagnósticos automáticos. Falta probarla de principio a fin en una GPU física con Ubuntu nativo. |
| Ubuntu 24.04 nativo y AMD | **Experimental e implementada** | Exige una GPU incluida por AMD y ROCm 7.0.0–7.0.2 ya operativo. Falta la validación sobre una GPU AMD física del proyecto. |
| WSL2 y AMD | **No admitida con GPU** | AMD indica que JAX todavía no está habilitado ni validado bajo WSL2. El modo `compatible` utiliza CPU y lo avisa. |
| Ubuntu nativo o WSL2 e Intel | **No admitida con GPU** | La combinación Intel/JAX disponible no coincide con las versiones y plataformas fijadas por Sim2Real MJX-JAX. El modo `compatible` utiliza CPU y lo avisa. |
| Ubuntu 22.04/24.04 nativo o WSL2 y CPU | **Implementada y cubierta por integración continua** | Sirve para desarrollo y comprobaciones pequeñas. No es una ruta realista para un entrenamiento PPO masivo. |

Esta tabla es deliberadamente más conservadora que la tabla oficial de JAX.
La primera describe lo que las herramientas originales permiten en determinadas
condiciones; esta describe lo que podemos defender con el código y las pruebas
de Sim2Real MJX-JAX.

## Selección automática con `compatible`

La instalación sencilla utiliza:

```bash
./scripts/install.sh --accelerator compatible
```

`compatible` adapta la instalación al equipo sin obligarnos a cambiar el
comando:

```text
NVIDIA operativa                           -> perfil NVIDIA
AMD/ROCm admitida en Ubuntu 24.04 nativo  -> perfil AMD
ninguna GPU utilizable por Sim2Real MJX-JAX     -> perfil CPU con un aviso visible
```

Si NVIDIA y AMD están operativas al mismo tiempo, `compatible` prioriza NVIDIA
porque es la ruta que más se ha validado en este proyecto. El instalador lo
indica y permite forzar AMD de forma explícita.

`compatible` solo decide durante la instalación. No se crea un entorno llamado
`.venvs/compatible`: el valor se resuelve a `nvidia`, `amd` o `cpu`, y ese perfil
concreto es el que se comprueba y se guarda.

En una instalación nueva de Ubuntu nativo, el autoinstalador también puede
reconocer una NVIDIA por PCI aunque el controlador todavía no esté cargado. En
ese caso prepara el controlador, solicita el reinicio cuando sea necesario y
continúa al repetir el mismo comando.

## Selección estricta con `auto`

El modo `auto` tiene otro propósito:

```bash
./scripts/install.sh --accelerator auto
```

Significa «esta ejecución debe utilizar una GPU compatible». Nunca sustituye la
GPU por CPU:

```text
una GPU y su entorno están preparados      -> selecciona ese perfil
NVIDIA y AMD están preparadas a la vez     -> se detiene y pide elegir
la GPU aparece pero su entorno está roto   -> se detiene con un diagnóstico
no existe una GPU utilizable               -> se detiene
```

Este comportamiento resulta útil en una máquina de entrenamiento o en una
prueba automatizada donde continuar en CPU ocultaría un problema y podría dejar
un proceso extremadamente lento trabajando durante horas.

## Selección explícita del perfil

También podemos indicar la ruta que queremos comprobar:

```bash
./scripts/install.sh --accelerator nvidia
./scripts/install.sh --accelerator amd
./scripts/install.sh --accelerator cpu
```

Pedir un fabricante de forma explícita no autoriza una sustitución. Si
solicitamos NVIDIA en un equipo Intel, o AMD bajo WSL2, el instalador se detiene
y explica la incompatibilidad.

Los entornos quedan separados para impedir que CUDA y ROCm mezclen versiones de
`jaxlib`:

| Perfil final | Entorno aislado | Plataforma esperada en JAX |
|---|---|---|
| `nvidia` | `.venvs/nvidia-cuda12` | `gpu` mediante CUDA 12 |
| `amd` | `.venvs/amd-rocm70` | `gpu` mediante ROCm 7.0 |
| `cpu` | `.venvs/cpu` | `cpu` |
| `intel` | No se crea | Perfil bloqueado en esta versión |

El perfil activo se guarda en `.sim2real/accelerator`. Los lanzadores
reutilizan después el entorno que ya superó las comprobaciones. No debemos
copiar `.venvs/` entre ordenadores; el código se clona y cada equipo reconstruye
su entorno.

## Versiones comunes del proyecto

El código científico se mantiene sobre un conjunto de versiones fijo:

| Componente | Versión |
|---|---:|
| Python | 3.12.x |
| JAX / JAXLIB | 0.6.2 |
| MuJoCo / MuJoCo MJX | 3.6.0 |
| Brax | 0.14.2 |
| Flax | 0.11.2 |
| MuJoCo Playground | commit `9c2dce4a3519cd4bb9d299bf28a6ef3f5086844b` |

El sistema admitido es Ubuntu 22.04 o 24.04 `x86_64`, nativo o dentro de WSL2
cuando el perfil lo permita. ARM, WSL1, otras distribuciones y versiones
intermedias quedan fuera del contrato de esta versión.

## NVIDIA en Ubuntu nativo

El perfil NVIDIA instala `jax[cuda12]==0.6.2` dentro de
`.venvs/nvidia-cuda12`. Las bibliotecas CUDA y cuDNN que necesita JAX llegan en
el entorno Python; Sim2Real MJX-JAX no necesita instalar un CUDA Toolkit global.

El autoinstalador puede preparar el controlador oficial mediante las
herramientas de Ubuntu. Algunos equipos requieren reiniciar y completar la
inscripción de la clave MOK si Secure Boot está activo. Después del reinicio,
esta orden debe mostrar la tarjeta sin errores:

```bash
nvidia-smi
```

El proyecto exige una serie de controlador compatible con CUDA 12; esta versión
comprueba como mínimo la serie 525. Que el instalador pueda preparar el
controlador no convierte todavía esta ruta en validada: falta la prueba completa
sobre una máquina Ubuntu nativa con GPU NVIDIA física.

## NVIDIA dentro de WSL2

En WSL2, el controlador de la GPU pertenece a Windows. Dentro de Ubuntu no se
deben instalar `ubuntu-drivers`, DKMS ni un controlador NVIDIA para Linux. El
controlador de Windows publica la interfaz CUDA que utiliza WSL2.

Antes de instalar Sim2Real MJX-JAX comprobamos desde Ubuntu:

```bash
nvidia-smi
```

Si falla, actualizamos el controlador NVIDIA en Windows, ejecutamos desde
PowerShell:

```powershell
wsl --shutdown
```

y volvemos a abrir Ubuntu. La documentación de JAX clasifica esta plataforma
como experimental, pero la ruta concreta de Sim2Real MJX-JAX sí se ha validado de
extremo a extremo con Ubuntu 24.04 bajo WSL2 en el equipo del proyecto.

## AMD en Ubuntu nativo

El perfil AMD solo se admite en Ubuntu 24.04 nativo `x86_64`. Esta versión fija
JAX 0.6.2 y los complementos de ROCm preparados para ROCm 7.0.0, 7.0.1 o 7.0.2.
No debemos sustituirlos por la versión más reciente sin volver a resolver y
validar todas las dependencias.

Antes de ejecutar el instalador deben cumplirse todas estas condiciones:

- la GPU figura en la
  [matriz oficial de ROCm 7.0.2](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.0.2/reference/system-requirements.html);
- la combinación de Ubuntu y núcleo Linux figura en esa misma matriz;
- ROCm 7.0.0–7.0.2 ya está instalado y operativo;
- `/dev/kfd` existe y el usuario puede leerlo y escribir en él;
- `rocminfo` enumera al menos un agente gráfico `gfx`.

La comprobación manual mínima es:

```bash
test -r /dev/kfd && test -w /dev/kfd
rocminfo
```

El autoinstalador no instala ni sustituye a ciegas `amdgpu`, DKMS o ROCm. Esa
operación depende del modelo exacto de GPU, del núcleo y de la tabla publicada
por AMD. El instalador sí valida el entorno ya preparado y crea
`.venvs/amd-rocm70` sin tocar el perfil NVIDIA.

Esta ruta sigue marcada como experimental porque todavía no se ha ejecutado de
principio a fin sobre una GPU AMD física del proyecto.

## AMD dentro de WSL2

ROCm dispone de componentes para WSL2, pero eso no implica que todos sus marcos
de cálculo estén admitidos. La documentación actual de AMD indica expresamente
que su versión de JAX todavía no está habilitada ni validada bajo WSL2 y que las
cargas JAX pueden fallar durante la instalación, la inicialización o la
ejecución.

Por esta razón, Sim2Real MJX-JAX rechaza `--accelerator amd` dentro de WSL2. El modo
`compatible` utiliza CPU y lo anuncia. `auto` se detiene porque ha sido diseñado
para exigir una GPU real. Esta restricción se podrá revisar cuando exista una
combinación oficial compatible con las versiones del proyecto y podamos
validarla sobre hardware físico.

## Intel

PyTorch ofrece actualmente una ruta XPU para varias GPU Intel. Eso no resuelve
automáticamente la simulación de Sim2Real MJX-JAX porque el proyecto no ejecuta la
física MJX mediante PyTorch.

JAX publica un complemento experimental de Intel para Linux, pero la
implementación considerada por esta versión trabaja con un conjunto diferente
de versiones de JAX/JAXLIB, oneAPI, sistemas y modelos de GPU. La documentación
de JAX tampoco admite Intel GPU bajo WSL2. Mezclar ese complemento con JAX 0.6.2,
Flax, Brax y MJX sin una nueva validación rompería el entorno reproducible.

Por ello no afirmamos que JAX sea incapaz de funcionar sobre cualquier Intel.
La afirmación correcta es que **Sim2Real MJX-JAX no implementa ni valida actualmente
un perfil Intel GPU**. `--accelerator intel` se rechaza, `auto` se detiene y
`compatible` crea el perfil CPU con un aviso claro.

## CPU

El perfil CPU se puede solicitar directamente:

```bash
./scripts/install.sh --accelerator cpu
```

`.venvs/cpu` fija `JAX_PLATFORMS=cpu`. Permite cargar el XML, importar la red,
ejecutar las pruebas automáticas y comprobar pasos pequeños de MJX. No debe
presentarse como una alternativa para el entrenamiento PPO masivo que motiva el
proyecto.

CPU es también el resultado seguro del modo `compatible` cuando el sistema no
ofrece una GPU admitida. El instalador lo muestra expresamente y nunca llama
«GPU» a esa ejecución.

## Comprobación completa del entorno

Detectar el nombre de una tarjeta por PCI no demuestra que JAX pueda utilizarla.
La instalación solo se considera correcta después de comprobar toda la cadena:

1. El sistema es Ubuntu, WSL2 y arquitectura admitida para el perfil.
2. El controlador expone la GPU mediante `nvidia-smi` o `rocminfo`.
3. El entorno aislado contiene exactamente las versiones fijadas.
4. JAX carga la plataforma esperada y enumera el fabricante correcto.
5. Un cálculo compilado mediante JIT se ejecuta en el dispositivo.
6. MJX carga el XML y completa un paso físico real.

La orden pública para repetir estas comprobaciones es:

```bash
./scripts/doctor.sh
```

Para NVIDIA o AMD debe aparecer `JAX backend: gpu` junto con el dispositivo
correcto. Si un perfil GPU termina utilizando CPU, la comprobación falla; no se
acepta una sustitución silenciosa.

`doctor: OK` demuestra el contrato de software y el dispositivo visible durante
esa ejecución. No convierte automáticamente en validada por el proyecto una
combinación de hardware que todavía no hemos ensayado.

## Máquinas virtuales y pruebas sin particionar

La aceleración 3D de una máquina VirtualBox sirve para mostrar gráficos, pero su
adaptador virtual no se convierte en una tarjeta CUDA o ROCm para JAX. En una VM
sin acceso directo a una GPU admitida, `compatible` selecciona CPU, mientras que
`auto`, `nvidia` y `amd` se detienen.

Esa VM resulta útil para comprobar la instalación CPU, pero no valida Linux
nativo con GPU. Para una prueba nativa realista sin modificar las particiones
del disco interno podemos instalar Ubuntu completamente en un SSD externo y
arrancar el ordenador desde él. De ese modo se prueban el núcleo, el controlador
y el acceso físico a la GPU como en una instalación nativa.

## Fuentes oficiales

La explicación anterior se apoya en documentación de los proyectos que forman
esta arquitectura:

- [Descripción general de MuJoCo y sus motores acelerados](https://mujoco.readthedocs.io/en/stable/overview.html)
- [MuJoCo XLA y funcionamiento de MJX-JAX](https://mujoco.readthedocs.io/en/stable/mjx.html)
- [MuJoCo Warp y comparación de las rutas CPU/GPU](https://mujoco.readthedocs.io/en/latest/mjwarp/)
- [Plataformas y métodos de instalación admitidos por JAX](https://docs.jax.dev/en/latest/installation.html#supported-platforms)
- [Instalación de PyTorch con CUDA o ROCm](https://docs.pytorch.org/get-started/locally/)
- [Compatibilidad XPU de PyTorch con GPU Intel](https://docs.pytorch.org/docs/main/notes/get_start_xpu.html)
- [Compatibilidad MPS de PyTorch con GPU Apple](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [PPO y entrenamiento en JAX dentro de Brax](https://github.com/google/brax/blob/main/README.md)
- [Entrenador JAX/PPO oficial de MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground/blob/main/learning/train_jax_ppo.py)
- [CUDA sobre WSL2 según NVIDIA](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [Estado de JAX/ROCm bajo WSL2 según AMD](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html)
- [Complemento OpenXLA para GPU Intel](https://github.com/intel/intel-extension-for-openxla)

Las páginas generales reflejan el estado actual de cada proyecto. Las versiones
de JAX, CUDA y ROCm utilizadas por Sim2Real MJX-JAX permanecen fijadas en el
repositorio para que una actualización externa no cambie el experimento sin
control.
