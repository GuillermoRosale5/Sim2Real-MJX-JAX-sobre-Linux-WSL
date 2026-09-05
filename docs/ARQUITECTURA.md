# Arquitectura y datos locales

Sim2Real MJX-JAX mantiene un único código científico para todos los equipos. El
modelo, el entorno MJX, la recompensa y PPO no cambian según el fabricante de la
GPU. La diferencia queda aislada en el entorno que proporciona JAX.

```text
Código Sim2Real MJX-JAX · XML + entorno + red + PPO
                         │
                      MJX/JAX
                         │
          ┌──────────────┼──────────────┐
          │              │              │
 NVIDIA / CUDA 12   AMD / ROCm 7.0      CPU
 .venvs/            .venvs/             .venvs/
 nvidia-cuda12      amd-rocm70           cpu
```

Esta separación evita mezclar en el mismo Python complementos y variantes de
`jaxlib` incompatibles. La GPU Intel no tiene un entorno propio en esta versión;
cuando se utiliza el modo `compatible`, el sistema continúa mediante CPU y lo
indica claramente.

## Organización del repositorio

```text
Sim2Real-MJX-JAX-sobre-Linux-WSL/
├── pyproject.toml          versiones y dependencias científicas
├── uv.lock                 resolución reproducible de dependencias
├── requirements/           dependencias adicionales de cada fabricante
├── sim2real_mjx/             capa de simulación y aprendizaje
│   ├── entorno_robot_mjx.py     entorno propio sobre MjxEnv
│   ├── curriculo_recompensas.py   currículo propio de tres fases
│   ├── hiperparametros.py          perfiles PPO propios
│   ├── entrenar_ppo_mjx.py         entrenamiento sobre Brax PPO
│   └── xmls/                       modelos físicos del robot
├── modelo_preentrenado/             red de referencia y manifiesto SHA-256
├── scripts/
│   ├── bootstrap_ubuntu.sh instalación pública para Ubuntu y WSL2
│   ├── install.sh          creación o reparación de un perfil
│   ├── doctor.sh           diagnóstico del equipo y del entorno
│   ├── sim2real.sh       entrada principal real
│   ├── install_windows.ps1 instalación desde PowerShell y WSL2
│   └── lib/platform.sh     detección del sistema y del acelerador
├── .venvs/                 entornos locales aislados
│   ├── nvidia-cuda12/
│   ├── amd-rocm70/
│   └── cpu/
├── .sim2real/            perfil activo y bloqueo del entorno
├── logs_sim2real_mjx/    resultados y puntos de control locales
├── docs/                   documentación
└── tests/                  pruebas que no necesitan una GPU
```

`.venvs/`, `.sim2real/` y `logs_sim2real_mjx/` son datos locales y Git no
los publica. La excepción es `modelo_preentrenado/`, donde se conserva la red de
referencia de la fase 2 en el paso 45.932.544.

La capa propia de aprendizaje de Sim2Real MJX-JAX utiliza nombres en español. Las
carpetas y APIs técnicas que pertenecen al ecosistema de Python, MuJoCo, MJX,
JAX, Brax u Orbax mantienen su forma original para que el código se pueda
comparar con las fuentes. Cuando un nombre propio del proyecto cambia, se
actualizan sus llamadas y se elimina el anterior; no se conserva otra entrada
que haga exactamente lo mismo.

La red congelada conserva en `model.json` los identificadores con los que fue
entrenada, por ejemplo `source_run: EntornoRobotMJX-...`. Son datos de
procedencia incluidos en el manifiesto del modelo, no módulos ejecutables ni
accesos alternativos al código actual.

## Selección del perfil

`scripts/lib/platform.sh` detecta el sistema, resuelve el perfil y configura JAX
antes de iniciar Python.

```text
--accelerator nvidia     -> .venvs/nvidia-cuda12 -> gpu
--accelerator amd        -> .venvs/amd-rocm70    -> gpu
--accelerator cpu        -> .venvs/cpu           -> cpu
--accelerator auto       -> una GPU admitida o un error
--accelerator compatible -> nvidia, amd o cpu con aviso si utiliza CPU
```

`compatible` es únicamente una opción de instalación. Se convierte en un perfil
concreto antes de crear el entorno, de modo que nunca se guarda el texto
`compatible` ni se crea una carpeta `.venvs/compatible`.

El modo `auto` tiene una función distinta: exige una GPU admitida. Durante la
instalación vuelve a comprobar NVIDIA y AMD; si las dos están disponibles al
mismo tiempo, obliga a seleccionar una. En los comandos normales puede
reutilizar el último perfil validado para que el dispositivo no cambie entre
ejecuciones.

El perfil activo solo se guarda en `.sim2real/accelerator` después de superar
el diagnóstico, un cálculo JIT y un paso físico de MJX. Cambiar de fabricante no
borra los demás entornos.

## Protección del entorno

`.sim2real/runtime.lock` evita que una instalación modifique el entorno
mientras está siendo utilizado. La simulación, el entrenamiento, el visor, las
gráficas y el diagnóstico toman un bloqueo compartido. `install.sh` necesita un
bloqueo exclusivo y se detiene antes de cambiar archivos si todavía queda otro
proceso activo.

Cada perfil se reconstruye dentro de una transacción. Si la instalación falla,
se restaura la copia anterior y se conserva el perfil que ya funcionaba. Una
ejecución posterior también recupera una transacción interrumpida por el cierre
de WSL o del equipo.

El diagnóstico comprueba cuatro capas:

1. sistema operativo y tipo de instalación;
2. hardware y controlador del fabricante;
3. paquetes del entorno aislado;
4. dispositivo enumerado por JAX, cálculo JIT y paso físico de MJX.

Una importación correcta de JAX no basta. Un perfil NVIDIA o AMD falla si JAX
termina ejecutando sobre CPU.

## Responsabilidad de cada capa

`bootstrap_ubuntu.sh` prepara la instalación desde cero: comprueba la plataforma,
instala herramientas básicas, sincroniza el repositorio y llama a `install.sh`.

- En Ubuntu nativo puede preparar el controlador NVIDIA oficial mediante
  `ubuntu-drivers` y detenerse para completar un reinicio.
- En WSL2 utiliza el controlador NVIDIA de Windows y nunca instala un
  controlador Linux dentro de la distribución.
- AMD necesita Ubuntu 24.04 nativo, hardware incluido en la matriz oficial y
  ROCm 7.0.0–7.0.2 ya operativo.
- AMD sobre WSL2 y la GPU Intel no están admitidos en esta versión.
- CPU se puede forzar o puede ser elegida por el modo `compatible`.

`install.sh` administra Python y las dependencias del proyecto. No puede crear
acceso a una GPU que el sistema operativo, el hipervisor o el controlador no
expongan.

## Diferencias entre Ubuntu nativo y WSL2

Los dos sistemas ejecutan el mismo código, pero no validan las mismas capas.
WSL2 comprueba la integración con el controlador de Windows. Ubuntu nativo
utiliza su propio kernel, controlador, arranque, Secure Boot y MOK.

Una instalación completa en un SSD externo cuenta como Ubuntu nativo. Un Live
USB solo proporciona una comprobación preliminar. VirtualBox 7.2 no expone la
GPU PCI como CUDA o ROCm, por lo que allí `compatible` selecciona CPU. Los
equipos de integración continua también validan el software sobre CPU, no una
GPU física.

## Resultados y red de referencia

Cada entrenamiento crea una subcarpeta dentro de `logs_sim2real_mjx/`. Allí
quedan el comando reproducible, la configuración, el PID, el estado, los CSV,
los registros y los puntos de control. `ultima_ejecucion.txt` señala la ejecución
activa o la más reciente de esa copia del repositorio.

Las dos rutas de visualización se mantienen separadas:

- `visualizar-modelo-preentrenado` verifica `SHA256SUMS` y carga siempre
  `modelo_preentrenado/modelo_referencia_fase2_45932544/checkpoints/000045932544`;
- `visualizar-resultados` y `visualizar_ultimo_checkpoint.sh` cargan resultados locales,
  que pueden estar incompletos.

Antes de enviar una señal a un PID, los scripts comprueban que el proceso
pertenece al entrenador o a la protección térmica de esa ejecución. Esta
protección no cambia con el acelerador.
