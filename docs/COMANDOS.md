# Comandos disponibles

La instalación completa desde cero se encuentra en
[INSTALACION.md](INSTALACION.md). Este documento reúne los comandos de trabajo
una vez que Sim2Real MJX-JAX ya está instalado.

Todos los perfiles utilizan los mismos scripts. El entorno activo cambia según
el acelerador:

```text
nvidia -> .venvs/nvidia-cuda12
amd    -> .venvs/amd-rocm70
cpu    -> .venvs/cpu
```

## Instalación y diagnóstico

La selección automática recomendada es:

```bash
./scripts/install.sh --accelerator compatible
```

También se puede exigir un perfil concreto:

```bash
./scripts/install.sh --accelerator nvidia
./scripts/install.sh --accelerator amd
./scripts/install.sh --accelerator cpu
```

`nvidia` admite Ubuntu nativo y WSL2. `amd` solo admite Ubuntu 24.04 nativo con
ROCm 7.0.0–7.0.2 ya preparado. En AMD sobre WSL2 y en equipos Intel, el modo
`compatible` utiliza CPU y muestra un aviso.

La instalación se comprueba con:

```bash
./scripts/doctor.sh
./scripts/sim2real.sh test-mjx --steps 150
./scripts/sim2real.sh visualizar-modelo-preentrenado --solo-comprobar
```

La primera orden revisa el sistema, las versiones y el dispositivo que JAX ve
realmente. La segunda ejecuta una prueba física corta. La tercera valida la red
incluida sin abrir el visor.

En `doctor.sh`, `auto` significa «revisar el perfil activo que dejó la
instalación». `compatible` solo se utiliza al instalar y por eso no es una
opción de `doctor`.

Para revisar únicamente la plataforma, sin inicializar JAX ni MuJoCo:

```bash
./scripts/doctor.sh --quick
```

Las comprobaciones estáticas del repositorio se ejecutan con:

```bash
./scripts/test_static.sh
```

## Inicio del entrenamiento

El menú preparado para el uso habitual se abre con:

```bash
./scripts/lanzar_sim2real.sh
```

Permite elegir el perfil PPO y la fase inicial. El perfil recomendado es
`ligero`. También están disponibles `depuracion`, `ligero_rapido` y `completo`.

Para lanzar el entrenamiento sin pasar por el menú:

```bash
./scripts/sim2real.sh entrenar --segundo-plano --setup \
  --perfil-ppo ligero \
  --fase-recompensa 1
```

`--segundo-plano` deja el proceso en segundo plano. `--setup` comprueba primero la
instalación y MJX. Las fases de recompensa son:

- `0`: configuración base histórica;
- `1`: mantener la postura indicada por el XML;
- `2`: alcanzar la postura desde el suelo;
- `3`: recuperarse desde una caída.

El currículo automático de tres fases se inicia con:

```bash
./scripts/curriculo_automatico_sim2real.sh
```

La fase de una ejecución curricular se puede cambiar de forma manual con:

```bash
./scripts/cambiar_fase_sim2real.sh 2
```

Los archivos de estado quedan dentro de la propia ejecución. Cambiar de fase no
modifica el código ni la configuración global.

## Continuación y reinicio

Para continuar desde el último punto de control local:

```bash
./scripts/sim2real.sh entrenar --segundo-plano --continuar-ultimo \
  --perfil-ppo ligero \
  --fase-recompensa 1
```

Hay que volver a indicar el perfil y la fase; `--continuar-ultimo` no los recupera
por sí solo.

Para descartar únicamente los puntos de control de la ejecución seleccionada:

```bash
./scripts/sim2real.sh reiniciar-checkpoint
```

No debe combinarse `--desde-cero` con una restauración en el mismo
comando.

## Seguimiento y parada

El estado del entrenamiento se muestra en directo con:

```bash
./scripts/monitor_sim2real.sh
```

La pantalla incluye pasos, velocidad, recompensa, uso de GPU y memoria,
temperatura y puntos de control guardados.

El proceso se detiene de forma ordenada con:

```bash
./scripts/parar_sim2real.sh
```

Los resultados ya creados se conservan. La protección térmica de la ejecución
activa puede iniciarse con:

```bash
./scripts/sim2real.sh iniciar-proteccion-termica
```

En CPU se omite porque no existe un sensor GPU que vigilar.

## Visualización

El acceso recomendado a la red incluida es:

```bash
./scripts/visualizar_modelo_preentrenado.sh
```

Para comprobar su carga sin abrir la ventana:

```bash
./scripts/sim2real.sh visualizar-modelo-preentrenado --solo-comprobar
```

Carga siempre la red versionada de fase 2, semilla 42 y paso 45.932.544. Su
metadato conserva el nombre histórico `lite`; en el código actual esa misma
configuración se llama `ligero`. Antes de abrirla comprueba el manifiesto
SHA-256. No utiliza el último entrenamiento local.

El menú con todas las opciones se abre mediante:

```bash
./scripts/visualizar_sim2real.sh
```

Desde ahí se puede elegir la red incluida, el último punto de control local, uno
anterior o una vista estática de los XML.

Los accesos directos a resultados locales son:

```bash
./scripts/visualizar_ultimo_checkpoint.sh --longitud-episodio 1500
./scripts/minisimular_ultimo_checkpoint.sh
```

La interfaz propia del visualizador utiliza nombres en español:
`--ruta-checkpoint`, `--indice-checkpoint`, `--longitud-episodio`, `--semilla`,
`--ruta-xml`, `--postura-inicial`, `--reinicio-automatico`,
`--congelar-al-terminar` y `--solo-comprobar`. `--impl` conserva el nombre de
la interfaz técnica MJX.

El último punto local puede pertenecer a otra fase o a una ejecución detenida
antes de tiempo. No debe confundirse con la red de referencia incluida en el
repositorio.

## Gráficas de recompensa

Para generar las gráficas guardadas:

```bash
./scripts/graficar_recompensas.sh
```

Para limitar la gráfica a un CSV concreto o abrirla al terminar:

```bash
./scripts/graficar_recompensas.sh \
  --ruta-csv logs_sim2real_mjx/MI_EJECUCION/recompensas.csv \
  --mostrar
```

Para verlas mientras el entrenamiento continúa:

```bash
./scripts/ver_recompensas_en_directo.sh
```

Las opciones propias de estas herramientas están en español. Por ejemplo,
`--directorio-ejecucion`, `--modo-csv activo`, `--origen evaluacion`,
`--segmento ultimo`, `--unidades-recompensa escaladas`, `--intervalo 5` y
`--sin-ventana --salida recompensas.png`. Las claves seleccionadas con
`--eje-x` o `--columnas` mantienen el nombre exacto del CSV histórico.

Los dos comandos utilizan el entorno activo y buscan los datos bajo
`logs_sim2real_mjx/`.

## Prueba de rendimiento

La prueba MJX corta sirve para comprobar una instalación:

```bash
./scripts/sim2real.sh test-mjx --steps 150
```

La prueba amplia de rendimiento es una operación avanzada:

```bash
./scripts/sim2real.sh benchmark
```

Recorre varias combinaciones y puede tardar bastante. En CPU conviene reducir
manualmente `--envs`, `--warmup-steps` y `--measure-steps`.

## Nombres de la capa propia y de las dependencias

Cada operación tiene un solo archivo real. No hay copias ni archivos puente con
el nombre utilizado por una versión anterior.

Los elementos escritos específicamente para el aprendizaje de Sim2Real MJX-JAX usan
nombres en español: currículo, recompensas, hiperparámetros, entrenamiento y
herramientas de visualización. Las interfaces procedentes de MuJoCo, MJX, JAX,
Brax y Orbax conservan su forma original. También se mantienen las convenciones
habituales de un proyecto Python, como `README.md`, `pyproject.toml`, `tests/` o
el método `step` de un entorno MJX. Así se puede comparar esta implementación
con sus fuentes sin inventar una traducción distinta para su API.

La instalación común para Ubuntu nativo y WSL2 es `install.sh`; el propio
script detecta en cuál de los dos sistemas está trabajando. Desde PowerShell la
entrada completa es `install_windows.ps1`. El lanzador general es
`sim2real.sh` y contiene la implementación real, no una llamada a otro
lanzador con un nombre antiguo.

## Variables de entorno

| Variable | Uso |
|---|---|
| `SIM2REAL_ACCELERATOR` | Selecciona `auto`, `nvidia`, `amd` o `cpu` en comandos posteriores. `intel` solo se reconoce para detenerse con un diagnóstico claro; no activa una GPU Intel. |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | Limita la fracción de memoria; el entrenamiento utiliza 0.70 por defecto. |
| `SIM2REAL_THERMAL_GUARD=0` | Desactiva la protección térmica para esa ejecución. |
| `SIM2REAL_GPU_TEMP_WARN_C` | Temperatura de aviso; 78 °C por defecto. |
| `SIM2REAL_GPU_TEMP_STOP_C` | Umbral de parada sostenida; 84 °C por defecto. |
| `MUJOCO_VIEWER_GL` | Sistema gráfico del visor; `glfw` por defecto. |
| `SIM2REAL_REPO_URL` | Repositorio alternativo para el instalador de PowerShell. |
