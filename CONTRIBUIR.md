# Contribuir

Trabaja en una rama y no subas `.venvs/`, `.venv`, registros, puntos de control
ni `external/`. Los entornos `nvidia-cuda12`, `amd-rocm70` y `cpu` se
reconstruyen en cada equipo; nunca copies uno para fabricar otro perfil.
La única excepción actual es la red curada que ya vive en
`modelo_preentrenado/modelo_referencia_fase2_45932544/`: fase 2, paso
45.932.544. No
coloques allí el último punto de control local ni cambies esa referencia como parte
de un entrenamiento normal.

Si se sustituye deliberadamente la red preentrenada, el cambio debe incluir los
pesos Orbax, metadatos, configuración reproducible, documentación y un
`SHA256SUMS` regenerado. El comando `visualizar-modelo-preentrenado` debe seguir
siendo determinista y no depender de `logs_sim2real_mjx/ultima_ejecucion.txt`.

Antes de confirmar cambios ejecuta:

```bash
./scripts/test_static.sh
uv lock --check
./scripts/doctor.sh --quick
```

Si cambias una versión científica, actualiza en el mismo cambio
`pyproject.toml`, `uv.lock`, la tabla de `docs/ACELERADORES.md` y
`VERSIONES_ESPERADAS` en `sim2real_mjx/hiperparametros.py`. No cambies la matriz de
fabricantes sin una prueba real del dispositivo y una prueba corta de MJX.

Un cambio en perfiles debe conservar estas reglas:

- el código MJX/JAX es común y no contiene ramas científicas por fabricante;
- cada pila se instala en `.venvs/<perfil>` sin modificar las demás;
- `auto` selecciona una GPU admitida o falla; `compatible` puede seleccionar CPU
  con un aviso visible;
- un perfil GPU debe fallar si JAX devuelve CPU;
- AMD solo se ofrece en Ubuntu 24.04 nativo; AMD/WSL e Intel siguen bloqueados
  mientras no exista una combinación compatible y validada.

Si añades soporte para otro fabricante, incluye un entorno aislado, dependencias
reproducibles, detección que falle de forma segura, diagnóstico, pruebas y
documentación. Una prueba de importación no basta: registra también el
dispositivo JAX, una prueba corta de MJX y la carga de la red preentrenada sobre
hardware real.

Los scripts Bash deben funcionar con `set -euo pipefail` y obtener sus rutas a
partir de `BASH_SOURCE`. Cada operación tiene una sola entrada real: al cambiar
un nombre hay que actualizar sus llamadas, pruebas y documentación en el mismo
cambio, sin conservar un archivo puente con el nombre anterior.

La capa desarrollada para Sim2Real MJX-JAX se nombra en español: entorno, currículo,
recompensas, hiperparámetros, entrenamiento y herramientas propias. No se
traducen los contratos que llegan de Python, MuJoCo, MJX, JAX, Brax u Orbax,
como `reset`, `step`, `jit`, `vmap` o la estructura interna de un punto de
control. Esta frontera permite reconocer el código tomado como referencia y
separa con claridad la aportación propia del proyecto.

La documentación principal se mantiene únicamente en `docs/INSTALACION.md`,
`docs/COMANDOS.md`, `docs/ARQUITECTURA.md` y `docs/ACELERADORES.md`.
