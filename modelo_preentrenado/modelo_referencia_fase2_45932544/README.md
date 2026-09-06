# Modelo preentrenado de referencia · fase 1

Este es el modelo de referencia incluido para comprobar el sistema sin entrenar
desde cero.

Es la red entrenada más estable que  tenemos. 

Se optimizó para la siguiente configuración arquitectónica:

Actor:
59 observaciones
      ↓
256 neuronas + SiLU
      ↓
256 neuronas + SiLU
      ↓
24 parámetros de distribución
      ↓
12 acciones de los motores
Critic:
59 → 256 → 256 → 1 valor estimado

Configuración adicional:
- Dos capas ocultas de 256 neuronas.
- Activación SiLU/Swish.
- Distribución de acciones tanh_normal.
- Normalización de observaciones.
- Fase curricular 2.
- 512 entornos paralelos.
- Episodios de 1.500 pasos máx.


El paquete conserva los pesos y las estadísticas de normalización de la
política para inferencia y visualización. No incluye un estado completo del
optimizador que permita continuar exactamente desde la misma operación. Los
metadatos identifican la ejecución original como
`EntornoRobotMJX-20260520-185753`.

Desde Ubuntu nativo o desde una terminal de Ubuntu en WSL2:

```bash
./scripts/visualizar_modelo_preentrenado.sh
```

El comando comprueba `SHA256SUMS` antes de cargar los pesos. Esta carpeta sí se
versiona de forma deliberada; los checkpoints generados por entrenamientos
normales continúan ignorados por Git.
