# Modelo preentrenado de referencia · fase 2

Este es el modelo de referencia incluido para comprobar el sistema sin entrenar
desde cero.

- Perfil PPO guardado: `lite`, nombre histórico equivalente al perfil actual
  `ligero`.
- Fase de recompensa: `2`, levantarse desde el suelo.
- Semilla: `42`.
- Checkpoint: `45.932.544` pasos de un objetivo de 100 millones.
- Recompensa de evaluación guardada: `158,104767`.
- Longitud media de evaluación: `1.432,25` pasos.
- Formato: Orbax OCDBT.

No se presenta como un modelo terminado al 100 %. Es la referencia más avanzada
y estable que se conserva actualmente. Sustituye como referencia visual al
checkpoint nuevo de 1.548.288 pasos, que todavía estaba demasiado poco
entrenado.

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
