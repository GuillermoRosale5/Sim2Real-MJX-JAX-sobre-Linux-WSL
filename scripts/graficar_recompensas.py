from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
import re
import sys

try:
  import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
  if exc.name != "matplotlib":
    raise
  print(
      "Falta matplotlib en este Python. Usa el entorno del repositorio desde Ubuntu/WSL:\n"
      "  ./scripts/graficar_recompensas.sh",
      file=sys.stderr,
  )
  raise SystemExit(2) from exc


COLUMNAS_PREDETERMINADAS = [
    "reward_total",
    "positive_reward",
    "penalties",
    "supervivencia_reward_ponderado",
    "vector_gravedad_reward_ponderado",
    "soporte_estatico_reward_ponderado",
    "altura_reward_ponderado",
    "plano_CoG_arana_reward_ponderado",
    "contactos_suelo_reward_ponderado",
    "poligono_CoG_reward_ponderado",
    "apertura_efectores_q3_cerca_reward_ponderado",
    "q1_centrados_reward_ponderado",
    "q3_separados_centro_reward_ponderado",
    "simetria_patas_reward_ponderado",
    "contacto_invalido_penalty_ponderado",
    "contacto_invalido_persistente_penalty_ponderado",
    "efector_encima_q3_penalty_ponderado",
    "velocidad_vertical_cuerpo_penalty_ponderado",
    "velocidad_angular_cuerpo_penalty_ponderado",
    "control_penalty_ponderado",
    "cambio_accion_penalty_ponderado",
    "limite_articular_penalty_ponderado",
]

COLUMNAS_PREDETERMINADAS_ARG = "predeterminadas"

ALIAS_COLUMNAS_HISTORICAS = {
    "eval_episode_reward": ("eval_episode_reward",),
    "eval_reward_per_step": ("eval_reward_per_step",),
    "positive_reward": ("positive_reward", "reward_positive_total"),
    "penalties": ("penalties", "penalty_total"),
    "supervivencia_reward_ponderado": (
        "supervivencia_reward_ponderado",
        "reward_supervivencia_reward",
        "reward_supervivencia_gate",
        "supervivencia_reward_gate",
    ),
    "vector_gravedad_reward_ponderado": ("vector_gravedad_reward_ponderado", "reward_vector_gravedad"),
    "poligono_CoG_reward_ponderado": ("poligono_CoG_reward_ponderado", "reward_poligono_CoG"),
    "altura_reward_ponderado": ("altura_reward_ponderado", "reward_altura"),
    "plano_CoG_arana_reward_ponderado": (
        "plano_CoG_arana_reward_ponderado",
        "plano_CoG_ara?a_reward",
        "reward_plano_CoG_ara?a",
    ),
    "contactos_suelo_reward_ponderado": ("contactos_suelo_reward_ponderado", "reward_contactos_suelo"),
    "soporte_estatico_reward_ponderado": (
        "soporte_estatico_reward_ponderado",
        "reward_soporte_estatico",
        "reward/soporte_estatico",
    ),
    "apertura_efectores_q3_cerca_reward_ponderado": ("apertura_efectores_q3_cerca_reward_ponderado",),
    "q1_centrados_reward_ponderado": ("q1_centrados_reward_ponderado",),
    "q3_separados_centro_reward_ponderado": (
        "q3_separados_centro_reward_ponderado",
        "reward_q3_separados_centro",
    ),
    "simetria_patas_reward_ponderado": ("simetria_patas_reward_ponderado", "reward_simetria_patas"),
    "contacto_invalido_penalty_ponderado": (
        "contacto_invalido_penalty_ponderado",
        "invalid_contact_penalty",
        "penalty_invalid_contact",
        "penalty_invalid_ground_contact",
        "contacto_invalido_legacy_penalty",
        "contacto_suelo_invalido_penalty",
    ),
    "contacto_invalido_persistente_penalty_ponderado": (
        "contacto_invalido_persistente_penalty_ponderado",
        "invalid_contact_persistence_penalty",
    ),
    "efector_encima_q3_penalty_ponderado": (
        "efector_encima_q3_penalty_ponderado",
        "penalty_efector_encima_q3",
    ),
    "velocidad_vertical_cuerpo_penalty_ponderado": (
        "velocidad_vertical_cuerpo_penalty_ponderado",
        "penalty_velocidad_vertical_cuerpo",
    ),
    "velocidad_angular_cuerpo_penalty_ponderado": (
        "velocidad_angular_cuerpo_penalty_ponderado",
        "penalty_velocidad_angular_cuerpo",
    ),
    "control_penalty_ponderado": ("control_penalty_ponderado", "penalty_control"),
    "cambio_accion_penalty_ponderado": (
        "cambio_accion_penalty_ponderado",
        "action_rate_penalty",
        "penalty_action_rate",
    ),
    "limite_articular_penalty_ponderado": (
        "limite_articular_penalty_ponderado",
        "joint_limit_penalty",
        "penalty_joint_limit",
    ),
}

ALIAS_COLUMNAS: dict[str, tuple[str, ...]] = {}

COLUMNAS_NEGADAS_EN_GRAFICA = {
    "contacto_invalido_penalty_ponderado",
    "contacto_invalido_persistente_penalty_ponderado",
    "efector_encima_q3_penalty_ponderado",
    "velocidad_vertical_cuerpo_penalty_ponderado",
    "velocidad_angular_cuerpo_penalty_ponderado",
    "control_penalty_ponderado",
    "cambio_accion_penalty_ponderado",
    "limite_articular_penalty_ponderado",
    "penalties",
}

COLUMNAS_OMITIDAS_EN_GRAFICA = {
    "num_steps",
    "elapsed_seconds",
    "elapsed_hours",
    "percent",
    "source",
    "eval_episode_reward",
    "eval_episode_length",
    "eval_reward_per_step",
    "wall_steps_per_second",
    "steps_per_second",
    "training_steps_per_second",
}

COLORES_RECOMPENSAS = [
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#17becf",
    "#bcbd22",
    "#8c564b",
    "#7f7f7f",
    "#aec7e8",
    "#98df8a",
    "#c5b0d5",
]

COLORES_PENALIZACIONES = [
    "#d62728",
    "#ff7f0e",
    "#e377c2",
    "#a55194",
    "#b15928",
    "#fb9a99",
    "#fdbf6f",
    "#cab2d6",
]

COLORES_ESTADO = [
    "#4c78a8",
    "#72b7b2",
    "#54a24b",
    "#b279a2",
    "#9d755d",
]

MARCADORES = ("o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p")

DUPLICADOS_SI_EXISTE_COLUMNA_CANONICA = {
    "positive_reward": "positive_reward",
    "reward_positive_total": "positive_reward",
    "penalties": "penalties",
    "penalty_total": "penalties",
    "supervivencia_reward_ponderado": "supervivencia_reward_ponderado",
    "reward_supervivencia_gate": "supervivencia_reward_ponderado",
    "reward_supervivencia_reward": "supervivencia_reward_ponderado",
    "supervivencia_reward_gate": "supervivencia_reward_ponderado",
    "vector_gravedad_reward_ponderado": "vector_gravedad_reward_ponderado",
    "reward_vector_gravedad": "vector_gravedad_reward_ponderado",
    "poligono_CoG_reward_ponderado": "poligono_CoG_reward_ponderado",
    "reward_poligono_CoG": "poligono_CoG_reward_ponderado",
    "altura_reward_ponderado": "altura_reward_ponderado",
    "reward_altura": "altura_reward_ponderado",
    "plano_CoG_arana_reward_ponderado": "plano_CoG_arana_reward_ponderado",
    "plano_CoG_ara?a_reward": "plano_CoG_arana_reward_ponderado",
    "reward_plano_CoG_ara?a": "plano_CoG_arana_reward_ponderado",
    "contactos_suelo_reward_ponderado": "contactos_suelo_reward_ponderado",
    "reward_contactos_suelo": "contactos_suelo_reward_ponderado",
    "soporte_estatico_reward_ponderado": "soporte_estatico_reward_ponderado",
    "reward_soporte_estatico": "soporte_estatico_reward_ponderado",
    "reward/soporte_estatico": "soporte_estatico_reward_ponderado",
    "apertura_efectores_q3_cerca_reward_ponderado": "apertura_efectores_q3_cerca_reward_ponderado",
    "q1_centrados_reward_ponderado": "q1_centrados_reward_ponderado",
    "q3_separados_centro_reward_ponderado": "q3_separados_centro_reward_ponderado",
    "reward_q3_separados_centro": "q3_separados_centro_reward_ponderado",
    "simetria_patas_reward_ponderado": "simetria_patas_reward_ponderado",
    "reward_simetria_patas": "simetria_patas_reward_ponderado",
    "contacto_invalido_penalty_ponderado": "contacto_invalido_penalty_ponderado",
    "invalid_contact_penalty": "contacto_invalido_penalty_ponderado",
    "penalty_invalid_contact": "contacto_invalido_penalty_ponderado",
    "penalty_invalid_ground_contact": "contacto_invalido_penalty_ponderado",
    "contacto_suelo_invalido_penalty": "contacto_invalido_penalty_ponderado",
    "contacto_invalido_legacy_penalty": "contacto_invalido_penalty_ponderado",
    "contacto_invalido_persistente_penalty_ponderado": "contacto_invalido_persistente_penalty_ponderado",
    "invalid_contact_persistence_penalty": "contacto_invalido_persistente_penalty_ponderado",
    "efector_encima_q3_penalty_ponderado": "efector_encima_q3_penalty_ponderado",
    "penalty_efector_encima_q3": "efector_encima_q3_penalty_ponderado",
    "velocidad_vertical_cuerpo_penalty_ponderado": "velocidad_vertical_cuerpo_penalty_ponderado",
    "penalty_velocidad_vertical_cuerpo": "velocidad_vertical_cuerpo_penalty_ponderado",
    "velocidad_angular_cuerpo_penalty_ponderado": "velocidad_angular_cuerpo_penalty_ponderado",
    "penalty_velocidad_angular_cuerpo": "velocidad_angular_cuerpo_penalty_ponderado",
    "control_penalty_ponderado": "control_penalty_ponderado",
    "penalty_control": "control_penalty_ponderado",
    "action_rate_penalty": "cambio_accion_penalty_ponderado",
    "penalty_action_rate": "cambio_accion_penalty_ponderado",
    "joint_limit_penalty": "limite_articular_penalty_ponderado",
    "penalty_joint_limit": "limite_articular_penalty_ponderado",
}

def _ultima_ejecucion_desde_puntero(directorio_registros: Path) -> Path | None:
  ruta_puntero = directorio_registros / "ultima_ejecucion.txt"
  if not ruta_puntero.exists() or ruta_puntero.is_symlink():
    return None
  try:
    directorio_ejecucion = Path(ruta_puntero.read_text(encoding="utf-8").strip())
    registros_resueltos = directorio_registros.resolve(strict=True)
    ejecucion_resuelta = directorio_ejecucion.resolve(strict=True)
  except OSError:
    return None
  if (
      directorio_ejecucion.is_dir()
      and not directorio_ejecucion.is_symlink()
      and ejecucion_resuelta.parent == registros_resueltos
      and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", ejecucion_resuelta.name)
  ):
    return ejecucion_resuelta
  return None


def _clave_orden_ejecucion(directorio_ejecucion: Path) -> float:
  candidatos = [directorio_ejecucion.stat().st_mtime]
  for nombre in ("recompensas.csv", "progreso.csv"):
    ruta = directorio_ejecucion / nombre
    if ruta.exists():
      candidatos.append(ruta.stat().st_mtime)
  return max(candidatos)


def _ultima_ejecucion(directorio_registros: Path) -> Path:
  ejecucion_indicada = _ultima_ejecucion_desde_puntero(directorio_registros)
  if ejecucion_indicada is not None:
    return ejecucion_indicada
  ejecuciones = [
      ruta
      for ruta in directorio_registros.iterdir()
      if ruta.is_dir()
      and not ruta.is_symlink()
      and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", ruta.name)
  ]
  if not ejecuciones:
    raise FileNotFoundError(f"No hay ejecuciones en {directorio_registros}")
  return max(ejecuciones, key=_clave_orden_ejecucion)


def _csvs_recompensa_de_ejecucion(directorio_ejecucion: Path) -> list[Path]:
  rutas_csv = [
      ruta
      for ruta in directorio_ejecucion.glob("recompensas*.csv")
      if ruta.is_file() and ruta.stat().st_size > 0
  ]
  if not rutas_csv:
    return []

  def clave_orden_csv(ruta: Path) -> tuple[int, float, str]:
    prioridad_activo = 0 if ruta.name == "recompensas.csv" else 1
    return (prioridad_activo, -ruta.stat().st_mtime, ruta.name)

  return sorted(rutas_csv, key=clave_orden_csv)


def _leer_csv(path: Path) -> list[dict[str, str]]:
  with path.open(newline="", encoding="utf-8") as f:
    return list(csv.DictReader(f))


def _a_numero(value: str) -> float | None:
  if value is None or value == "":
    return None
  try:
    return float(value)
  except ValueError:
    return None


def _columna_metrica_historica(nombre_metrica: str) -> str:
  return nombre_metrica.replace("/", "_")


def _nombre_metrica_real(column: str) -> str:
  for nombre_real, alias in ALIAS_COLUMNAS_HISTORICAS.items():
    if column == nombre_real or column in alias:
      return nombre_real
  if "/" in column:
    return column
  for prefix in ("reward", "penalty", "done", "state", "final", "debug"):
    marker = f"{prefix}_"
    if column.startswith(marker):
      return f"{prefix}/{column[len(marker):]}"
  return column


def _nombre_metrica_mostrado(column: str) -> str:
  for prefix in ("reward", "penalty", "done", "state", "final", "debug"):
    marker = f"{prefix}_"
    if column.startswith(marker):
      return f"{prefix}/{column[len(marker):]}"
  return column


def _columnas_candidatas(column: str) -> tuple[str, ...]:
  candidates = [column]
  candidates.extend(ALIAS_COLUMNAS.get(column, ()))
  aliases = ALIAS_COLUMNAS_HISTORICAS.get(column)
  if aliases:
    candidates.extend(aliases)
  real_name = _nombre_metrica_real(column)
  if real_name != column:
    candidates.append(real_name)
  if "/" in real_name:
    candidates.append(_columna_metrica_historica(real_name))
  return tuple(dict.fromkeys(candidates))


def _nombre_columna_encontrada(row: dict[str, str], column: str) -> str | None:
  for candidate in _columnas_candidatas(column):
    if candidate in row:
      return candidate
  return None


def _etiqueta_leyenda(row: dict[str, str], column: str) -> str:
  del row
  if column == "reward_total":
    return "recompensa total = recompensa positiva - penalizaciones"
  if column == "positive_reward":
    return "+ recompensa positiva"
  if column == "penalties":
    return "- penalizaciones"
  if column.endswith("_reward_ponderado"):
    return f"+ {column.removesuffix('_reward_ponderado')}_recompensa_ponderada"
  if column.endswith("_penalty_ponderado"):
    return f"- {column.removesuffix('_penalty_ponderado')}_penalizacion_ponderada"
  return _nombre_metrica_mostrado(column)


def _valor_columna(row: dict[str, str], column: str) -> str:
  for candidate in _columnas_candidatas(column):
    if candidate in row:
      return row.get(candidate, "")
  return ""


def _serie(
    rows: list[dict[str, str]],
    column: str,
    unidades_recompensa: str = "brutas",
) -> list[float | None]:
  if column == "reward_total":
    direct_values = [_a_numero(_valor_columna(row, column)) for row in rows]
    if _tiene_datos(direct_values):
      return _quizas_escalar_serie_recompensa(rows, column, direct_values, unidades_recompensa)
    valores_recompensa = _primera_serie_disponible(rows, ("positive_reward",))
    valores_penalizacion = _primera_serie_disponible(rows, ("penalties",))
    values = [
        None
        if valor_recompensa is None or valor_penalizacion is None
        else valor_recompensa - valor_penalizacion
        for valor_recompensa, valor_penalizacion in zip(
            valores_recompensa, valores_penalizacion, strict=True
        )
    ]
    return _quizas_escalar_serie_recompensa(rows, column, values, unidades_recompensa)
  values = [_a_numero(_valor_columna(row, column)) for row in rows]
  if _debe_negar_columna(column):
    values = [None if value is None else -value for value in values]
  return _quizas_escalar_serie_recompensa(rows, column, values, unidades_recompensa)


def _tiene_datos(values: list[float | None]) -> bool:
  return any(value is not None for value in values)


def _tiene_datos_no_nulos(values: list[float | None]) -> bool:
  return any(value is not None and abs(value) > 1e-12 for value in values)


def _primera_serie_disponible(
    rows: list[dict[str, str]], columns: tuple[str, ...]
) -> list[float | None]:
  for column in columns:
    values = [_a_numero(_valor_columna(row, column)) for row in rows]
    if _tiene_datos(values):
      return values
  return [None for _ in rows]


def _mediana(values: list[float]) -> float:
  ordered = sorted(values)
  middle = len(ordered) // 2
  if len(ordered) % 2:
    return ordered[middle]
  return 0.5 * (ordered[middle - 1] + ordered[middle])


def _factor_escala_recompensa(rows: list[dict[str, str]]) -> float:
  explicit_values = []
  for row in rows:
    value = _a_numero(row.get("debug_reward_scale", ""))
    if value is not None and value > 0.0:
      explicit_values.append(value)
  if explicit_values:
    return _mediana(explicit_values)

  ratios = []
  for row in rows:
    reward_scaled = _a_numero(_valor_columna(row, "eval_episode_reward"))
    reward_raw = _a_numero(_valor_columna(row, "reward_total"))
    if (
        reward_scaled is not None
        and reward_raw is not None
        and abs(reward_raw) > 1e-9
    ):
      ratio = reward_scaled / reward_raw
      if 0.0 < abs(ratio) <= 1.0:
        ratios.append(abs(ratio))
  if ratios:
    return _mediana(ratios)
  return 1.0


def _quizas_escalar_serie_recompensa(
    rows: list[dict[str, str]],
    column: str,
    values: list[float | None],
    unidades_recompensa: str,
) -> list[float | None]:
  if unidades_recompensa != "escaladas" or not _es_columna_grafica_predeterminada(column):
    return values
  scale = _factor_escala_recompensa(rows)
  return [None if value is None else value * scale for value in values]


def _es_columna_recompensa(column: str) -> bool:
  if column in COLUMNAS_OMITIDAS_EN_GRAFICA:
    return False
  if column in {"reward_total", "positive_reward"}:
    return True
  return column.endswith("_reward_ponderado")


def _es_columna_penalizacion(column: str) -> bool:
  if column in COLUMNAS_OMITIDAS_EN_GRAFICA:
    return False
  if column == "penalties":
    return True
  return column.endswith("_penalty_ponderado")


def _debe_negar_columna(column: str) -> bool:
  return column in COLUMNAS_NEGADAS_EN_GRAFICA or _es_columna_penalizacion(column)


def _es_columna_grafica_predeterminada(column: str) -> bool:
  return column == "reward_total" or _es_columna_recompensa(column) or _es_columna_penalizacion(column)


def _indice_estable(text: str, modulo: int) -> int:
  return sum((index + 1) * ord(char) for index, char in enumerate(text)) % modulo


def _estilo_grafica(column: str) -> dict[str, object]:
  if column == "reward_total":
    return {
        "color": "#111111",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 3.0,
        "markersize": 4.8,
        "alpha": 1.0,
        "zorder": 5,
    }
  if column == "positive_reward":
    return {
        "color": "#178c4f",
        "linestyle": "-",
        "marker": "s",
        "linewidth": 2.4,
        "markersize": 4.4,
        "alpha": 0.95,
        "zorder": 4,
    }
  if column == "penalties":
    return {
        "color": "#b2182b",
        "linestyle": "--",
        "marker": "X",
        "linewidth": 2.4,
        "markersize": 4.4,
        "alpha": 0.95,
        "zorder": 4,
    }
  if _es_columna_penalizacion(column):
    index = _indice_estable(column, len(COLORES_PENALIZACIONES))
    return {
        "color": COLORES_PENALIZACIONES[index],
        "linestyle": "--",
        "marker": MARCADORES[index % len(MARCADORES)],
        "linewidth": 1.8,
        "markersize": 4.0,
        "alpha": 0.92,
        "zorder": 3,
    }
  if _es_columna_recompensa(column):
    index = _indice_estable(column, len(COLORES_RECOMPENSAS))
    return {
        "color": COLORES_RECOMPENSAS[index],
        "linestyle": "-",
        "marker": MARCADORES[index % len(MARCADORES)],
        "linewidth": 1.9,
        "markersize": 4.0,
        "alpha": 0.92,
        "zorder": 3,
    }
  index = _indice_estable(column, len(COLORES_ESTADO))
  return {
      "color": COLORES_ESTADO[index],
      "linestyle": ":",
      "marker": MARCADORES[index % len(MARCADORES)],
      "linewidth": 1.4,
      "markersize": 3.4,
      "alpha": 0.82,
      "zorder": 2,
  }


def _resolver_columnas(
    rows: list[dict[str, str]],
    columns_arg: str,
) -> list[str]:
  if columns_arg == "predeterminadas":
    return _resolver_columnas_predeterminadas(rows)

  if columns_arg != "todas":
    return [col.strip() for col in columns_arg.split(",") if col.strip()]

  return [
      col
      for col in rows[0].keys()
      if _es_columna_grafica_predeterminada(col) and _tiene_datos(_serie(rows, col))
  ]


def _resolver_columnas_predeterminadas(rows: list[dict[str, str]]) -> list[str]:
  if not rows:
    return []

  header = list(rows[0].keys())
  columns: list[str] = []
  for preferred in COLUMNAS_PREDETERMINADAS:
    matched = _nombre_columna_encontrada(rows[0], preferred)
    if matched is None or preferred in columns:
      continue
    values = _serie(rows, preferred)
    if preferred in {"reward_total", "positive_reward", "penalties"}:
      columns.append(preferred)
      continue
    if _tiene_datos_no_nulos(values):
      columns.append(preferred)

  for column in header:
    if not _es_columna_grafica_predeterminada(column) or column in columns:
      continue
    canonical = DUPLICADOS_SI_EXISTE_COLUMNA_CANONICA.get(column)
    if canonical is not None:
      canonical_match = _nombre_columna_encontrada(rows[0], canonical)
      if canonical_match is not None and canonical in columns:
        continue
    values = _serie(rows, column)
    if _tiene_datos_no_nulos(values):
      columns.append(column)

  return columns


def _elegir_filas_grafica(
    rows: list[dict[str, str]],
    origen: str,
    argumento_columnas: str,
) -> tuple[list[dict[str, str]], str]:
  filas_filtradas = _filtrar_origen(rows, origen)
  columnas = _resolver_columnas(filas_filtradas, argumento_columnas)
  if any(_tiene_datos(_serie(filas_filtradas, columna)) for columna in columnas):
    return filas_filtradas, origen

  if origen == "todos":
    return filas_filtradas, origen

  origen_alternativo = "evaluacion" if origen == "entrenamiento" else "todos"
  filas_alternativas = _filtrar_origen(rows, origen_alternativo)
  columnas_alternativas = _resolver_columnas(filas_alternativas, argumento_columnas)
  if any(
      _tiene_datos(_serie(filas_alternativas, columna))
      for columna in columnas_alternativas
  ):
    print(
        f"No hay columnas con datos para origen={origen!r}; "
        f"uso origen={origen_alternativo!r}.",
        file=sys.stderr,
    )
    return filas_alternativas, origen_alternativo

  return filas_filtradas, origen


def _formatear_tiempo_real(timestamp: float) -> str:
  return dt.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def _inicio_ejecucion_desde_nombre(run_dir: Path) -> float | None:
  match = re.search(r"(\d{8})-(\d{6})", run_dir.name)
  if not match:
    return None
  try:
    parsed = dt.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
  except ValueError:
    return None
  return parsed.timestamp()


def _texto_intervalo_temporal(csv_path: Path, run_dir: Path, rows: list[dict[str, str]]) -> str:
  generated_ts = dt.datetime.now().timestamp()
  csv_mtime = csv_path.stat().st_mtime
  last_row = rows[-1]
  elapsed = _a_numero(last_row.get("elapsed_seconds", ""))
  start_ts = None
  if elapsed is not None:
    start_ts = csv_mtime - elapsed
  if start_ts is None:
    start_ts = _inicio_ejecucion_desde_nombre(run_dir)
  if start_ts is None:
    start_ts = run_dir.stat().st_mtime

  active = generated_ts - csv_mtime <= 300.0
  end_label = "en curso" if active else "finalizada"
  parts = [
      f"inicio: {_formatear_tiempo_real(start_ts)}",
      f"estado: {end_label}",
      f"ultimo CSV: {_formatear_tiempo_real(csv_mtime)}",
      f"grafica: {_formatear_tiempo_real(generated_ts)}",
  ]
  num_steps = last_row.get("num_steps", "")
  debug_version = last_row.get("debug_version", "")
  if num_steps:
    parts.append(f"pasos: {num_steps}")
  if debug_version:
    parts.append(f"version_recompensa: {debug_version}")
  return " | ".join(parts)


def _filtrar_origen(rows: list[dict[str, str]], origen: str) -> list[dict[str, str]]:
  origen_historico = {
      "entrenamiento": "train",
      "evaluacion": "eval",
      "todos": "all",
  }[origen]
  if origen_historico == "all" or "source" not in rows[0]:
    return rows
  filas_filtradas = [row for row in rows if row.get("source") == origen_historico]
  if filas_filtradas:
    return filas_filtradas
  print(
      f"No hay filas con origen={origen!r}; uso todas las filas disponibles.",
      file=sys.stderr,
  )
  return rows


def _filtrar_ultimo_segmento(rows: list[dict[str, str]]) -> list[dict[str, str]]:
  """Devuelve las filas desde el ultimo reinicio de num_steps/tiempo."""

  if not rows:
    return rows
  start = 0
  prev_steps = _a_numero(rows[0].get("num_steps", ""))
  prev_elapsed = _a_numero(rows[0].get("elapsed_seconds", ""))
  for index, row in enumerate(rows[1:], start=1):
    steps = _a_numero(row.get("num_steps", ""))
    elapsed = _a_numero(row.get("elapsed_seconds", ""))
    steps_reset = (
        steps is not None and prev_steps is not None and steps < prev_steps
    )
    elapsed_reset = (
        elapsed is not None and prev_elapsed is not None and elapsed < prev_elapsed
    )
    if steps_reset or elapsed_reset:
      start = index
    if steps is not None:
      prev_steps = steps
    if elapsed is not None:
      prev_elapsed = elapsed
  if start > 0:
    print(
        f"CSV con varias sesiones detectadas; uso el ultimo segmento "
        f"({len(rows) - start} de {len(rows)} filas).",
        file=sys.stderr,
    )
  return rows[start:]


def _suavizar(values: list[float | None], ventana: int) -> list[float | None]:
  if ventana <= 1:
    return values
  valores_suavizados: list[float | None] = []
  for index in range(len(values)):
    tramo = [
        value
        for value in values[max(0, index - ventana + 1) : index + 1]
        if value is not None
    ]
    valores_suavizados.append(sum(tramo) / len(tramo) if tramo else None)
  return valores_suavizados


def _nombre_archivo_seguro(path: Path) -> str:
  chars = [
      char if char.isalnum() or char in ("-", "_") else "_"
      for char in path.stem
  ]
  return "".join(chars).strip("_") or "recompensas"


def _ruta_salida_para_csv(
    ruta_csv: Path,
    directorio_ejecucion: Path,
    argumento_salida: str | None,
    salidas_multiples: bool,
) -> Path:
  nombre_predeterminado = "recompensas.png"
  if salidas_multiples:
    nombre_predeterminado = f"{_nombre_archivo_seguro(ruta_csv)}.png"

  if not argumento_salida:
    return directorio_ejecucion / nombre_predeterminado

  salida = Path(argumento_salida)
  if salida.suffix.lower() == ".png" and not salidas_multiples:
    return salida
  if salida.suffix.lower() == ".png":
    return salida.with_name(
        f"{salida.stem}_{_nombre_archivo_seguro(ruta_csv)}{salida.suffix}"
    )
  return salida / nombre_predeterminado


def _graficar_csv(
    ruta_csv: Path,
    directorio_ejecucion: Path,
    ruta_salida: Path,
    eje_x: str,
    argumento_columnas: str,
    origen: str,
    segmento: str,
    ventana_suavizado: int,
    unidades_recompensa: str,
    mostrar: bool,
) -> None:
  filas = _leer_csv(ruta_csv)
  if not filas:
    raise RuntimeError(f"{ruta_csv} esta vacio.")
  if segmento == "ultimo":
    filas = _filtrar_ultimo_segmento(filas)
  filas, origen = _elegir_filas_grafica(filas, origen, argumento_columnas)

  valores_x = _serie(filas, eje_x)
  if not _tiene_datos(valores_x):
    raise RuntimeError(f"No hay datos validos para eje X: {eje_x}")
  x_limpio = [0.0 if value is None else value for value in valores_x]

  columnas = _resolver_columnas(filas, argumento_columnas)

  figura, ejes = plt.subplots(figsize=(13, 7))
  curvas_dibujadas = 0
  fila_muestra = filas[0]
  for columna in columnas:
    valores = _suavizar(
        _serie(filas, columna, unidades_recompensa=unidades_recompensa),
        ventana_suavizado,
    )
    if not _tiene_datos(valores):
      continue
    y = [float("nan") if value is None else value for value in valores]
    ejes.plot(
        x_limpio,
        y,
        label=_etiqueta_leyenda(fila_muestra, columna),
        **_estilo_grafica(columna),
    )
    curvas_dibujadas += 1

  if curvas_dibujadas == 0:
    raise RuntimeError(f"No hay columnas con datos para graficar en {ruta_csv}.")

  texto_temporal = _texto_intervalo_temporal(
      ruta_csv, directorio_ejecucion, filas
  )
  ejes.set_title(
      f"Evolucion de recompensas - {directorio_ejecucion.name} / {ruta_csv.name} "
      f"(origen={origen}, tramo={segmento}, suavizado={ventana_suavizado}, "
      f"unidades={unidades_recompensa})\n"
      f"{texto_temporal}"
  )
  ejes.set_xlabel(eje_x)
  unidades_y = (
      "recompensa PPO escalada"
      if unidades_recompensa == "escaladas"
      else "recompensa bruta"
  )
  ejes.set_ylabel(
      f"recompensa ponderada ({unidades_y}); penalizaciones dibujadas en negativo"
  )
  ejes.grid(True, alpha=0.3)
  ejes.legend(loc="best", fontsize="small", ncols=3)
  figura.text(
      0.01,
      0.01,
      f"CSV: {ruta_csv.resolve()}",
      ha="left",
      va="bottom",
      fontsize="x-small",
  )
  figura.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))

  ruta_salida.parent.mkdir(parents=True, exist_ok=True)
  figura.savefig(ruta_salida, dpi=160)
  print(f"CSV usado: {ruta_csv}")
  print(f"Intervalo: {texto_temporal}")
  print(f"Grafica guardada en: {ruta_salida}")

  if mostrar:
    plt.show()
  plt.close(figura)


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Grafica la evolucion temporal de los componentes de recompensa."
  )
  parser.add_argument("--directorio-registros", default="logs_sim2real_mjx")
  parser.add_argument("--directorio-ejecucion", default=None)
  parser.add_argument("--ruta-csv", default=None)
  parser.add_argument(
      "--modo-csv",
      choices=["todos", "activo"],
      default="todos",
      help=(
          "todos genera una grafica por cada recompensas*.csv de la ejecucion; "
          "activo usa solo recompensas.csv."
      ),
  )
  parser.add_argument("--salida", default=None)
  parser.add_argument(
      "--eje-x",
      choices=["num_steps", "elapsed_seconds", "elapsed_hours", "percent"],
      default="num_steps",
  )
  parser.add_argument(
      "--columnas",
      default=COLUMNAS_PREDETERMINADAS_ARG,
      help=(
          "Columnas separadas por coma. Usa 'predeterminadas' para reward_total, "
          "recompensas y penalizaciones ponderadas presentes; 'todas' para todos "
          "los componentes finales disponibles. Los nombres de metricas conservan "
          "las claves historicas del CSV."
      ),
  )
  parser.add_argument(
      "--origen",
      choices=["entrenamiento", "evaluacion", "todos"],
      default="todos",
      help=(
          "Origen de metricas a graficar. 'todos' usa todo lo disponible y evita "
          "quedarse esperando si la ultima fila no es de evaluacion."
      ),
  )
  parser.add_argument(
      "--segmento",
      choices=["ultimo", "todos"],
      default="ultimo",
      help="Usa solo la ultima sesion dentro del CSV o todas las filas.",
  )
  parser.add_argument(
      "--suavizado",
      type=int,
      default=5,
      help="Media movil causal en numero de puntos. Usa 1 para desactivar.",
  )
  parser.add_argument(
      "--unidades-recompensa",
      choices=["escaladas", "brutas"],
      default="escaladas",
      help=(
          "escaladas muestra las contribuciones ya multiplicadas por reward_scale; "
          "brutas muestra las sumas internas sin reward_scale."
      ),
  )
  parser.add_argument("--mostrar", action="store_true")
  args = parser.parse_args()

  directorio_ejecucion = (
      Path(args.directorio_ejecucion)
      if args.directorio_ejecucion
      else _ultima_ejecucion(Path(args.directorio_registros))
  )
  if args.ruta_csv:
    rutas_csv = [Path(args.ruta_csv)]
  elif args.modo_csv == "activo":
    rutas_csv = [directorio_ejecucion / "recompensas.csv"]
  else:
    rutas_csv = _csvs_recompensa_de_ejecucion(directorio_ejecucion)

  if not rutas_csv:
    raise FileNotFoundError(
        f"No hay recompensas*.csv en {directorio_ejecucion}. "
        "Lanza una ejecucion nueva o usa --ruta-csv."
    )

  salidas_multiples = len(rutas_csv) > 1
  for ruta_csv in rutas_csv:
    if not ruta_csv.exists():
      raise FileNotFoundError(f"No existe {ruta_csv}.")
    ruta_salida = _ruta_salida_para_csv(
        ruta_csv=ruta_csv,
        directorio_ejecucion=directorio_ejecucion,
        argumento_salida=args.salida,
        salidas_multiples=salidas_multiples,
    )
    _graficar_csv(
        ruta_csv=ruta_csv,
        directorio_ejecucion=directorio_ejecucion,
        ruta_salida=ruta_salida,
        eje_x=args.eje_x,
        argumento_columnas=args.columnas,
        origen=args.origen,
        segmento=args.segmento,
        ventana_suavizado=args.suavizado,
        unidades_recompensa=args.unidades_recompensa,
        mostrar=args.mostrar,
    )


if __name__ == "__main__":
  main()
