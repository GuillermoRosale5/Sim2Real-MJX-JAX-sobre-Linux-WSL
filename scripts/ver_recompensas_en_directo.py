from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

try:
  import matplotlib
  import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
  if exc.name != "matplotlib":
    raise
  print(
      "Falta matplotlib en este Python. No ejecutes este archivo con Python de Windows.\n"
      "Desde WSL usa:\n"
      "  ./scripts/ver_recompensas_en_directo.sh",
      file=sys.stderr,
  )
  raise SystemExit(2) from exc

from scripts.graficar_recompensas import COLUMNAS_PREDETERMINADAS_ARG
from scripts.graficar_recompensas import _filtrar_ultimo_segmento
from scripts.graficar_recompensas import _elegir_filas_grafica
from scripts.graficar_recompensas import _estilo_grafica
from scripts.graficar_recompensas import _etiqueta_leyenda
from scripts.graficar_recompensas import _resolver_columnas
from scripts.graficar_recompensas import _serie
from scripts.graficar_recompensas import _suavizar
from scripts.graficar_recompensas import _texto_intervalo_temporal
from scripts.graficar_recompensas import _tiene_datos
from scripts.graficar_recompensas import _ultima_ejecucion


def _leer_csv_directo(ruta: Path) -> list[dict[str, str]]:
  if not ruta.exists() or ruta.stat().st_size == 0:
    return []
  try:
    with ruta.open(newline="", encoding="utf-8") as archivo:
      return list(csv.DictReader(archivo))
  except (OSError, csv.Error):
    return []


def _resolver_rutas_directo(argumentos: argparse.Namespace) -> tuple[Path, Path]:
  if argumentos.ruta_csv:
    ruta_csv = Path(argumentos.ruta_csv)
    return ruta_csv.parent, ruta_csv
  directorio_ejecucion = (
      Path(argumentos.directorio_ejecucion)
      if argumentos.directorio_ejecucion
      else _ultima_ejecucion(Path(argumentos.directorio_registros))
  )
  return directorio_ejecucion, directorio_ejecucion / "recompensas.csv"


def _dibujar_espera(ejes, mensaje: str) -> None:
  ejes.clear()
  ejes.set_title(mensaje)
  ejes.grid(True, alpha=0.3)


def _dibujar_grafica(
    ejes,
    filas: list[dict[str, str]],
    ruta_csv: Path,
    directorio_ejecucion: Path,
    argumentos: argparse.Namespace,
) -> tuple[int, str]:
  if argumentos.segmento == "ultimo":
    filas = _filtrar_ultimo_segmento(filas)
  filas, origen_usado = _elegir_filas_grafica(
      filas, argumentos.origen, argumentos.columnas
  )
  if argumentos.ultimas_filas > 0:
    filas = filas[-argumentos.ultimas_filas :]

  valores_x = _serie(filas, argumentos.eje_x)
  if not _tiene_datos(valores_x):
    _dibujar_espera(
        ejes, f"Esperando datos validos para eje X: {argumentos.eje_x}"
    )
    return 0, ""

  x_limpio = [0.0 if value is None else value for value in valores_x]
  columnas = _resolver_columnas(filas, argumentos.columnas)

  ejes.clear()
  curvas_dibujadas = 0
  fila_muestra = filas[0]
  for columna in columnas:
    valores = _suavizar(
        _serie(
            filas,
            columna,
            unidades_recompensa=argumentos.unidades_recompensa,
        ),
        argumentos.suavizado,
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
    _dibujar_espera(
        ejes, f"Esperando columnas de recompensa con datos en {ruta_csv.name}"
    )
    return 0, ""

  texto_temporal = _texto_intervalo_temporal(
      ruta_csv, directorio_ejecucion, filas
  )
  ejes.set_title(
      f"Recompensas en directo - {directorio_ejecucion.name} / {ruta_csv.name}\n"
      f"{texto_temporal} | origen={origen_usado}"
  )
  ejes.set_xlabel(argumentos.eje_x)
  unidades_y = (
      "recompensa PPO escalada"
      if argumentos.unidades_recompensa == "escaladas"
      else "recompensa bruta"
  )
  ejes.set_ylabel(
      f"recompensa ponderada ({unidades_y}); penalizaciones dibujadas en negativo"
  )
  ejes.grid(True, alpha=0.3)
  ejes.legend(loc="best", fontsize="small", ncols=3)
  return curvas_dibujadas, texto_temporal


def _ultimo_no_vacio(filas: list[dict[str, str]], clave: str) -> str:
  for fila in reversed(filas):
    valor = fila.get(clave, "")
    if valor not in ("", None):
      return str(valor)
  return ""


def _linea_estado(
    filas: list[dict[str, str]], ruta_csv: Path, curvas_dibujadas: int
) -> str:
  if not filas:
    return f"Esperando {ruta_csv}..."
  partes = [
      f"filas={len(filas)}",
      f"curvas={curvas_dibujadas}",
      f"pasos={_ultimo_no_vacio(filas, 'num_steps')}",
  ]
  reward_total = _ultimo_no_vacio(filas, "reward_total")
  positive_reward = _ultimo_no_vacio(filas, "positive_reward")
  penalties = _ultimo_no_vacio(filas, "penalties")
  fase = _ultimo_no_vacio(filas, "fase_curriculum_recompensa")
  nombre_fase = _ultimo_no_vacio(filas, "nombre_curriculum_recompensa")
  if reward_total:
    partes.append(f"reward_total={reward_total}")
  if positive_reward:
    partes.append(f"positive_reward={positive_reward}")
  if penalties:
    partes.append(f"penalties={penalties}")
  if fase or nombre_fase:
    partes.append(f"fase={fase} {nombre_fase}".strip())
  return " | ".join(partes)


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Muestra en directo la evolucion de recompensas desde recompensas.csv."
  )
  parser.add_argument("--directorio-registros", default="logs_sim2real_mjx")
  parser.add_argument("--directorio-ejecucion", default=None)
  parser.add_argument("--ruta-csv", default=None)
  parser.add_argument(
      "--eje-x",
      choices=["num_steps", "elapsed_seconds", "elapsed_hours", "percent"],
      default="num_steps",
  )
  parser.add_argument("--columnas", default=COLUMNAS_PREDETERMINADAS_ARG)
  parser.add_argument(
      "--origen",
      choices=["evaluacion", "entrenamiento", "todos"],
      default="todos",
  )
  parser.add_argument(
      "--segmento", choices=["ultimo", "todos"], default="ultimo"
  )
  parser.add_argument("--suavizado", type=int, default=3)
  parser.add_argument(
      "--unidades-recompensa",
      choices=["escaladas", "brutas"],
      default="escaladas",
      help=(
          "escaladas muestra las contribuciones ya multiplicadas por reward_scale; "
          "brutas muestra las sumas internas sin reward_scale."
      ),
  )
  parser.add_argument("--intervalo", type=float, default=5.0)
  parser.add_argument(
      "--ultimas-filas",
      type=int,
      default=300,
      help="Numero maximo de filas recientes a dibujar. Usa 0 para todo el CSV.",
  )
  parser.add_argument(
      "--salida",
      default=None,
      help="PNG que se sobrescribe en cada refresco. Opcional.",
  )
  parser.add_argument(
      "--sin-ventana",
      action="store_true",
      help="No abre ventana; solo actualiza --salida e imprime estado.",
  )
  parser.add_argument(
      "--una-vez",
      action="store_true",
      help="Hace un unico refresco y termina. Util para probar o generar PNG puntual.",
  )
  argumentos = parser.parse_args()

  if argumentos.sin_ventana and not argumentos.salida:
    raise SystemExit("--sin-ventana necesita --salida para poder ver algo.")

  mostrar_ventana = not argumentos.sin_ventana
  if mostrar_ventana:
    plt.ion()
    figura, ejes = plt.subplots(figsize=(13, 7))
  else:
    matplotlib.use("Agg", force=True)
    figura, ejes = plt.subplots(figsize=(13, 7))

  print("Ctrl+C para salir.", flush=True)
  ultima_modificacion = None
  try:
    while True:
      directorio_ejecucion, ruta_csv = _resolver_rutas_directo(argumentos)
      filas = _leer_csv_directo(ruta_csv)
      modificacion = ruta_csv.stat().st_mtime if ruta_csv.exists() else None
      redibujar = modificacion != ultima_modificacion or mostrar_ventana

      if redibujar:
        ultima_modificacion = modificacion
        if not filas:
          _dibujar_espera(ejes, f"Esperando datos en {ruta_csv}")
          curvas_dibujadas = 0
        else:
          curvas_dibujadas, _ = _dibujar_grafica(
              ejes, filas, ruta_csv, directorio_ejecucion, argumentos
          )

        figura.tight_layout()
        if argumentos.salida:
          ruta_salida = Path(argumentos.salida)
          ruta_salida.parent.mkdir(parents=True, exist_ok=True)
          figura.savefig(ruta_salida, dpi=150)
        if mostrar_ventana:
          figura.canvas.draw_idle()
          plt.pause(0.05)

        print(
            _linea_estado(filas, ruta_csv, curvas_dibujadas),
            flush=True,
        )
        if argumentos.una_vez:
          break

      if mostrar_ventana and not plt.fignum_exists(figura.number):
        break
      time.sleep(max(argumentos.intervalo, 0.5))
  except KeyboardInterrupt:
    print("\nMonitor de recompensas detenido.", flush=True)
  finally:
    plt.close(figura)


if __name__ == "__main__":
  main()
