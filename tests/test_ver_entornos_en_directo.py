"""Pruebas ligeras del visor de varios entornos en una sola escena."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import stat
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
VISOR = ROOT / "scripts/ver_entornos_en_directo.py"
VISOR_ANTERIOR = ROOT / "scripts/visualizar_resultados_mjx.py"
LANZADOR = ROOT / "scripts/VER_EN_DIRECTO_100_ENTRENAMIENTOS_EN_PARALELO"
SIM2REAL = ROOT / "scripts/sim2real.sh"


class _MatrizMinima(list[tuple[float, float]]):

  def __isub__(self, desplazamiento):
    x_centro, y_centro = desplazamiento[0]
    self[:] = [
        (x - float(x_centro), y - float(y_centro)) for x, y in self
    ]
    return self


class _NumpyMinimo:
  """Sustituto suficiente para probar la cuadricula sin depender de NumPy."""

  float64 = float

  @staticmethod
  def asarray(valores, dtype=None):
    if dtype is not _NumpyMinimo.float64:
      raise AssertionError("La cuadricula debe usar float64.")
    return _MatrizMinima(
        tuple(float(coordenada) for coordenada in fila) for fila in valores
    )

  @staticmethod
  def mean(valores, axis=None, keepdims=False):
    if axis != 0 or not keepdims:
      raise AssertionError("La cuadricula debe recentrarse por columnas.")
    cantidad = len(valores)
    return ((
        sum(x for x, _ in valores) / cantidad,
        sum(y for _, y in valores) / cantidad,
    ),)


def _cargar_funciones_puras() -> dict[str, object]:
  """Extrae solo helpers sin importar JAX, Brax, NumPy ni MuJoCo."""
  arbol = ast.parse(VISOR.read_text(encoding="utf-8"))
  nombres = {
      "_candidatos_automaticos",
      "_fps_automaticos",
      "_cadencia_actualizacion_automatica",
      "_posiciones_cuadricula",
      "_checkpoint_completo",
      "_ultimo_checkpoint_de_directorio",
  }
  seleccionados: list[ast.stmt] = [
      nodo
      for nodo in arbol.body
      if (
          isinstance(nodo, ast.ImportFrom)
          and nodo.module == "__future__"
      )
      or (isinstance(nodo, ast.FunctionDef) and nodo.name in nombres)
  ]
  encontrados = {
      nodo.name for nodo in seleccionados if isinstance(nodo, ast.FunctionDef)
  }
  if encontrados != nombres:
    raise AssertionError(f"Faltan helpers puros del visor: {sorted(nombres - encontrados)}")

  espacio: dict[str, object] = {
      "json": json,
      "math": math,
      "np": _NumpyMinimo,
      "Path": Path,
  }
  modulo = ast.fix_missing_locations(ast.Module(body=seleccionados, type_ignores=[]))
  exec(compile(modulo, str(VISOR), "exec"), espacio)
  return espacio


def _crear_checkpoint(ruta: Path, contenido: object | None = None) -> Path:
  ruta.mkdir(parents=True)
  if contenido is not None:
    (ruta / "ppo_network_config.json").write_text(
        json.dumps(contenido), encoding="utf-8"
    )
  return ruta


class VisualizadorEntornosDirectoTests(unittest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    cls.funciones = _cargar_funciones_puras()

  def test_cuadricula_admite_uno_siete_y_cien_entornos(self) -> None:
    posiciones = self.funciones["_posiciones_cuadricula"]
    separacion = 1.1

    uno = posiciones(1, separacion)
    self.assertEqual(tuple(uno), ((0.0, 0.0),))

    for cantidad in (7, 100):
      with self.subTest(cantidad=cantidad):
        cuadricula = posiciones(cantidad, separacion)
        self.assertEqual(len(cuadricula), cantidad)
        self.assertEqual(len(set(cuadricula)), cantidad)
        for x, y in cuadricula:
          self.assertTrue(math.isfinite(x))
          self.assertTrue(math.isfinite(y))
        distancia_minima = min(
            math.hypot(x_a - x_b, y_a - y_b)
            for indice, (x_a, y_a) in enumerate(cuadricula)
            for x_b, y_b in cuadricula[indice + 1:]
        )
        self.assertGreaterEqual(distancia_minima, separacion - 1e-12)

    cien = posiciones(100, separacion)
    self.assertEqual(len({x for x, _ in cien}), 10)
    self.assertEqual(len({y for _, y in cien}), 10)
    self.assertAlmostEqual(min(x for x, _ in cien), -4.5 * separacion)
    self.assertAlmostEqual(max(x for x, _ in cien), 4.5 * separacion)
    self.assertAlmostEqual(min(y for _, y in cien), -4.5 * separacion)
    self.assertAlmostEqual(max(y for _, y in cien), 4.5 * separacion)

  def test_defaults_escalan_candidatos_y_fps_sin_exceder_limites(self) -> None:
    candidatos = self.funciones["_candidatos_automaticos"]
    fps = self.funciones["_fps_automaticos"]
    cadencia = self.funciones["_cadencia_actualizacion_automatica"]

    casos_candidatos = {
        1: 32,
        7: 32,
        16: 32,
        17: 34,
        64: 128,
        65: 128,
        100: 128,
    }
    for cantidad, esperado in casos_candidatos.items():
      with self.subTest(cantidad=cantidad, valor="candidatos"):
        self.assertEqual(candidatos(cantidad), esperado)
        self.assertGreaterEqual(candidatos(cantidad), cantidad)

    casos_fps = {1: 20.0, 25: 20.0, 26: 12.0, 64: 12.0, 65: 8.0, 100: 8.0}
    for cantidad, esperado in casos_fps.items():
      with self.subTest(cantidad=cantidad, valor="fps"):
        self.assertEqual(fps(cantidad), esperado)

    for cantidad, esperado in {1: 1, 63: 1, 64: 2, 100: 2}.items():
      with self.subTest(cantidad=cantidad, valor="cadencia"):
        self.assertEqual(cadencia(cantidad), esperado)

  def test_checkpoint_completo_exige_directorio_numerico_y_marcador_valido(self) -> None:
    checkpoint_completo = self.funciones["_checkpoint_completo"]

    with tempfile.TemporaryDirectory() as temporal:
      raiz = Path(temporal)
      completo = _crear_checkpoint(raiz / "000000000010", {"action_size": 12})
      parcial = _crear_checkpoint(raiz / "000000000011")
      corrupto = _crear_checkpoint(raiz / "000000000012")
      (corrupto / "ppo_network_config.json").write_text("{", encoding="utf-8")
      sin_acciones = _crear_checkpoint(raiz / "000000000013", {"otra": 12})
      no_numerico = _crear_checkpoint(raiz / "ultimo", {"action_size": 12})

      self.assertTrue(checkpoint_completo(completo))
      for ruta in (parcial, corrupto, sin_acciones, no_numerico):
        with self.subTest(ruta=ruta.name):
          self.assertFalse(checkpoint_completo(ruta))

  def test_checkpoint_y_marcador_simbolicos_se_ignoran(self) -> None:
    checkpoint_completo = self.funciones["_checkpoint_completo"]

    with tempfile.TemporaryDirectory() as temporal:
      raiz = Path(temporal)
      destino = _crear_checkpoint(
          raiz / "destino" / "000000000020", {"action_size": 12}
      )
      enlace_checkpoint = raiz / "000000000021"
      enlace_checkpoint.symlink_to(destino, target_is_directory=True)

      checkpoint_marcador = _crear_checkpoint(raiz / "000000000022")
      marcador_real = raiz / "marcador_real.json"
      marcador_real.write_text(json.dumps({"action_size": 12}), encoding="utf-8")
      (checkpoint_marcador / "ppo_network_config.json").symlink_to(marcador_real)

      self.assertFalse(checkpoint_completo(enlace_checkpoint))
      self.assertFalse(checkpoint_completo(checkpoint_marcador))

  def test_ultimo_checkpoint_ignora_un_guardado_parcial_y_un_enlace(self) -> None:
    ultimo_checkpoint = self.funciones["_ultimo_checkpoint_de_directorio"]

    with tempfile.TemporaryDirectory() as temporal:
      raiz = Path(temporal) / "checkpoints"
      _crear_checkpoint(raiz / "000000000010", {"action_size": 12})
      esperado = _crear_checkpoint(raiz / "000000000020", {"action_size": 12})
      _crear_checkpoint(raiz / "000000000030")
      destino = _crear_checkpoint(
          Path(temporal) / "fuera" / "000000000040", {"action_size": 12}
      )
      (raiz / "000000000040").symlink_to(destino, target_is_directory=True)

      self.assertEqual(ultimo_checkpoint(raiz), esperado.resolve())

  def test_launcher_exacto_delega_una_sola_vez_y_conserva_argumentos(self) -> None:
    self.assertTrue(LANZADOR.is_file())
    self.assertTrue(LANZADOR.stat().st_mode & stat.S_IXUSR)
    fuente = LANZADOR.read_text(encoding="utf-8")
    self.assertTrue(fuente.startswith("#!/usr/bin/env bash\n"))
    self.assertIn("set -euo pipefail", fuente)
    self.assertEqual(fuente.count("exec "), 1)
    self.assertIn(
        'exec "${SCRIPT_DIR}/sim2real.sh" visualizar-entornos-en-directo "$@"',
        fuente,
    )

  def test_sim2real_expone_comando_y_configura_runtime_del_visor(self) -> None:
    fuente = SIM2REAL.read_text(encoding="utf-8")
    self.assertIn("visualizar_entornos_en_directo() {", fuente)
    self.assertIn(
        'visualizar-entornos-en-directo) visualizar_entornos_en_directo "$@" ;;',
        fuente,
    )
    bloque = fuente.split("visualizar_entornos_en_directo() {", 1)[1].split(
        "\n}\n", 1
    )[0]
    self.assertIn("viewer_env", bloque)
    self.assertIn('export JAX_PLATFORM_NAME=cpu', bloque)
    self.assertIn('export JAX_PLATFORMS=cpu', bloque)
    self.assertIn('"${REPO_ROOT}/scripts/ver_entornos_en_directo.py"', bloque)
    self.assertIn('--directorio-registros "${LOGS_DIR}"', bloque)
    self.assertIn('--checkpoint-respaldo "${CHECKPOINT_PREENTRENADO}"', bloque)
    self.assertIn('"${argumentos_visualizador[@]}"', bloque)
    self.assertLess(bloque.index("viewer_env"), bloque.index("uv_python"))

  def test_nucleo_conserva_una_ventana_y_una_escena_sin_simulacion_clasica(self) -> None:
    fuente = VISOR.read_text(encoding="utf-8")
    self.assertEqual(fuente.count("mujoco.viewer.launch_passive("), 1)
    self.assertNotIn("mujoco.mj_step(", fuente)
    self.assertIn('"--cantidad-entornos"', fuente)
    self.assertIn('"--solo-comprobar"', fuente)
    self.assertIn("queue.Queue(maxsize=1)", fuente)
    self.assertIn("for actuador in list(hijo.actuators):", fuente)
    self.assertIn("for textura in list(hijo.textures):", fuente)

  def test_ambos_visores_restauran_el_preprocesado_guardado_por_brax(self) -> None:
    fuente_multiple = VISOR.read_text(encoding="utf-8")
    fuente_anterior = VISOR_ANTERIOR.read_text(encoding="utf-8")
    self.assertIn(
        "from visualizar_resultados_mjx import _crear_redes_desde_checkpoint",
        fuente_multiple,
    )
    self.assertIn("def _crear_redes_desde_checkpoint(", fuente_anterior)
    self.assertIn("running_statistics.normalize", fuente_anterior)
    self.assertIn('"ppo_network_config.json"', fuente_anterior)


if __name__ == "__main__":
  unittest.main()
