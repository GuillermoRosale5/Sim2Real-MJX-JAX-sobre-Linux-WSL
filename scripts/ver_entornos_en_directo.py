"""Muestra en una sola ventana los mejores intentos de la politica actual.

La evaluacion se ejecuta en MJX/JAX por lotes. La ventana no vuelve a simular
esas fisicas: reproduce un muestreo pequeno de qpos sobre una unica escena de
MuJoCo con varias copias visuales del robot. De esta forma, la ventana sigue
siendo fluida aunque el entrenamiento utilice cientos de entornos en la GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time
import warnings

try:
  import jax
  import jax.numpy as jp
  import mujoco
  import mujoco.viewer
  import numpy as np
  from brax.training import checkpoint
  from brax.training.agents.ppo import networks as ppo_networks
  from mujoco_playground import wrapper
except ModuleNotFoundError as exc:
  print(
      f"Falta {exc.name} en este Python. Ejecuta antes ./scripts/install.sh.",
      file=sys.stderr,
  )
  raise SystemExit(2) from exc

from sim2real_mjx.entorno_robot_mjx import EntornoRobotMJX
from sim2real_mjx.entorno_robot_mjx import default_config
from visualizar_resultados_mjx import _aplicar_configuracion_entorno_guardada
from visualizar_resultados_mjx import _checkpoints_ordenados
from visualizar_resultados_mjx import _crear_redes_desde_checkpoint


warnings.filterwarnings(
    "ignore",
    message="overflow encountered in cast",
    category=RuntimeWarning,
)

RAIZ_REPOSITORIO = Path(__file__).resolve().parent.parent
MAXIMO_ENTORNOS_VISIBLES = 100
MAXIMO_CANDIDATOS = 512


class AnalizadorArgumentos(argparse.ArgumentParser):
  """Traduce las cabeceras que argparse deja en ingles por defecto."""

  def format_usage(self) -> str:
    return super().format_usage().replace("usage:", "uso:", 1)

  def format_help(self) -> str:
    texto = super().format_help().replace("usage:", "uso:", 1)
    return texto.replace("options:", "opciones:", 1)

  def error(self, message: str) -> None:
    self.print_usage(sys.stderr)
    self.exit(2, f"{self.prog}: error: {message}\n")


@dataclass(frozen=True)
class LoteVisual:
  """Trayectorias pequenas que ya se pueden copiar a la escena clasica."""

  qpos: np.ndarray
  puntuaciones: np.ndarray
  longitudes: np.ndarray
  ruta_checkpoint: Path
  pasos_evaluados: int
  segundos_calculo: float
  intervalo_fotograma: float


@dataclass(frozen=True)
class ActualizacionPreparada:
  evaluador: "EvaluadorLote"
  lote: LoteVisual


@dataclass(frozen=True)
class RespuestaProductor:
  token: object
  preparada: ActualizacionPreparada | None = None
  error: Exception | None = None


def _entero_entre(nombre: str, minimo: int, maximo: int):
  def convertir(texto: str) -> int:
    try:
      valor = int(texto)
    except ValueError as exc:
      raise argparse.ArgumentTypeError(f"{nombre} debe ser un numero entero") from exc
    if not minimo <= valor <= maximo:
      raise argparse.ArgumentTypeError(
          f"{nombre} debe estar entre {minimo} y {maximo}"
      )
    return valor

  return convertir


def _flotante_positivo(nombre: str):
  def convertir(texto: str) -> float:
    try:
      valor = float(texto)
    except ValueError as exc:
      raise argparse.ArgumentTypeError(f"{nombre} debe ser un numero") from exc
    if not math.isfinite(valor) or valor <= 0.0:
      raise argparse.ArgumentTypeError(f"{nombre} debe ser mayor que cero")
    return valor

  return convertir


def _cantidad_por_menu() -> int:
  if not sys.stdin.isatty():
    print("Entrada no interactiva: se mostraran 100 entornos.")
    return 100

  opciones = (1, 4, 9, 16, 25, 50, 100)
  print("\nCantidad de entrenamientos que quieres ver:")
  for indice, cantidad in enumerate(opciones, start=1):
    texto = " (recomendado para la vista completa)" if cantidad == 100 else ""
    print(f"  {indice}) {cantidad}{texto}")
  print(f"  {len(opciones) + 1}) Otra cantidad, entre 1 y 100")

  while True:
    try:
      seleccion = input(f"Elige opcion [1-{len(opciones) + 1}]: ").strip()
    except EOFError:
      return 100
    if seleccion.isdigit() and 1 <= int(seleccion) <= len(opciones):
      return opciones[int(seleccion) - 1]
    if seleccion == str(len(opciones) + 1):
      try:
        personalizada = input("Cantidad [1-100]: ").strip()
      except EOFError:
        return 100
      if personalizada.isdigit() and 1 <= int(personalizada) <= 100:
        return int(personalizada)
    print("Opcion no valida.")


def _candidatos_automaticos(cantidad: int) -> int:
  if cantidad <= 16:
    return max(32, cantidad * 2)
  if cantidad <= 64:
    return cantidad * 2
  return 128


def _fps_automaticos(cantidad: int) -> float:
  if cantidad <= 25:
    return 20.0
  if cantidad <= 64:
    return 12.0
  return 8.0


def _cadencia_actualizacion_automatica(cantidad: int) -> int:
  # En la vista grande, dos checkpoints consecutivos suelen contener cambios
  # visuales muy pequenos. Saltar uno reduce aproximadamente a la mitad el
  # trabajo lateral sin dejar de seguir el entrenamiento.
  return 2 if cantidad >= 64 else 1


def _posiciones_cuadricula(cantidad: int, separacion: float) -> np.ndarray:
  columnas = math.ceil(math.sqrt(cantidad))
  filas = math.ceil(cantidad / columnas)
  posiciones: list[tuple[float, float]] = []
  for indice in range(cantidad):
    fila, columna = divmod(indice, columnas)
    x = (columna - (columnas - 1) / 2.0) * separacion
    y = ((filas - 1) / 2.0 - fila) * separacion
    posiciones.append((x, y))
  resultado = np.asarray(posiciones, dtype=np.float64)
  # Una ultima fila incompleta desplaza el centro de masas de la cuadricula.
  # Recentrar el conjunto mantiene todos los robots dentro de la camara.
  resultado -= np.mean(resultado, axis=0, keepdims=True)
  return resultado


def _checkpoint_completo(ruta: Path) -> bool:
  if not ruta.is_dir() or ruta.is_symlink() or not ruta.name.isdigit():
    return False
  marcador = ruta / "ppo_network_config.json"
  if not marcador.is_file() or marcador.is_symlink():
    return False
  try:
    contenido = json.loads(marcador.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return False
  return isinstance(contenido, dict) and "action_size" in contenido


def _ultimo_checkpoint_completo(directorio_registros: Path) -> Path | None:
  if not directorio_registros.is_dir():
    return None
  try:
    candidatos = _checkpoints_ordenados(directorio_registros)
  except (FileNotFoundError, OSError):
    return None
  for ruta in candidatos:
    if _checkpoint_completo(ruta):
      return ruta.resolve()
  return None


def _ultimo_checkpoint_de_directorio(directorio: Path) -> Path | None:
  if not directorio.is_dir():
    return None
  candidatos = [ruta for ruta in directorio.iterdir() if _checkpoint_completo(ruta)]
  if not candidatos:
    return None
  return max(candidatos, key=lambda ruta: int(ruta.name)).resolve()


def _normalizar_ruta_checkpoint(ruta: Path) -> Path:
  try:
    ruta = ruta.expanduser().resolve(strict=True)
  except OSError as exc:
    raise FileNotFoundError(f"No existe el checkpoint indicado: {ruta}") from exc

  if ruta.name == "checkpoints":
    candidato = _ultimo_checkpoint_de_directorio(ruta)
  elif (ruta / "checkpoints").is_dir():
    candidato = _ultimo_checkpoint_de_directorio(ruta / "checkpoints")
  else:
    candidato = ruta
  if candidato is None or not _checkpoint_completo(candidato):
    raise FileNotFoundError(
        f"No hay un checkpoint completo dentro de {ruta}. "
        "El visor ignora guardados que aun se estan escribiendo."
    )
  return candidato.resolve()


def _seleccionar_checkpoint_inicial(argumentos) -> tuple[Path, bool]:
  if argumentos.ruta_checkpoint:
    return _normalizar_ruta_checkpoint(Path(argumentos.ruta_checkpoint)), False

  registros = Path(argumentos.directorio_registros).expanduser().resolve()
  ultimo = _ultimo_checkpoint_completo(registros)
  if ultimo is not None:
    return ultimo, False

  if argumentos.checkpoint_respaldo:
    respaldo = _normalizar_ruta_checkpoint(Path(argumentos.checkpoint_respaldo))
    print(
        "Todavia no hay un checkpoint local completo; se mostrara el modelo "
        "de referencia hasta que aparezca uno."
    )
    return respaldo, True

  raise FileNotFoundError(
      f"No hay checkpoints completos en {registros} ni un modelo de respaldo."
  )


def _ruta_xml_portatil(configuracion) -> Path:
  ruta = Path(str(configuracion.xml_path)).expanduser()
  candidatos = [ruta]
  if not ruta.is_absolute():
    candidatos.insert(0, RAIZ_REPOSITORIO / ruta)
  candidatos.append(RAIZ_REPOSITORIO / "sim2real_mjx" / "xmls" / ruta.name)
  for candidato in candidatos:
    if candidato.is_file():
      ruta_resuelta = candidato.resolve()
      configuracion.xml_path = ruta_resuelta.as_posix()
      return ruta_resuelta
  raise FileNotFoundError(
      "El checkpoint hace referencia a un XML que no existe en este equipo: "
      f"{configuracion.xml_path}"
  )


def _firma_parametros(parametros) -> tuple[tuple[tuple[int, ...], str], ...]:
  firma: list[tuple[tuple[int, ...], str]] = []
  for hoja in jax.tree_util.tree_leaves(parametros):
    forma = tuple(int(dimension) for dimension in np.shape(hoja))
    firma.append((forma, str(getattr(hoja, "dtype", type(hoja).__name__))))
  return tuple(firma)


def _firma_archivo(ruta: Path) -> str:
  resumen = hashlib.sha256()
  with ruta.open("rb") as archivo:
    for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
      resumen.update(bloque)
  return resumen.hexdigest()


def _firma_configuracion_entorno(ruta_checkpoint: Path) -> str | None:
  if ruta_checkpoint.parent.name != "checkpoints":
    return None
  ruta_configuracion = ruta_checkpoint.parent.parent / "config_entorno.json"
  try:
    return _firma_archivo(ruta_configuracion)
  except OSError:
    return None


def _ancho_qpos_articulacion(tipo: int) -> int:
  if tipo == int(mujoco.mjtJoint.mjJNT_FREE):
    return 7
  if tipo == int(mujoco.mjtJoint.mjJNT_BALL):
    return 4
  return 1


class EvaluadorLote:
  """Compila una evaluacion vectorizada y la reutiliza entre checkpoints."""

  def __init__(
      self,
      ruta_checkpoint: Path,
      cantidad: int,
      candidatos: int,
      fps_objetivo: float,
      longitud_solicitada: int,
      semilla: int,
      solo_comprobar: bool,
  ):
    self.cantidad = cantidad
    self.candidatos = candidatos
    self.semilla = semilla
    self.directorio_checkpoints = ruta_checkpoint.parent.resolve()

    configuracion = default_config()
    _aplicar_configuracion_entorno_guardada(configuracion, ruta_checkpoint)
    configuracion.impl = "jax"
    self.ruta_xml = _ruta_xml_portatil(configuracion)
    if longitud_solicitada:
      configuracion.episode_length = longitud_solicitada
    longitud = int(configuracion.episode_length)
    if solo_comprobar:
      longitud = min(longitud, 8)
      configuracion.episode_length = longitud
    if longitud < 1:
      raise ValueError("La longitud del episodio debe ser mayor que cero.")

    self.entorno = EntornoRobotMJX(config=configuracion)
    self.modelo_fuente = self.entorno.mj_model
    self.firma_xml = _firma_archivo(self.ruta_xml)
    self.firma_configuracion = _firma_configuracion_entorno(ruta_checkpoint)
    self.longitud = longitud
    self.ctrl_dt = float(configuracion.ctrl_dt)
    self.action_repeat = int(configuracion.action_repeat)
    if self.action_repeat < 1:
      raise ValueError("action_repeat debe ser mayor que cero.")
    duracion_accion = self.ctrl_dt * self.action_repeat
    pasos_deseados = max(1, round(1.0 / (fps_objetivo * duracion_accion)))
    llamadas_episodio = max(1, longitud // self.action_repeat)
    self.pasos_por_fotograma = min(llamadas_episodio, pasos_deseados)
    self.fotogramas = max(1, llamadas_episodio // self.pasos_por_fotograma)
    self.pasos_evaluados = (
        self.fotogramas * self.pasos_por_fotograma * self.action_repeat
    )
    self.intervalo_fotograma = self.pasos_por_fotograma * duracion_accion

    self.entorno_envuelto = wrapper.wrap_for_brax_training(
        self.entorno,
        episode_length=longitud,
        action_repeat=self.action_repeat,
    )

    parametros = checkpoint.load(ruta_checkpoint.resolve())
    redes = _crear_redes_desde_checkpoint(
        ruta_checkpoint,
        self.entorno_envuelto.observation_size,
        self.entorno_envuelto.action_size,
    )
    self._crear_politica = ppo_networks.make_inference_fn(redes)
    self._firma_parametros = _firma_parametros(parametros)
    self._funcion_lote = jax.jit(self._construir_funcion_lote())
    self._parametros_iniciales = parametros

  def _construir_funcion_lote(self):
    cantidad = self.cantidad
    candidatos = self.candidatos
    fotogramas = self.fotogramas
    pasos_por_fotograma = self.pasos_por_fotograma
    entorno = self.entorno_envuelto
    crear_politica = self._crear_politica

    def evaluar(parametros, clave_maestra):
      politica = crear_politica(parametros, deterministic=True)
      clave_reinicio, clave_politica = jax.random.split(clave_maestra)
      claves_reinicio = jax.random.split(clave_reinicio, candidatos)
      claves_politica = jax.random.split(clave_politica, candidatos)
      estado = entorno.reset(claves_reinicio)
      vivos = jp.ones((candidatos,), dtype=jp.bool_)
      retornos = jp.zeros((candidatos,), dtype=jp.float32)
      longitudes = jp.zeros((candidatos,), dtype=jp.int32)
      ultimos_qpos = estado.data.qpos

      def avanzar_un_paso(carry, _):
        estado, claves, retornos, longitudes, vivos, ultimos_qpos = carry
        pares_claves = jax.vmap(jax.random.split)(claves)
        siguientes_claves = pares_claves[:, 0]
        claves_accion = pares_claves[:, 1]
        acciones, _ = politica(estado.obs, claves_accion)
        siguiente_estado = entorno.step(estado, acciones)
        recompensa = jp.nan_to_num(
            siguiente_estado.reward, nan=-1e6, posinf=1e6, neginf=-1e6
        )
        retornos = retornos + jp.where(vivos, recompensa, 0.0)
        longitudes = longitudes + vivos.astype(jp.int32)
        qpos_finito = jp.all(jp.isfinite(siguiente_estado.data.qpos), axis=-1)
        guardar_pose = vivos & (siguiente_estado.done < 0.5) & qpos_finito
        ultimos_qpos = jp.where(
            guardar_pose[:, None], siguiente_estado.data.qpos, ultimos_qpos
        )
        vivos = vivos & (siguiente_estado.done < 0.5) & qpos_finito
        return (
            siguiente_estado,
            siguientes_claves,
            retornos,
            longitudes,
            vivos,
            ultimos_qpos,
        ), None

      def generar_fotograma(carry, _):
        carry, _ = jax.lax.scan(
            avanzar_un_paso, carry, None, length=pasos_por_fotograma
        )
        return carry, carry[-1]

      carry_inicial = (
          estado,
          claves_politica,
          retornos,
          longitudes,
          vivos,
          ultimos_qpos,
      )
      carry_final, historial_qpos = jax.lax.scan(
          generar_fotograma, carry_inicial, None, length=fotogramas
      )
      retornos = carry_final[2]
      longitudes = carry_final[3]
      # Es el mismo criterio principal que usa la evaluacion de Brax: retorno
      # acumulado del episodio. Con recompensas no negativas, terminar pronto
      # queda penalizado de manera natural al dejar de sumar pasos.
      puntuaciones = retornos
      puntuaciones = jp.where(jp.isfinite(puntuaciones), puntuaciones, -jp.inf)
      mejores_puntuaciones, indices = jax.lax.top_k(puntuaciones, cantidad)
      qpos_inicial = estado.data.qpos[indices]
      mejores_qpos = jp.concatenate(
          (qpos_inicial[None, :, :], historial_qpos[:, indices, :]), axis=0
      )
      mejores_longitudes = longitudes[indices]
      return mejores_qpos, mejores_puntuaciones, mejores_longitudes

    return evaluar

  def _cargar_parametros(self, ruta_checkpoint: Path):
    parametros = checkpoint.load(ruta_checkpoint.resolve())
    if _firma_parametros(parametros) != self._firma_parametros:
      raise ValueError(
          "La arquitectura de la red ha cambiado. El visor conserva la escena "
          "anterior; reinicialo para cargar esta nueva ejecucion."
      )
    return parametros

  def generar(self, ruta_checkpoint: Path, usar_parametros_iniciales=False) -> LoteVisual:
    inicio = time.monotonic()
    parametros = (
        self._parametros_iniciales
        if usar_parametros_iniciales
        else self._cargar_parametros(ruta_checkpoint)
    )
    numero_checkpoint = int(ruta_checkpoint.name)
    clave = jax.random.fold_in(
        jax.random.PRNGKey(self.semilla), numero_checkpoint & 0x7FFFFFFF
    )
    qpos, puntuaciones, longitudes = self._funcion_lote(parametros, clave)
    qpos, puntuaciones, longitudes = jax.device_get(
        (qpos, puntuaciones, longitudes)
    )
    return LoteVisual(
        qpos=np.asarray(qpos, dtype=np.float32),
        puntuaciones=np.asarray(puntuaciones, dtype=np.float32),
        longitudes=np.asarray(longitudes, dtype=np.int32),
        ruta_checkpoint=ruta_checkpoint,
        pasos_evaluados=self.pasos_evaluados,
        segundos_calculo=time.monotonic() - inicio,
        intervalo_fotograma=self.intervalo_fotograma,
    )


class EscenaCompuesta:
  """Una sola escena sin fisica con N copias independientes del robot."""

  def __init__(self, evaluador: EvaluadorLote, cantidad: int, separacion: float):
    self.cantidad = cantidad
    self.posiciones = _posiciones_cuadricula(cantidad, separacion)
    semilado = float(np.max(np.abs(self.posiciones))) + separacion

    especificacion = mujoco.MjSpec()
    especificacion.modelname = "entrenamientos_en_directo"
    especificacion.worldbody.add_geom(
        name="suelo_visual",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[float(max(5.0, semilado)), float(max(5.0, semilado)), 0.1],
        rgba=[0.82, 0.82, 0.84, 1.0],
    )
    especificacion.visual.global_.offwidth = 1280
    especificacion.visual.global_.offheight = 720
    especificacion.visual.quality.shadowsize = 0

    for indice, (x, y) in enumerate(self.posiciones):
      hijo = mujoco.MjSpec.from_file(evaluador.ruta_xml.as_posix())
      for actuador in list(hijo.actuators):
        hijo.delete(actuador)
      for textura in list(hijo.textures):
        if (
            int(textura.type) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX)
            or str(textura.name).lower().endswith("sky")
        ):
          hijo.delete(textura)
      marco = especificacion.worldbody.add_frame(
          name=f"marco_{indice:03d}", pos=[float(x), float(y), 0.0]
      )
      cuerpo = hijo.worldbody.first_body()
      if cuerpo is None:
        raise ValueError(f"El XML {evaluador.ruta_xml} no contiene un robot.")
      marco.attach_body(cuerpo, prefix=f"entorno_{indice:03d}_")

    self.modelo = especificacion.compile()
    self.modelo.geom_contype[:] = 0
    self.modelo.geom_conaffinity[:] = 0
    try:
      self.modelo.vis.headlight.ambient[:] = (0.45, 0.45, 0.45)
      self.modelo.vis.headlight.diffuse[:] = (0.75, 0.75, 0.75)
    except AttributeError:
      pass
    self.datos = mujoco.MjData(self.modelo)
    self._indices_qpos, self._qpos_xy = self._crear_mapeo_qpos(
        evaluador.modelo_fuente
    )
    mujoco.mj_forward(self.modelo, self.datos)

  def _crear_mapeo_qpos(
      self, modelo_fuente: mujoco.MjModel
  ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    indices = np.full((self.cantidad, modelo_fuente.nq), -1, dtype=np.int32)
    qpos_xy_fuente: tuple[int, int] | None = None
    for id_articulacion in range(modelo_fuente.njnt):
      nombre = mujoco.mj_id2name(
          modelo_fuente, mujoco.mjtObj.mjOBJ_JOINT, id_articulacion
      )
      if not nombre:
        raise ValueError("Todas las articulaciones del robot deben tener nombre.")
      inicio_fuente = int(modelo_fuente.jnt_qposadr[id_articulacion])
      ancho = _ancho_qpos_articulacion(int(modelo_fuente.jnt_type[id_articulacion]))
      if int(modelo_fuente.jnt_type[id_articulacion]) == int(
          mujoco.mjtJoint.mjJNT_FREE
      ):
        qpos_xy_fuente = (inicio_fuente, inicio_fuente + 1)
      for indice in range(self.cantidad):
        nombre_destino = f"entorno_{indice:03d}_{nombre}"
        id_destino = mujoco.mj_name2id(
            self.modelo, mujoco.mjtObj.mjOBJ_JOINT, nombre_destino
        )
        if id_destino < 0:
          raise ValueError(f"No se ha podido mapear {nombre_destino}.")
        inicio_destino = int(self.modelo.jnt_qposadr[id_destino])
        indices[indice, inicio_fuente:inicio_fuente + ancho] = np.arange(
            inicio_destino, inicio_destino + ancho, dtype=np.int32
        )
    if np.any(indices < 0):
      raise ValueError("El mapeo qpos de la escena compuesta ha quedado incompleto.")
    if qpos_xy_fuente is None:
      raise ValueError("El robot necesita una articulacion libre para separar copias.")
    x_fuente, y_fuente = qpos_xy_fuente
    return indices, (indices[:, x_fuente], indices[:, y_fuente])

  def aplicar(self, qpos: np.ndarray) -> None:
    if qpos.shape != self._indices_qpos.shape:
      raise ValueError(
          f"Fotograma incompatible: {qpos.shape}; esperado {self._indices_qpos.shape}."
      )
    self.datos.qpos[:] = self.modelo.qpos0
    self.datos.qpos[self._indices_qpos] = qpos
    indices_x, indices_y = self._qpos_xy
    self.datos.qpos[indices_x] += self.posiciones[:, 0]
    self.datos.qpos[indices_y] += self.posiciones[:, 1]
    self.datos.qvel[:] = 0.0
    mujoco.mj_forward(self.modelo, self.datos)


def _describir_lote(lote: LoteVisual) -> None:
  paso = int(lote.ruta_checkpoint.name)
  mejor = float(lote.puntuaciones[0])
  peor = float(lote.puntuaciones[-1])
  print(
      f"Checkpoint {paso:,}: {len(lote.puntuaciones)} mejores intentos, "
      f"recompensa acumulada {mejor:.4f} .. {peor:.4f}, "
      f"calculado en {lote.segundos_calculo:.1f} s."
  )


def _preparar_actualizacion(
    evaluador: EvaluadorLote,
    ruta_checkpoint: Path,
    argumentos,
) -> ActualizacionPreparada:
  misma_ejecucion = (
      ruta_checkpoint.parent.resolve() == evaluador.directorio_checkpoints
  )
  misma_configuracion = (
      _firma_configuracion_entorno(ruta_checkpoint)
      == evaluador.firma_configuracion
  )
  if misma_ejecucion and misma_configuracion:
    nuevo_evaluador = evaluador
    usar_parametros_iniciales = False
  else:
    nuevo_evaluador = EvaluadorLote(
        ruta_checkpoint=ruta_checkpoint,
        cantidad=argumentos.cantidad,
        candidatos=argumentos.candidatos,
        fps_objetivo=argumentos.fps,
        longitud_solicitada=argumentos.longitud_episodio,
        semilla=argumentos.semilla,
        solo_comprobar=False,
    )
    usar_parametros_iniciales = True
  lote = nuevo_evaluador.generar(
      ruta_checkpoint, usar_parametros_iniciales=usar_parametros_iniciales
  )
  return ActualizacionPreparada(evaluador=nuevo_evaluador, lote=lote)


def _token_checkpoint(ruta: Path | None):
  if ruta is None:
    return None
  marcador = ruta / "ppo_network_config.json"
  try:
    return ruta.resolve(), marcador.stat().st_mtime_ns
  except OSError:
    return None


class ProductorActualizaciones:
  """Un unico trabajador daemon con cola latest-wins.

  La evaluacion MJX no puede cancelarse a mitad de un ejecutable XLA. Al ser
  daemon, cerrar la ventana nunca obliga al usuario a esperar a que termine
  una evaluacion que ya estaba en curso.
  """

  def __init__(self, evaluador: EvaluadorLote, argumentos):
    self._evaluador = evaluador
    self._argumentos = argumentos
    self._solicitudes: queue.Queue[tuple[object, Path]] = queue.Queue(maxsize=1)
    self._respuestas: queue.Queue[RespuestaProductor] = queue.Queue()
    self._detener = threading.Event()
    self._hilo = threading.Thread(
        target=self._trabajar,
        name="evaluacion_visor",
        daemon=True,
    )
    self._hilo.start()

  def solicitar(self, token: object, ruta_checkpoint: Path) -> None:
    # Si llegaron varios checkpoints durante una evaluacion, solo interesa el
    # mas reciente. Nunca se acumula una cola de trabajo atrasada.
    while True:
      try:
        self._solicitudes.get_nowait()
      except queue.Empty:
        break
    self._solicitudes.put_nowait((token, ruta_checkpoint))

  def recoger(self) -> RespuestaProductor | None:
    respuesta = None
    while True:
      try:
        respuesta = self._respuestas.get_nowait()
      except queue.Empty:
        return respuesta

  def cerrar(self) -> None:
    self._detener.set()
    while True:
      try:
        self._solicitudes.get_nowait()
      except queue.Empty:
        break
    self._hilo.join(timeout=0.1)

  def _trabajar(self) -> None:
    while not self._detener.is_set():
      try:
        token, ruta_checkpoint = self._solicitudes.get(timeout=0.2)
      except queue.Empty:
        continue
      try:
        preparada = _preparar_actualizacion(
            self._evaluador, ruta_checkpoint, self._argumentos
        )
        self._evaluador = preparada.evaluador
        respuesta = RespuestaProductor(token=token, preparada=preparada)
      except Exception as exc:  # La ventana debe continuar con el lote anterior.
        respuesta = RespuestaProductor(token=token, error=exc)
      self._respuestas.put(respuesta)


def _configurar_camara(visor, posiciones: np.ndarray, separacion: float) -> None:
  if posiciones.size:
    visor.cam.lookat[:] = (0.0, 0.0, 0.08)
  extension = max(np.ptp(posiciones[:, 0]), np.ptp(posiciones[:, 1]), separacion)
  visor.cam.distance = max(3.0, extension * 1.25)
  visor.cam.azimuth = 45.0
  visor.cam.elevation = -62.0
  try:
    visor.scn.flags[int(mujoco.mjtRndFlag.mjRND_SHADOW)] = 0
  except (AttributeError, IndexError):
    pass


def _reproducir(
    escena: EscenaCompuesta,
    evaluador: EvaluadorLote,
    lote_inicial: LoteVisual,
    argumentos,
) -> None:
  lote = lote_inicial
  checkpoint_mostrado = lote.ruta_checkpoint.resolve()
  checkpoint_observado = checkpoint_mostrado
  checkpoints_acumulados = 0
  token_intentado = _token_checkpoint(checkpoint_mostrado)
  siguiente_busqueda = time.monotonic() + argumentos.intervalo_busqueda
  inicio_reproduccion = time.monotonic()
  ultimo_fotograma = -1
  seguir = not argumentos.sin_seguir_checkpoints
  registros = Path(argumentos.directorio_registros).expanduser().resolve()
  productor = ProductorActualizaciones(evaluador, argumentos)

  try:
    with mujoco.viewer.launch_passive(escena.modelo, escena.datos) as visor:
      _configurar_camara(visor, escena.posiciones, argumentos.separacion)
      print("Ventana abierta. Cierrala o pulsa Ctrl+C para terminar el visor.")
      while visor.is_running():
        ahora = time.monotonic()

        respuesta = productor.recoger()
        if respuesta is not None and respuesta.token == token_intentado:
          if respuesta.error is not None:
            print(
                f"Aviso: no se pudo preparar el nuevo checkpoint: {respuesta.error}",
                file=sys.stderr,
            )
          elif respuesta.preparada is not None:
            preparada = respuesta.preparada
            if preparada.evaluador.firma_xml != evaluador.firma_xml:
              print(
                  "Aviso: el nuevo checkpoint usa otro XML. Se mantiene la "
                  "escena actual; reinicia el visor para reconstruirla.",
                  file=sys.stderr,
              )
            else:
              evaluador = preparada.evaluador
              lote = preparada.lote
              checkpoint_mostrado = lote.ruta_checkpoint.resolve()
              inicio_reproduccion = ahora
              ultimo_fotograma = -1
              _describir_lote(lote)

        if seguir and ahora >= siguiente_busqueda:
          candidato = _ultimo_checkpoint_completo(registros)
          token_candidato = _token_checkpoint(candidato)
          if candidato is not None:
            candidato_resuelto = candidato.resolve()
            if candidato_resuelto != checkpoint_observado:
              nueva_ejecucion = (
                  candidato_resuelto.parent != checkpoint_observado.parent
              )
              configuracion_cambiada = (
                  _firma_configuracion_entorno(candidato_resuelto)
                  != evaluador.firma_configuracion
              )
              checkpoint_observado = candidato_resuelto
              checkpoints_acumulados += 1
              if (
                  nueva_ejecucion
                  or configuracion_cambiada
                  or checkpoints_acumulados
                  >= argumentos.actualizar_cada_checkpoints
              ):
                checkpoints_acumulados = 0
                if (
                    candidato_resuelto != checkpoint_mostrado
                    and token_candidato != token_intentado
                ):
                  token_intentado = token_candidato
                  print(
                      f"Nuevo checkpoint {int(candidato.name):,}; preparando los "
                      f"{argumentos.cantidad} mejores en segundo plano..."
                  )
                  productor.solicitar(token_candidato, candidato)
          siguiente_busqueda = ahora + argumentos.intervalo_busqueda

        indice_fotograma = int(
            (ahora - inicio_reproduccion) / lote.intervalo_fotograma
        ) % lote.qpos.shape[0]
        if indice_fotograma != ultimo_fotograma:
          escena.aplicar(lote.qpos[indice_fotograma])
          visor.sync()
          ultimo_fotograma = indice_fotograma

        siguiente_fotograma = inicio_reproduccion + (
            int((ahora - inicio_reproduccion) / lote.intervalo_fotograma) + 1
        ) * lote.intervalo_fotograma
        time.sleep(max(0.001, min(0.02, siguiente_fotograma - time.monotonic())))
  finally:
    productor.cerrar()


def _crear_argumentos() -> argparse.Namespace:
  parser = AnalizadorArgumentos(
      description=(
          "Muestra los mejores entrenamientos en una sola ventana de MuJoCo, "
          "sin hacer interactuar los robots."
      ),
      add_help=False,
  )
  parser.add_argument(
      "-h", "--help", action="help", help="muestra esta ayuda y termina"
  )
  parser.add_argument(
      "--cantidad",
      "--cantidad-entornos",
      dest="cantidad",
      type=_entero_entre("cantidad", 1, MAXIMO_ENTORNOS_VISIBLES),
      default=None,
      help="robots visibles, entre 1 y 100; sin esta opcion aparece un menu",
  )
  parser.add_argument(
      "--candidatos",
      type=_entero_entre("candidatos", 1, MAXIMO_CANDIDATOS),
      default=None,
      help="intentos evaluados antes de seleccionar los mejores",
  )
  parser.add_argument(
      "--fps",
      type=_flotante_positivo("fps"),
      default=None,
      help="fotogramas objetivo; automatico segun la cantidad si se omite",
  )
  parser.add_argument("--directorio-registros", default="logs_sim2real_mjx")
  parser.add_argument("--checkpoint-respaldo", default=None)
  parser.add_argument("--ruta-checkpoint", default=None)
  parser.add_argument(
      "--longitud-episodio",
      type=_entero_entre("longitud-episodio", 1, 100_000),
      default=0,
  )
  parser.add_argument("--semilla", type=int, default=0)
  parser.add_argument(
      "--intervalo-busqueda",
      type=_flotante_positivo("intervalo-busqueda"),
      default=2.0,
  )
  parser.add_argument(
      "--actualizar-cada-checkpoints",
      type=_entero_entre("actualizar-cada-checkpoints", 1, 100),
      default=None,
      help=(
          "actualiza la escena cada N checkpoints; por defecto 2 para 64-100 "
          "robots y 1 para cantidades menores"
      ),
  )
  parser.add_argument(
      "--separacion", type=_flotante_positivo("separacion"), default=1.1
  )
  parser.add_argument("--sin-seguir-checkpoints", action="store_true")
  parser.add_argument("--solo-comprobar", action="store_true")
  argumentos = parser.parse_args()
  if argumentos.cantidad is None:
    argumentos.cantidad = _cantidad_por_menu()
  if argumentos.candidatos is None:
    argumentos.candidatos = _candidatos_automaticos(argumentos.cantidad)
  if argumentos.candidatos < argumentos.cantidad:
    parser.error("--candidatos no puede ser menor que --cantidad")
  if argumentos.fps is None:
    argumentos.fps = _fps_automaticos(argumentos.cantidad)
  if argumentos.actualizar_cada_checkpoints is None:
    argumentos.actualizar_cada_checkpoints = (
        _cadencia_actualizacion_automatica(argumentos.cantidad)
    )
  return argumentos


def _comprobar_backend() -> None:
  backend = jax.default_backend()
  dispositivos = jax.devices()
  acelerador = os.environ.get("SIM2REAL_ACCELERATOR", "auto")
  cpu_solicitada = os.environ.get("JAX_PLATFORM_NAME", "").lower() == "cpu"
  print(f"Backend JAX: {backend} | dispositivos: {dispositivos}")
  if acelerador in {"nvidia", "amd"} and backend == "cpu" and not cpu_solicitada:
    raise RuntimeError(
        f"Se solicito el perfil {acelerador}, pero JAX solo ve CPU. "
        "El visor no continuara ocultando este problema."
    )


def main() -> None:
  argumentos = _crear_argumentos()
  ruta_checkpoint, usando_respaldo = _seleccionar_checkpoint_inicial(argumentos)
  _comprobar_backend()
  print(
      f"Preparando {argumentos.candidatos} intentos para mostrar los "
      f"{argumentos.cantidad} mejores. Checkpoint: {ruta_checkpoint}"
  )
  if not argumentos.sin_seguir_checkpoints:
    print(
        "Actualizacion automatica: cada "
        f"{argumentos.actualizar_cada_checkpoints} checkpoint(s)."
    )
  evaluador = EvaluadorLote(
      ruta_checkpoint=ruta_checkpoint,
      cantidad=argumentos.cantidad,
      candidatos=argumentos.candidatos,
      fps_objetivo=argumentos.fps,
      longitud_solicitada=argumentos.longitud_episodio,
      semilla=argumentos.semilla,
      solo_comprobar=argumentos.solo_comprobar,
  )
  print(
      f"Muestreo: {evaluador.fotogramas + 1} poses, una cada "
      f"{evaluador.pasos_por_fotograma} pasos ({1/evaluador.intervalo_fotograma:.1f} FPS)."
  )
  print("Compilando la primera evaluacion MJX; las siguientes reutilizaran este trabajo.")
  lote = evaluador.generar(ruta_checkpoint, usar_parametros_iniciales=True)
  _describir_lote(lote)
  print("Construyendo una sola escena de MuJoCo...")
  escena = EscenaCompuesta(evaluador, argumentos.cantidad, argumentos.separacion)
  escena.aplicar(lote.qpos[0])
  print(
      f"Escena lista: {escena.modelo.nbody - 1} cuerpos, "
      f"{escena.modelo.ngeom} geometrias, {escena.modelo.nu} actuadores, "
      f"{escena.datos.ncon} contactos."
  )
  if escena.modelo.nu != 0 or escena.datos.ncon != 0:
    raise RuntimeError(
        "La escena visual no ha quedado completamente desacoplada de la fisica."
    )
  if argumentos.solo_comprobar:
    print("Comprobacion terminada correctamente; no se ha abierto ninguna ventana.")
    return
  if usando_respaldo:
    print("El visor comprobara automaticamente si aparece un checkpoint local.")
  _reproducir(escena, evaluador, lote, argumentos)


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    print("\nVisor detenido.")
  except (FileNotFoundError, RuntimeError, ValueError) as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc
