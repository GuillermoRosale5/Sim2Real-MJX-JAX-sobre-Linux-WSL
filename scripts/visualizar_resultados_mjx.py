from __future__ import annotations

import argparse
from collections.abc import Mapping
import functools
import json
from pathlib import Path
import re
import sys
import time
import warnings

try:
  import flax.linen as nn
  import jax
  import jax.numpy as jp
  import mujoco
  import mujoco.viewer
  from brax.training import checkpoint
  from brax.training.agents.ppo import networks as ppo_networks
except ModuleNotFoundError as exc:
  if exc.name not in {"jax", "mujoco", "brax"}:
    raise
  print(
      f"Falta {exc.name} en este Python. Usa el entorno del repositorio desde Ubuntu/WSL:\n"
      "  ./scripts/visualizar_ultimo_checkpoint.sh",
      file=sys.stderr,
  )
  raise SystemExit(2) from exc

from mujoco_playground import wrapper

from sim2real_mjx.hiperparametros import configuracion_ppo_ligera
from sim2real_mjx.entorno_robot_mjx import EntornoRobotMJX
from sim2real_mjx.entorno_robot_mjx import default_config


warnings.filterwarnings(
    "ignore",
    message="overflow encountered in cast",
    category=RuntimeWarning,
)


def _inferir_tamanos_capas_ocultas(arbol) -> tuple[int, ...] | None:
  if isinstance(arbol, (list, tuple)):
    for elemento in arbol:
      resultado_anidado = _inferir_tamanos_capas_ocultas(elemento)
      if resultado_anidado:
        return resultado_anidado
    return None
  if not isinstance(arbol, Mapping):
    return None

  capas_ocultas: list[tuple[int, int]] = []
  for clave, valor in arbol.items():
    if isinstance(clave, str):
      coincidencia = re.fullmatch(r"hidden_(\d+)", clave)
      if coincidencia and isinstance(valor, Mapping) and "kernel" in valor:
        nucleo = valor["kernel"]
        if hasattr(nucleo, "shape") and len(nucleo.shape) == 2:
          capas_ocultas.append(
              (int(coincidencia.group(1)), int(nucleo.shape[1]))
          )
    resultado_anidado = _inferir_tamanos_capas_ocultas(valor)
    if resultado_anidado:
      return resultado_anidado

  if capas_ocultas:
    capas_ocultas.sort(key=lambda elemento: elemento[0])
    if len(capas_ocultas) == 1:
      return tuple(tamano for _, tamano in capas_ocultas)
    return tuple(tamano for _, tamano in capas_ocultas[:-1])
  return None


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


def _ultimo_checkpoint(directorio_registros: Path) -> Path:
  ejecucion_indicada = _ultima_ejecucion_desde_puntero(directorio_registros)
  if ejecucion_indicada is not None:
    directorio_checkpoints = ejecucion_indicada / "checkpoints"
    if directorio_checkpoints.is_dir():
      candidatos = [
          checkpoint_ruta for checkpoint_ruta in directorio_checkpoints.iterdir()
          if checkpoint_ruta.is_dir() and checkpoint_ruta.name.isdigit()
      ]
      if candidatos:
        return max(candidatos, key=lambda ruta: int(ruta.name))

  candidatos: list[Path] = []
  registros_resueltos = directorio_registros.resolve()
  for raiz_checkpoints in directorio_registros.glob("*/checkpoints"):
    directorio_ejecucion = raiz_checkpoints.parent
    if (
        directorio_ejecucion.is_symlink()
        or directorio_ejecucion.resolve().parent != registros_resueltos
        or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", directorio_ejecucion.name
        )
    ):
      continue
    for ruta_checkpoint in raiz_checkpoints.iterdir():
      if (
          ruta_checkpoint.is_dir()
          and not ruta_checkpoint.is_symlink()
          and ruta_checkpoint.name.isdigit()
      ):
        candidatos.append(ruta_checkpoint)
  if not candidatos:
    raise FileNotFoundError(f"No hay checkpoints en {directorio_registros}.")
  return max(
      candidatos,
      key=lambda ruta: max(
          ruta.parent.parent.stat().st_mtime, ruta.stat().st_mtime
      ),
  )


def _checkpoints_ordenados(directorio_registros: Path) -> list[Path]:
  ejecucion_indicada = _ultima_ejecucion_desde_puntero(directorio_registros)
  if ejecucion_indicada is not None:
    directorio_checkpoints = ejecucion_indicada / "checkpoints"
    if directorio_checkpoints.is_dir():
      candidatos = [
          ruta for ruta in directorio_checkpoints.iterdir()
          if ruta.is_dir() and ruta.name.isdigit()
      ]
      if candidatos:
        return sorted(candidatos, key=lambda ruta: int(ruta.name), reverse=True)

  candidatos: list[Path] = []
  for raiz_checkpoints in directorio_registros.glob("*/checkpoints"):
    for ruta_checkpoint in raiz_checkpoints.iterdir():
      if ruta_checkpoint.is_dir() and ruta_checkpoint.name.isdigit():
        candidatos.append(ruta_checkpoint)
  return sorted(
      candidatos,
      key=lambda ruta: max(
          ruta.parent.parent.stat().st_mtime, ruta.stat().st_mtime
      ),
      reverse=True,
  )


def _checkpoint_por_indice(
    directorio_registros: Path, indice_desde_ultimo: int
) -> Path:
  if indice_desde_ultimo < 0:
    raise ValueError("checkpoint_index debe ser >= 0")
  checkpoints = _checkpoints_ordenados(directorio_registros)
  if not checkpoints:
    raise FileNotFoundError(f"No hay checkpoints en {directorio_registros}.")
  if indice_desde_ultimo >= len(checkpoints):
    raise FileNotFoundError(
        f"Solo hay {len(checkpoints)} checkpoints disponibles en "
        f"{directorio_registros}."
    )
  return checkpoints[indice_desde_ultimo]


def _directorio_ejecucion_desde_checkpoint(ruta_checkpoint: Path) -> Path | None:
  if ruta_checkpoint.parent.name == "checkpoints":
    return ruta_checkpoint.parent.parent
  if ruta_checkpoint.name == "checkpoints":
    return ruta_checkpoint.parent
  if (ruta_checkpoint / "checkpoints").is_dir():
    return ruta_checkpoint
  return None


def _aplicar_configuracion_entorno_guardada(
    configuracion_entorno, ruta_checkpoint: Path
) -> None:
  directorio_ejecucion = _directorio_ejecucion_desde_checkpoint(ruta_checkpoint)
  if directorio_ejecucion is None:
    return
  ruta_configuracion = directorio_ejecucion / "config_entorno.json"
  if not ruta_configuracion.exists():
    return
  try:
    configuracion_guardada = json.loads(
        ruta_configuracion.read_text(encoding="utf-8")
    )
  except (OSError, json.JSONDecodeError) as exc:
    print(f"Aviso: no pude leer {ruta_configuracion}: {exc}", file=sys.stderr)
    return
  for clave, valor in configuracion_guardada.items():
    configuracion_entorno[clave] = valor


_ACTIVACIONES_DISPONIBLES: dict = {
    "swish": nn.swish,
    "silu": nn.swish,
    "tanh": nn.tanh,
    "relu": nn.relu,
    "elu": nn.elu,
}


def _crear_fabrica_redes(ppo_params):
  """Crea la network factory extrayendo activacion_red del config sin pasarla
  como kwarg invalido a make_ppo_networks."""
  factory_kwargs = dict(ppo_params.network_factory)
  nombre_activacion = factory_kwargs.pop("activacion_red", "swish")
  activacion = _ACTIVACIONES_DISPONIBLES.get(nombre_activacion, nn.swish)
  factory_kwargs["activation"] = activacion
  return functools.partial(ppo_networks.make_ppo_networks, **factory_kwargs)


def _aplicar_pose_inicial(
    configuracion_entorno, pose_inicial: str | None
) -> None:
  if not pose_inicial or pose_inicial == "actual":
    return
  if pose_inicial == "suelo2":
    configuracion_entorno.reset_pose_mode = "suelo2"
    configuracion_entorno.reset_randomize_state = False
    configuracion_entorno.reset_use_joint_base_pose = True
    configuracion_entorno.reset_q1_base_deg = 0.0
    # El XML ideal tiene la pose horneada: qpos0 equivale a q2=+28.26,
    # q3=-93.261 en el modelo base. Aplicamos el delta para aproximar suelo2.
    configuracion_entorno.reset_q2_base_deg = -28.26
    configuracion_entorno.reset_q3_base_deg = 87.261
    configuracion_entorno.reset_project_feet_to_floor = True
    configuracion_entorno.reset_foot_ground_margin = 0.003
    configuracion_entorno.reset_body_z_base = 0.036
    configuracion_entorno.reset_body_xy_noise = 0.0
    configuracion_entorno.reset_body_z_noise = 0.0
    configuracion_entorno.reset_roll_pitch_deg = 0.0
    configuracion_entorno.reset_yaw_deg = 0.0
    configuracion_entorno.reset_q1_noise_deg = 0.0
    configuracion_entorno.reset_q2_noise_deg = 0.0
    configuracion_entorno.reset_q3_noise_deg = 0.0
    configuracion_entorno.reset_noise_scale = 0.0
    configuracion_entorno.reset_step_count_max = 0
    configuracion_entorno.reset_episode_length_jitter_steps = 0
    configuracion_entorno.reset_grace_step_jitter_steps = 0
  elif pose_inicial == "ideal":
    configuracion_entorno.reset_pose_mode = "default"
    configuracion_entorno.reset_randomize_state = False
    configuracion_entorno.reset_use_joint_base_pose = False
    configuracion_entorno.reset_project_feet_to_floor = False
    configuracion_entorno.reset_body_z_base = -1.0
    configuracion_entorno.reset_noise_scale = 0.0
    configuracion_entorno.reset_step_count_max = 0
    configuracion_entorno.reset_episode_length_jitter_steps = 0
    configuracion_entorno.reset_grace_step_jitter_steps = 0
  elif pose_inicial in ("caida_lateral", "boca_abajo"):
    configuracion_entorno.reset_pose_mode = pose_inicial
    configuracion_entorno.disable_done = True
    configuracion_entorno.reset_randomize_state = True
    configuracion_entorno.reset_use_joint_base_pose = False
    configuracion_entorno.reset_project_feet_to_floor = False
    configuracion_entorno.reset_body_z_base = 0.63
    configuracion_entorno.reset_body_xy_noise = 0.060
    configuracion_entorno.reset_body_z_noise = 0.060
    configuracion_entorno.reset_roll_pitch_deg = 70.0
    configuracion_entorno.reset_yaw_deg = 180.0
    configuracion_entorno.reset_q1_noise_deg = 45.0
    configuracion_entorno.reset_q2_noise_deg = 50.0
    configuracion_entorno.reset_q3_noise_deg = 50.0
    configuracion_entorno.reset_noise_scale = 0.0
    configuracion_entorno.reset_step_count_max = 0
  else:
    raise ValueError(f"pose_inicial no reconocida: {pose_inicial}")


def _cargar_politica(
    ruta_checkpoint: Path,
    impl: str,
    longitud_episodio: int,
    ruta_xml: str | None,
    pose_inicial: str | None,
):
  configuracion_entorno = default_config()
  _aplicar_configuracion_entorno_guardada(configuracion_entorno, ruta_checkpoint)
  if ruta_xml:
    xml = Path(ruta_xml)
    if not xml.is_absolute():
      xml = (Path.cwd() / xml).resolve()
    if not xml.exists():
      raise FileNotFoundError(f"No existe el XML seleccionado: {xml}")
    configuracion_entorno.xml_path = xml.as_posix()
  _aplicar_pose_inicial(configuracion_entorno, pose_inicial)
  configuracion_entorno.impl = impl
  if longitud_episodio:
    configuracion_entorno.episode_length = longitud_episodio
  entorno = EntornoRobotMJX(config=configuracion_entorno)

  parametros_ppo = configuracion_ppo_ligera()
  if longitud_episodio:
    parametros_ppo.episode_length = longitud_episodio

  entorno_envuelto = wrapper.wrap_for_brax_training(
      entorno,
      episode_length=parametros_ppo.episode_length,
      action_repeat=parametros_ppo.action_repeat,
  )

  # Usa _crear_fabrica_redes para filtrar activacion_red (no es un argumento
  # válido de make_ppo_networks) y asociarla al objeto nn.* correcto.
  fabrica_redes = _crear_fabrica_redes(parametros_ppo)
  parametros = checkpoint.load(ruta_checkpoint.resolve())
  tamanos_capas_ocultas = _inferir_tamanos_capas_ocultas(parametros)
  if tamanos_capas_ocultas:
    nombre_activacion = getattr(
        parametros_ppo.network_factory, "activacion_red", "swish"
    )
    activacion = _ACTIVACIONES_DISPONIBLES.get(nombre_activacion, nn.swish)
    fabrica_redes = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tamanos_capas_ocultas,
        value_hidden_layer_sizes=tamanos_capas_ocultas,
        policy_obs_key=parametros_ppo.network_factory.policy_obs_key,
        value_obs_key=parametros_ppo.network_factory.value_obs_key,
        activation=activacion,
    )

  red_ppo = fabrica_redes(entorno_envuelto.observation_size, entorno_envuelto.action_size)
  crear_politica = ppo_networks.make_inference_fn(red_ppo)
  politica = jax.jit(crear_politica(parametros, deterministic=True))
  return entorno, entorno_envuelto, politica, parametros_ppo


def _copiar_estado_a_datos_mujoco(estado, datos_mj: mujoco.MjData) -> None:
  qpos = jax.device_get(estado.data.qpos)
  qvel = jax.device_get(estado.data.qvel)
  ctrl = jax.device_get(estado.data.ctrl)
  if qpos.ndim > 1:
    qpos = qpos[0]
  if qvel.ndim > 1:
    qvel = qvel[0]
  if ctrl is not None and ctrl.ndim > 1:
    ctrl = ctrl[0]
  datos_mj.qpos[:] = qpos
  datos_mj.qvel[:] = qvel
  if ctrl is not None and datos_mj.ctrl.size:
    datos_mj.ctrl[:] = ctrl


def _reiniciar_episodio(funcion_reinicio, aleatorio, datos, entorno):
  aleatorio, clave_reinicio = jax.random.split(aleatorio)
  estado = funcion_reinicio(jax.random.split(clave_reinicio, 1))
  _copiar_estado_a_datos_mujoco(estado, datos)
  mujoco.mj_forward(entorno.mj_model, datos)
  return aleatorio, estado


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--impl", choices=["jax", "warp"], default="jax")
  parser.add_argument("--directorio-registros", default="logs_sim2real_mjx")
  parser.add_argument("--ruta-checkpoint", default=None)
  parser.add_argument("--indice-checkpoint", type=int, default=0)
  parser.add_argument("--longitud-episodio", type=int, default=0)
  parser.add_argument("--espera", type=float, default=0.01)
  parser.add_argument("--pausa-episodio", type=float, default=1.0)
  parser.add_argument("--semilla", type=int, default=0)
  parser.add_argument("--ruta-xml", default=None)
  parser.add_argument(
      "--postura-inicial",
      choices=["actual", "suelo2", "ideal", "caida_lateral", "boca_abajo"],
      default="actual",
  )
  parser.add_argument("--reinicio-automatico", action="store_true")
  parser.add_argument("--congelar-al-terminar", action="store_true")
  parser.add_argument("--solo-comprobar", action="store_true")
  argumentos = parser.parse_args()

  if not argumentos.congelar_al_terminar:
    argumentos.reinicio_automatico = True

  print("Backend JAX:", jax.default_backend())
  print("Dispositivos JAX:", jax.devices())

  ruta_checkpoint = (
      Path(argumentos.ruta_checkpoint)
      if argumentos.ruta_checkpoint
      else _checkpoint_por_indice(
          Path(argumentos.directorio_registros), argumentos.indice_checkpoint
      )
  )
  print("Checkpoint:", ruta_checkpoint.resolve())
  entorno, entorno_envuelto, politica, parametros_ppo = _cargar_politica(
      ruta_checkpoint,
      argumentos.impl,
      argumentos.longitud_episodio,
      argumentos.ruta_xml,
      argumentos.postura_inicial,
  )
  print("XML:", entorno.xml_path)
  print("Pose inicial:", argumentos.postura_inicial)

  modelo = mujoco.MjModel.from_xml_path(entorno.xml_path)
  datos = mujoco.MjData(modelo)
  reiniciar = jax.jit(entorno_envuelto.reset)
  avanzar = jax.jit(entorno_envuelto.step)
  desactivar_finalizacion = argumentos.postura_inicial in (
      "caida_lateral",
      "boca_abajo",
  )

  @jax.jit
  def avanzar_episodio(estado, aleatorio):
    aleatorio, clave_accion = jax.random.split(aleatorio)
    accion, _ = politica(estado.obs, clave_accion)
    estado_siguiente = avanzar(estado, accion)
    if desactivar_finalizacion:
      estado_siguiente = estado_siguiente.replace(
          done=jp.zeros_like(estado_siguiente.done)
      )
    recompensa = estado_siguiente.reward[0]
    finalizado = estado_siguiente.done[0]
    z = estado_siguiente.data.xpos[0, entorno._main_body_id, 2]
    return estado_siguiente, aleatorio, jp.array([recompensa, finalizado, z])

  aleatorio = jax.random.PRNGKey(argumentos.semilla)
  aleatorio, estado = _reiniciar_episodio(
      reiniciar, aleatorio, datos, entorno
  )

  if argumentos.solo_comprobar:
    estado, aleatorio, _ = avanzar_episodio(estado, aleatorio)
    _copiar_estado_a_datos_mujoco(estado, datos)
    mujoco.mj_forward(modelo, datos)
    print("Prueba en seco de visualizar-resultados correcta.")
    return

  print("Abriendo mujoco.viewer. Cierra la ventana para terminar.")
  reinicio_manual_pendiente = False

  def al_pulsar_tecla(codigo_tecla: int) -> None:
    nonlocal reinicio_manual_pendiente
    if codigo_tecla in (ord("r"), ord("R")):
      reinicio_manual_pendiente = True

  with mujoco.viewer.launch_passive(
      modelo, datos, key_callback=al_pulsar_tecla
  ) as visor:
    paso_episodio = 0
    retorno_episodio = 0.0
    episodio_finalizado = False
    instante_fin_episodio = 0.0
    while visor.is_running():
      inicio_bucle = time.time()
      if reinicio_manual_pendiente:
        aleatorio, estado = _reiniciar_episodio(
            reiniciar, aleatorio, datos, entorno
        )
        paso_episodio = 0
        retorno_episodio = 0.0
        episodio_finalizado = False
        instante_fin_episodio = 0.0
        reinicio_manual_pendiente = False

      if episodio_finalizado:
        if (
            argumentos.reinicio_automatico
            and time.time() - instante_fin_episodio
            >= argumentos.pausa_episodio
        ):
          aleatorio, estado = _reiniciar_episodio(
              reiniciar, aleatorio, datos, entorno
          )
          paso_episodio = 0
          retorno_episodio = 0.0
          episodio_finalizado = False
          instante_fin_episodio = 0.0
        visor.sync()
      else:
        estado, aleatorio, estadisticas = avanzar_episodio(estado, aleatorio)
        recompensa, valor_finalizado, valor_z = jax.device_get(estadisticas)
        paso_episodio += 1
        retorno_episodio += float(recompensa)
        finalizado = bool(valor_finalizado)
        limite_temporal = paso_episodio >= parametros_ppo.episode_length
        if finalizado or (limite_temporal and not desactivar_finalizacion):
          print(
              f"Episodio terminado: pasos={paso_episodio}, "
              f"retorno={retorno_episodio:.3f}, z={float(valor_z):.3f}, "
              f"finalizado={finalizado}",
              flush=True,
          )
          episodio_finalizado = True
          instante_fin_episodio = time.time()
        _copiar_estado_a_datos_mujoco(estado, datos)
        mujoco.mj_forward(modelo, datos)
        visor.sync()

      tiempo_transcurrido = time.time() - inicio_bucle
      if argumentos.espera > tiempo_transcurrido:
        time.sleep(argumentos.espera - tiempo_transcurrido)


if __name__ == "__main__":
  main()
