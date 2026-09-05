"""Hiperparámetros PPO para el entorno genérico del robot.

Este archivo concentra lo que normalmente tocarias al ajustar entrenamiento:
numero de entornos, batch, red actor-critic y escalas generales de PPO. La
implementacion de PPO sigue siendo la oficial de Brax.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ml_collections import config_dict


VERSIONES_ESPERADAS: dict[str, str] = {
    "jax": "0.6.2",
    "jaxlib": "0.6.2",
    "mujoco": "3.6.0",
    "mujoco-mjx": "3.6.0",
    "brax": "0.14.2",
    "flax": "0.11.2",
    "optax": "0.2.6",
    "orbax-checkpoint": "0.11.31",
    "ml_collections": "1.1.0",
    "warp-lang": "1.11.0",
}


def _crear_configuracion_base_ppo(
    *,
    num_timesteps: int,
    num_evals: int,
    num_envs: int,
    num_eval_envs: int,
    episode_length: int,
    action_repeat: int,
    unroll_length: int,
    batch_size: int,
    num_minibatches: int,
    num_updates_per_batch: int,
    discounting: float,
    learning_rate: float,
    entropy_cost: float,
    clipping_epsilon: float,
    gae_lambda: float,
    max_grad_norm: float,
    normalize_observations: bool,
    reward_scaling: float,
    run_evals: bool,
    num_resets_per_eval: int,
    log_training_metrics: bool,
    reward_log_policy_updates_interval: int,
    training_metrics_steps: int | None,
    policy_hidden_layer_sizes: tuple[int, ...],
    value_hidden_layer_sizes: tuple[int, ...],
) -> config_dict.ConfigDict:
  """Construye una configuracion PPO con red actor-critic simetrica."""

  return config_dict.create(
      num_timesteps=num_timesteps,
      num_evals=num_evals,
      num_envs=num_envs,
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      unroll_length=unroll_length,
      batch_size=batch_size,
      num_minibatches=num_minibatches,
      num_updates_per_batch=num_updates_per_batch,
      discounting=discounting,
      learning_rate=learning_rate,
      entropy_cost=entropy_cost,
      clipping_epsilon=clipping_epsilon,
      gae_lambda=gae_lambda,
      max_grad_norm=max_grad_norm,
      normalize_observations=normalize_observations,
      reward_scaling=reward_scaling,
      run_evals=run_evals,
      num_resets_per_eval=num_resets_per_eval,
      log_training_metrics=log_training_metrics,
      reward_log_policy_updates_interval=reward_log_policy_updates_interval,
      training_metrics_steps=training_metrics_steps,
      network_factory=config_dict.create(
          policy_hidden_layer_sizes=policy_hidden_layer_sizes,
          value_hidden_layer_sizes=value_hidden_layer_sizes,
          policy_obs_key="state",
          value_obs_key="state",
          # Nombre de la funcion de activacion para actor y critic.
          # Opciones: "swish" (SiLU), "tanh", "relu". Swish es el valor recomendado.
          activacion_red="swish",
      ),
  )


# Diagnóstico:
#   Perfil ultrarrapido para comprobar compilacion y tendencia inicial.
#   No sirve para conclusiones finales.
#
# ligero:
#   Perfil principal actual. Reduce pasos, acorta episodios, reduce
#   evaluaciones y usa una red mas pequena para pose estatica/semi-estatica.
#
# ligero_rapido:
#   Igual que ligero, pero con 1024 entornos. Usalo si la GPU lo soporta.
#
# completo:
#   Configuración pesada anterior. Úsala cuando la recompensa y la fase ya estén validadas.
# Pendiente: más adelante se podría probar value_obs_key="privileged_state".


def configuracion_ppo_depuracion() -> config_dict.ConfigDict:
  """Perfil rapido para validar compilacion y tendencia inicial de recompensa."""

  return _crear_configuracion_base_ppo(
      num_timesteps=5_000_000,
      num_evals=20,
      num_envs=512,
      num_eval_envs=4,
      episode_length=1000,
      action_repeat=1,
      unroll_length=20,
      batch_size=256,
      num_minibatches=4,
      num_updates_per_batch=1,
      discounting=0.95,
      learning_rate=3e-4,
      entropy_cost=0.015,
      clipping_epsilon=0.2,
      gae_lambda=0.95,
      max_grad_norm=1.0,
      normalize_observations=True,
      reward_scaling=1.0,
      run_evals=True,
      num_resets_per_eval=0,
      log_training_metrics=False,
      reward_log_policy_updates_interval=10,
      training_metrics_steps=None,
      policy_hidden_layer_sizes=(256, 128),
      value_hidden_layer_sizes=(256, 128),
  )


def configuracion_ppo_ligera() -> config_dict.ConfigDict:
  """Perfil principal para iterar pose estable/semi-estatica con menor coste."""

  return _crear_configuracion_base_ppo(
      num_timesteps=100_000_000,
      num_evals=200,
      num_envs=512,
      num_eval_envs=4,
      episode_length=1500,
      action_repeat=1,
      unroll_length=24,
      batch_size=256,
      num_minibatches=4,
      num_updates_per_batch=2,
      discounting=0.95,
      learning_rate=3e-4,
      entropy_cost=0.01,
      clipping_epsilon=0.2,
      gae_lambda=0.95,
      max_grad_norm=1.0,
      normalize_observations=True,
      reward_scaling=1.0,
      run_evals=True,
      num_resets_per_eval=0,
      log_training_metrics=False,
      reward_log_policy_updates_interval=10,
      training_metrics_steps=None,
      policy_hidden_layer_sizes=(256, 256),
      value_hidden_layer_sizes=(256, 256),
  )


def configuracion_ppo_ligera_rapida() -> config_dict.ConfigDict:
  """Perfil ligero con 1024 entornos para exprimir mas paralelismo MJX."""

  return _crear_configuracion_base_ppo(
      num_timesteps=100_000_000,
      num_evals=200,
      num_envs=1024,
      num_eval_envs=4,
      episode_length=1500,
      action_repeat=1,
      unroll_length=24,
      batch_size=256,
      num_minibatches=4,
      num_updates_per_batch=2,
      discounting=0.95,
      learning_rate=3e-4,
      entropy_cost=0.01,
      clipping_epsilon=0.2,
      gae_lambda=0.95,
      max_grad_norm=1.0,
      normalize_observations=True,
      reward_scaling=1.0,
      run_evals=True,
      num_resets_per_eval=0,
      log_training_metrics=False,
      reward_log_policy_updates_interval=10,
      training_metrics_steps=None,
      policy_hidden_layer_sizes=(256, 256),
      value_hidden_layer_sizes=(256, 256),
  )


def configuracion_ppo_completa() -> config_dict.ConfigDict:
  """Configuracion pesada para ejecuciones finales."""

  return _crear_configuracion_base_ppo(
      # 512 entornos reducen la sobrecarga de PPO de ~53 % a ~25 %.
      # Velocidad estimada: unos 800 pasos/s.
      # episode_length 3000 equivale a 12 s reales con ctrl_dt=0.004 s.
      # num_updates_per_batch=2 es suficiente para una tarea de postura estática.
      num_timesteps=50_000_000,
      num_evals=200,
      num_envs=512,
      num_eval_envs=2,
      episode_length=3000,
      action_repeat=1,
      unroll_length=20,
      batch_size=256,
      num_minibatches=8,
      num_updates_per_batch=2,
      discounting=0.97,
      learning_rate=3e-4,
      entropy_cost=0.02,
      clipping_epsilon=0.2,
      gae_lambda=0.95,
      max_grad_norm=1.0,
      normalize_observations=True,
      reward_scaling=1.0,
      run_evals=True,
      num_resets_per_eval=0,
      log_training_metrics=False,
      reward_log_policy_updates_interval=5,
      training_metrics_steps=81_920,
      policy_hidden_layer_sizes=(512, 256, 128),
      value_hidden_layer_sizes=(512, 256, 128),
  )


def configuracion_a_diccionario(obj: Any) -> Any:
  """Convierte ConfigDict/tuplas a tipos JSON simples."""

  if isinstance(obj, config_dict.ConfigDict):
    return {key: configuracion_a_diccionario(value) for key, value in obj.items()}
  if isinstance(obj, Mapping):
    return {key: configuracion_a_diccionario(value) for key, value in obj.items()}
  if isinstance(obj, tuple):
    return [configuracion_a_diccionario(value) for value in obj]
  if isinstance(obj, list):
    return [configuracion_a_diccionario(value) for value in obj]
  return obj
