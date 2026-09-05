"""Prueba mínima de estabilidad del entorno genérico MJX."""

from __future__ import annotations

import argparse
import warnings

import jax
import jax.numpy as jp

from sim2real_mjx.entorno_robot_mjx import EntornoRobotMJX
from sim2real_mjx.entorno_robot_mjx import default_config


warnings.filterwarnings(
    "ignore",
    message="overflow encountered in cast",
    category=RuntimeWarning,
)


def _resumen_backend() -> None:
  print("Backend JAX:", jax.default_backend())
  print("Dispositivos JAX:", jax.devices())


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=150)
  parser.add_argument("--impl", choices=["jax"], default="jax")
  parser.add_argument("--iterations", type=int, default=12)
  parser.add_argument("--ls_iterations", type=int, default=4)
  parser.add_argument("--qpos_limit", type=float, default=20.0)
  parser.add_argument("--qvel_limit", type=float, default=120.0)
  args = parser.parse_args()

  _resumen_backend()
  cfg = default_config()
  cfg.impl = args.impl
  cfg.solver_iterations = args.iterations
  cfg.solver_ls_iterations = args.ls_iterations
  env = EntornoRobotMJX(config=cfg)

  print("XML:", env.xml_path)
  print("nq/nv/nu:", env.mjx_model.nq, env.mjx_model.nv, env.action_size)
  print("Observaciones:", env.observation_size)

  @jax.jit
  def rollout(mode: int, rng: jax.Array):
    state = env.reset(rng)

    def body(carry, i):
      state, rng = carry
      rng, key = jax.random.split(rng)
      zero = jp.zeros(env.action_size)
      random_small = 0.1 * jax.random.uniform(
          key, (env.action_size,), minval=-1.0, maxval=1.0
      )
      extreme = jp.where((i % 2) == 0, jp.ones(env.action_size) * 2.0, -jp.ones(env.action_size) * 2.0)
      action = jp.where(mode == 0, zero, jp.where(mode == 1, random_small, extreme))
      state = env.step(state, action)
      finite = jp.isfinite(state.data.qpos).all() & jp.isfinite(state.data.qvel).all()
      qpos_max = jp.max(jp.abs(state.data.qpos))
      qvel_max = jp.max(jp.abs(state.data.qvel))
      bad = (~finite) | (qpos_max > args.qpos_limit) | (qvel_max > args.qvel_limit)
      return (state, rng), jp.array([bad.astype(jp.float32), qpos_max, qvel_max, state.reward])

    (_, _), stats = jax.lax.scan(body, (state, rng), jp.arange(args.steps))
    return stats

  names = ("accion_cero", "accion_aleatoria_pequena", "accion_extrema_recortada")
  for mode, name in enumerate(names):
    stats = rollout(mode, jax.random.PRNGKey(100 + mode))
    stats.block_until_ready()
    bad = bool(jax.device_get(jp.any(stats[:, 0] > 0)))
    qpos_max = float(jax.device_get(jp.max(stats[:, 1])))
    qvel_max = float(jax.device_get(jp.max(stats[:, 2])))
    reward_last = float(jax.device_get(stats[-1, 3]))
    print(
        f"{name}: bad={bad}, qpos_max={qpos_max:.3f}, "
        f"qvel_max={qvel_max:.3f}, reward_last={reward_last:.3f}"
    )
    if bad:
      raise RuntimeError(f"Test MJX fallido en {name}.")

  print("test-mjx OK.")


if __name__ == "__main__":
  main()
