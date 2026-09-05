"""Benchmark corto de stepping MJX vectorizado."""

from __future__ import annotations

import argparse
import csv
import functools
import os
from pathlib import Path
import shutil
import subprocess
import time
import warnings

import jax
import jax.numpy as jp

from mujoco_playground import wrapper

from sim2real_mjx.entorno_robot_mjx import EntornoRobotMJX
from sim2real_mjx.entorno_robot_mjx import default_config


warnings.filterwarnings(
    "ignore",
    message="overflow encountered in cast",
    category=RuntimeWarning,
)


def _nvidia_smi_command() -> str | None:
  wsl_binary = Path("/usr/lib/wsl/lib/nvidia-smi")
  if wsl_binary.is_file() and os.access(wsl_binary, os.X_OK):
    return str(wsl_binary)
  return shutil.which("nvidia-smi")


def _vram_mib() -> str:
  """Devuelve VRAM usada sin asumir que todas las GPU son NVIDIA."""
  accelerator = os.environ.get("SIM2REAL_ACCELERATOR", "auto")
  if accelerator in {"auto", "nvidia"}:
    nvidia_smi = _nvidia_smi_command()
    try:
      if nvidia_smi is None:
        raise FileNotFoundError("nvidia-smi no disponible")
      out = subprocess.check_output(
          [
              nvidia_smi,
              "--query-gpu=memory.used",
              "--format=csv,noheader,nounits",
          ],
          text=True,
          timeout=5,
      )
      return out.strip().splitlines()[0]
    except Exception:
      if accelerator == "nvidia":
        return ""

  if accelerator in {"auto", "amd"}:
    for card in Path("/sys/class/drm").glob("card*/device"):
      try:
        if (card / "vendor").read_text(encoding="ascii").strip() != "0x1002":
          continue
        used_bytes = int((card / "mem_info_vram_used").read_text(encoding="ascii"))
        return str(used_bytes // 1024 // 1024)
      except (OSError, ValueError):
        continue
  return ""


def _append_row(csv_path: Path, row: dict[str, str]) -> None:
  csv_path.parent.mkdir(parents=True, exist_ok=True)
  exists = csv_path.exists()
  with csv_path.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if not exists:
      writer.writeheader()
    writer.writerow(row)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--csv_path", required=True)
  parser.add_argument("--num_envs", type=int, required=True)
  parser.add_argument("--warmup_steps", type=int, default=64)
  parser.add_argument("--measure_steps", type=int, default=256)
  parser.add_argument("--iterations", type=int, default=12)
  parser.add_argument("--ls_iterations", type=int, default=4)
  parser.add_argument("--matmul_precision", default="high")
  parser.add_argument("--allocator", default="preallocate")
  parser.add_argument("--impl", choices=["jax"], default="jax")
  args = parser.parse_args()

  row = {
      "num_envs": str(args.num_envs),
      "tiempo_warmup": "",
      "tiempo_medido": "",
      "steps_s_total": "",
      "steps_s_por_entorno": "",
      "vram_usada_mib": "",
      "matmul_precision": args.matmul_precision,
      "allocator": args.allocator,
      "iterations": str(args.iterations),
      "ls_iterations": str(args.ls_iterations),
      "nan_inf": "false",
      "oom": "false",
      "error": "",
  }

  try:
    print("Backend JAX:", jax.default_backend())
    print("Dispositivos JAX:", jax.devices())
    cfg = default_config()
    cfg.impl = args.impl
    cfg.solver_iterations = args.iterations
    cfg.solver_ls_iterations = args.ls_iterations
    env = EntornoRobotMJX(config=cfg)
    env = wrapper.wrap_for_brax_training(
        env,
        episode_length=cfg.episode_length,
        action_repeat=cfg.action_repeat,
    )

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    @functools.partial(jax.jit, static_argnames=("steps",))
    def correr(state, rng, steps: int):
      def body(carry, _):
        state, rng = carry
        rng, key = jax.random.split(rng)
        action = jax.random.uniform(
            key,
            (args.num_envs, env.action_size),
            minval=-1.0,
            maxval=1.0,
        )
        state = step(state, action)
        return (state, rng), None

      return jax.lax.scan(body, (state, rng), xs=None, length=steps)[0]

    rng = jax.random.PRNGKey(0)
    reset_keys = jax.random.split(rng, args.num_envs)
    state = reset(reset_keys)
    state.obs["state"].block_until_ready()

    t0 = time.perf_counter()
    state, rng = correr(state, rng, args.warmup_steps)
    state.obs["state"].block_until_ready()
    warmup_time = time.perf_counter() - t0

    state, rng = correr(state, rng, args.measure_steps)
    state.obs["state"].block_until_ready()

    t0 = time.perf_counter()
    state, rng = correr(state, rng, args.measure_steps)
    state.obs["state"].block_until_ready()
    measured_time = time.perf_counter() - t0

    finite = jp.isfinite(state.data.qpos).all() & jp.isfinite(state.data.qvel).all()
    nan_inf = not bool(jax.device_get(finite))
    total_steps = args.measure_steps * args.num_envs
    steps_s = total_steps / measured_time

    row.update({
        "tiempo_warmup": f"{warmup_time:.6f}",
        "tiempo_medido": f"{measured_time:.6f}",
        "steps_s_total": f"{steps_s:.3f}",
        "steps_s_por_entorno": f"{(steps_s / args.num_envs):.6f}",
        "vram_usada_mib": _vram_mib(),
        "nan_inf": str(nan_inf).lower(),
    })
    if nan_inf:
      raise RuntimeError("NaNs/Infs tras benchmark.")
  except RuntimeError as exc:
    message = str(exc)
    row["error"] = message[:300]
    if "RESOURCE_EXHAUSTED" in message or "out of memory" in message.lower() or "oom" in message.lower():
      row["oom"] = "true"
    print("Benchmark fallido:", message)
  finally:
    _append_row(Path(args.csv_path), row)
    print("Fila benchmark:", row)


if __name__ == "__main__":
  main()
