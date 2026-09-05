#!/usr/bin/env python3
"""Diagnostico reproducible de la pila Python/JAX/MJX de SIM2REAL."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


EXPECTED_VERSIONS = {
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


def _command_output(command: list[str]) -> str:
  try:
    return subprocess.check_output(
        command, text=True, stderr=subprocess.DEVNULL, timeout=8
    ).strip()
  except (FileNotFoundError, subprocess.SubprocessError, OSError):
    return ""


def _os_release() -> dict[str, str]:
  values: dict[str, str] = {}
  path = Path("/etc/os-release")
  if not path.exists():
    return values
  for line in path.read_text(encoding="utf-8").splitlines():
    if "=" not in line or line.startswith("#"):
      continue
    key, value = line.split("=", 1)
    values[key] = value.strip().strip('"')
  return values


def _is_wsl() -> bool:
  release = platform.release().lower()
  return bool(os.environ.get("WSL_INTEROP")) or "microsoft" in release or "wsl" in release


def _wsl_version() -> int | None:
  if not _is_wsl():
    return None
  release = platform.release().lower()
  return 2 if "microsoft-standard" in release or "wsl2" in release else 1


def _nvidia_smi_command() -> str | None:
  wsl_binary = Path("/usr/lib/wsl/lib/nvidia-smi")
  if _is_wsl() and wsl_binary.is_file() and os.access(wsl_binary, os.X_OK):
    return str(wsl_binary)
  return shutil.which("nvidia-smi")


def _package_versions() -> tuple[dict[str, str | None], list[str]]:
  versions: dict[str, str | None] = {}
  errors: list[str] = []
  for package, expected in EXPECTED_VERSIONS.items():
    try:
      installed = metadata.version(package)
    except metadata.PackageNotFoundError:
      installed = None
    versions[package] = installed
    if installed is None:
      errors.append(f"Falta el paquete {package}=={expected}")
    elif installed != expected:
      errors.append(f"{package}: esperado {expected}, instalado {installed}")
  return versions, errors


def _filesystem_type(path: Path) -> str:
  output = _command_output(
      ["findmnt", "--target", str(path), "--noheadings", "--output", "FSTYPE"]
  )
  return output.splitlines()[0].strip().lower() if output else ""


def _vendor_versions(accelerator: str) -> tuple[dict[str, str | None], list[str]]:
  expected_by_accelerator = {
      "nvidia": {
          "jax-cuda12-plugin": "0.6.2",
          "jax-cuda12-pjrt": "0.6.2",
      },
      "amd": {
          "jax-rocm7-plugin": "0.6.0",
          "jax-rocm7-pjrt": "0.6.0",
      },
      "cpu": {},
      "intel": {},
  }
  versions: dict[str, str | None] = {}
  errors: list[str] = []
  for package, expected in expected_by_accelerator[accelerator].items():
    try:
      installed = metadata.version(package)
    except metadata.PackageNotFoundError:
      installed = None
    versions[package] = installed
    if installed != expected:
      errors.append(f"{package}: esperado {expected}, instalado {installed or 'ausente'}")

  forbidden_by_accelerator = {
      "nvidia": ("jax-rocm7-plugin", "jax-rocm7-pjrt"),
      "amd": ("jax-cuda12-plugin", "jax-cuda12-pjrt"),
      "cpu": (
          "jax-cuda12-plugin",
          "jax-cuda12-pjrt",
          "jax-rocm7-plugin",
          "jax-rocm7-pjrt",
      ),
      "intel": (),
  }
  for package in forbidden_by_accelerator[accelerator]:
    try:
      installed = metadata.version(package)
    except metadata.PackageNotFoundError:
      continue
    versions[package] = installed
    errors.append(
        f"El entorno {accelerator} contiene el plugin de otro backend: "
        f"{package}=={installed}"
    )
  return versions, errors


def _runtime_checks(repo_root: Path, accelerator: str) -> tuple[dict[str, Any], list[str]]:
  result: dict[str, Any] = {}
  errors: list[str] = []
  expected_backend = {
      "nvidia": "gpu",
      "amd": "gpu",
      "intel": "xpu",
      "cpu": "cpu",
  }[accelerator]
  try:
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    devices = list(jax.devices())
    result["jax_backend"] = jax.default_backend()
    result["jax_devices"] = [str(device) for device in devices]
    result["jax_device_kinds"] = [str(getattr(device, "device_kind", "")) for device in devices]
    result["jax_platform_versions"] = sorted(
        {
            str(getattr(getattr(device, "client", None), "platform_version", ""))
            for device in devices
            if getattr(getattr(device, "client", None), "platform_version", "")
        }
    )
    if result["jax_backend"] != expected_backend:
      errors.append(
          f"Backend JAX incorrecto: esperado {expected_backend}, obtenido "
          f"{result['jax_backend']}"
      )
    if not devices:
      errors.append("JAX no enumera ningun dispositivo")

    vendor_evidence = " ".join(
        result["jax_devices"]
        + result["jax_device_kinds"]
        + result["jax_platform_versions"]
    ).lower()
    if accelerator == "nvidia" and not any(
        marker in vendor_evidence for marker in ("nvidia", "cuda")
    ):
      errors.append("JAX usa GPU, pero el dispositivo no acredita NVIDIA/CUDA")
    if accelerator == "amd" and not any(
        marker in vendor_evidence for marker in ("amd", "radeon", "rocm", "gfx")
    ):
      errors.append("JAX usa GPU, pero el dispositivo no acredita AMD/ROCm")

    @jax.jit
    def _jit_probe(value: Any) -> Any:
      return jnp.sum((value @ value) + jnp.sin(value))

    jit_value = _jit_probe(jnp.arange(64, dtype=jnp.float32).reshape(8, 8))
    jit_value.block_until_ready()
    result["jit"] = {
        "ok": True,
        "value": float(jit_value),
        "device": str(getattr(jit_value, "device", "unknown")),
    }
  except Exception as exc:  # El doctor debe informar incluso de fallos de plugin.
    result["jax_import_error"] = f"{type(exc).__name__}: {exc}"
    errors.append(f"JAX/JIT no puede inicializarse: {type(exc).__name__}: {exc}")

  for module in ("mujoco", "mujoco.mjx", "mujoco_playground", "brax", "flax"):
    try:
      importlib.import_module(module)
    except Exception as exc:
      errors.append(f"No se puede importar {module}: {type(exc).__name__}: {exc}")

  xml_path = repo_root / "sim2real_mjx" / "xmls" / "ROBOT_MJX.xml"
  result["xml"] = str(xml_path)
  try:
    mujoco = importlib.import_module("mujoco")
    mjx = importlib.import_module("mujoco.mjx")
    jax = importlib.import_module("jax")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    result["xml_model"] = {"nq": model.nq, "nv": model.nv, "nu": model.nu}
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.make_data(mjx_model)
    stepped = jax.jit(mjx.step)(mjx_model, mjx_data)
    stepped.qpos.block_until_ready()
    result["mjx_step"] = {
        "ok": True,
        "qpos_shape": list(stepped.qpos.shape),
        "time": float(stepped.time),
    }
  except Exception as exc:
    errors.append(f"El XML o el paso fisico MJX falla: {type(exc).__name__}: {exc}")
  return result, errors


def _hardware_summary() -> dict[str, Any]:
  nvidia_smi = _nvidia_smi_command()
  nvidia = _command_output([nvidia_smi, "-L"]) if nvidia_smi else ""
  lspci = _command_output(["lspci"])
  return {
      "nvidia_smi": nvidia.splitlines() if nvidia else [],
      "amd_detected": any(
          vendor in line.lower() and any(kind in line.lower() for kind in ("vga", "3d", "display"))
          for line in lspci.splitlines()
          for vendor in ("amd", "ati")
      ),
      "intel_gpu_detected": any(
          "intel" in line.lower()
          and any(kind in line.lower() for kind in ("vga", "3d", "display"))
          for line in lspci.splitlines()
      ),
      "dev_kfd": Path("/dev/kfd").exists(),
      "rocm_path": Path("/opt/rocm").exists(),
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo-root", type=Path, required=True)
  parser.add_argument("--accelerator", choices=("nvidia", "amd", "intel", "cpu"), required=True)
  parser.add_argument("--environment", type=Path, required=True)
  parser.add_argument("--support-level", default="desconocido")
  parser.add_argument("--quick", action="store_true", help="No importa JAX/MuJoCo")
  parser.add_argument("--json", action="store_true")
  args = parser.parse_args()

  repo_root = args.repo_root.resolve()
  expected_environment = args.environment.resolve()
  distro = _os_release()
  host_kind = "wsl" if _is_wsl() else "ubuntu"
  wsl_version = _wsl_version()
  filesystem_type = _filesystem_type(repo_root)
  errors: list[str] = []
  warnings: list[str] = []

  if platform.system() != "Linux" or distro.get("ID") != "ubuntu":
    errors.append("El host no es Ubuntu Linux/WSL soportado")
  if host_kind == "wsl" and wsl_version != 2:
    errors.append("WSL1 no es compatible; convierte la distribucion a WSL2")
  windows_filesystems = {"drvfs", "9p", "v9fs", "virtiofs"}
  if host_kind == "wsl" and (
      filesystem_type in windows_filesystems
      or (not filesystem_type and str(repo_root).startswith("/mnt/"))
  ):
    errors.append(
        "El repositorio esta en un filesystem Windows montado en WSL; "
        "debe vivir en el filesystem Linux"
    )
  if Path(sys.prefix).resolve() != expected_environment:
    errors.append(
        "Python no pertenece al entorno aislado solicitado: "
        f"esperado {expected_environment}, activo {Path(sys.prefix).resolve()}"
    )
  if args.accelerator == "intel":
    errors.append("Intel GPU es incompatible con la baseline JAX 0.6.2 de este repositorio")
  elif args.accelerator == "amd":
    warnings.append("AMD/ROCm es experimental y aun no esta validado por el proyecto")
  elif args.accelerator == "cpu":
    warnings.append("CPU sirve para desarrollo/smoke tests; PPO sera extremadamente lento")

  versions, version_errors = _package_versions()
  errors.extend(version_errors)
  vendor_versions, vendor_errors = _vendor_versions(args.accelerator)
  versions.update(vendor_versions)
  errors.extend(vendor_errors)
  runtime: dict[str, Any] = {}
  if not args.quick and not version_errors and args.accelerator != "intel":
    runtime, runtime_errors = _runtime_checks(repo_root, args.accelerator)
    errors.extend(runtime_errors)

  payload = {
      "ok": not errors,
      "host": host_kind,
      "wsl_version": wsl_version,
      "distro": distro.get("PRETTY_NAME", distro.get("ID", "unknown")),
      "kernel": platform.release(),
      "architecture": platform.machine(),
      "repo_root": str(repo_root),
      "filesystem_type": filesystem_type or None,
      "python": sys.version.split()[0],
      "python_executable": sys.executable,
      "environment": str(expected_environment),
      "venv": sys.prefix != sys.base_prefix,
      "accelerator": args.accelerator,
      "support_level": args.support_level,
      "hardware": _hardware_summary(),
      "versions": versions,
      "runtime": runtime,
      "warnings": warnings,
      "errors": errors,
  }

  if args.json:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
  else:
    mark = "OK" if payload["ok"] else "FALLO"
    print(f"SIM2REAL doctor: {mark}")
    print(f"  Host:        {payload['distro']} ({host_kind}, {payload['architecture']})")
    print(f"  Workspace:   {repo_root}")
    print(f"  Python:      {payload['python']} ({payload['python_executable']})")
    print(f"  Entorno:     {payload['environment']}")
    print(f"  Acelerador:  {args.accelerator}")
    print(f"  Soporte:     {args.support_level}")
    if runtime.get("jax_backend"):
      print(f"  JAX backend: {runtime['jax_backend']}")
      for device in runtime.get("jax_devices", []):
        print(f"    - {device}")
    if runtime.get("xml_model"):
      dims = runtime["xml_model"]
      print(f"  XML MJX:     OK (nq={dims['nq']}, nv={dims['nv']}, nu={dims['nu']})")
    if runtime.get("jit", {}).get("ok"):
      print(f"  JIT:         OK ({runtime['jit']['device']})")
    if runtime.get("mjx_step", {}).get("ok"):
      print("  Paso MJX:    OK")
    if warnings:
      print("  Avisos:")
      for warning in warnings:
        print(f"    - {warning}")
    if errors:
      print("  Errores:")
      for error in errors:
        print(f"    - {error}")
  return 0 if not errors else 1


if __name__ == "__main__":
  raise SystemExit(main())
