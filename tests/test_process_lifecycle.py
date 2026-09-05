"""Pruebas hermeticas del ciclo de vida de entrenamientos y curriculo."""

from __future__ import annotations

import ast
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, content: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
  path.chmod(0o755)


class ProcessLifecycleTests(unittest.TestCase):

  def test_python_state_updates_are_atomic_and_preserve_configuration(self) -> None:
    trainer_path = ROOT / "sim2real_mjx/entrenar_ppo_mjx.py"
    parsed = ast.parse(trainer_path.read_text(encoding="utf-8"))
    selected = [
        node
        for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and getattr(node, "name", "") in {"_guardar_json", "_guardar_estado"}
    ]
    self.assertEqual([node.name for node in selected], ["_guardar_json", "_guardar_estado"])
    namespace = {
        "Any": Any,
        "Path": Path,
        "dt": dt,
        "json": json,
        "os": os,
        "tempfile": tempfile,
    }
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(trainer_path), "exec"), namespace)

    with tempfile.TemporaryDirectory() as temporary:
      state_path = Path(temporary) / "estado.json"
      namespace["_guardar_estado"](
          state_path,
          "entrenando",
          {
              "perfil_ppo": "ligero",
              "fase_curriculum_recompensa": 2,
              "nombre_curriculum_recompensa": "llegar_desde_suelo",
          },
      )
      namespace["_guardar_estado"](
          state_path, "cancelado", {"motivo_cancelacion": "prueba"}
      )
      state = json.loads(state_path.read_text(encoding="utf-8"))
      self.assertEqual(state["estado"], "cancelado")
      self.assertEqual(state["perfil_ppo"], "ligero")
      self.assertEqual(state["fase_curriculum_recompensa"], 2)
      self.assertEqual(state["nombre_curriculum_recompensa"], "llegar_desde_suelo")
      self.assertEqual(state["motivo_cancelacion"], "prueba")
      self.assertFalse(list(state_path.parent.glob(f".{state_path.name}.*.tmp")))

  def test_stop_cancels_a_training_that_is_still_starting(self) -> None:
    launcher = ROOT / "scripts/sim2real.sh"
    with tempfile.TemporaryDirectory() as temporary:
      fake_repo = Path(temporary) / "repo"
      run_dir = fake_repo / "logs_sim2real_mjx" / "inicio-lento"
      python_path = fake_repo / ".venvs/cpu/bin/python"
      trainer_path = fake_repo / "sim2real_mjx/entrenar_ppo_mjx.py"
      run_dir.mkdir(parents=True)
      python_path.parent.mkdir(parents=True)
      python_path.symlink_to(sys.executable)
      (fake_repo / ".sim2real").mkdir()
      (fake_repo / ".sim2real/accelerator").write_text("cpu\n", encoding="utf-8")
      (fake_repo / ".sim2real/runtime.lock").touch()
      trainer_path.parent.mkdir(parents=True)
      trainer_path.write_text(
          textwrap.dedent(
              """
              import json
              import os
              from pathlib import Path
              import signal
              import sys
              import time

              logdir = Path(sys.argv[sys.argv.index("--logdir") + 1])
              run_name = sys.argv[sys.argv.index("--run_name") + 1]
              run = logdir / run_name
              time.sleep(0.5)
              (run / "entrenamiento.pid").write_text(f"{os.getpid()}\\n", encoding="utf-8")
              state_path = run / "estado.json"
              state_path.write_text(json.dumps({
                  "estado": "entrenando",
                  "pid": os.getpid(),
                  "perfil_ppo": "ligero",
                  "fase_curriculum_recompensa": 2,
              }), encoding="utf-8")
              signal.signal(signal.SIGTERM, lambda *_: raise_exit())
              def_wait = time.sleep
              while True:
                  def_wait(1)
              """
          ).replace(
              "signal.signal(signal.SIGTERM, lambda *_: raise_exit())\n", ""
          ),
          encoding="utf-8",
      )
      state_path = run_dir / "estado.json"
      state_path.write_text(
          json.dumps(
              {
                  "estado": "iniciando",
                  "perfil_ppo": "ligero",
                  "fase_curriculum_recompensa": 2,
              }
          ),
          encoding="utf-8",
      )
      process = subprocess.Popen(
          [
              "flock",
              "--shared",
              str(fake_repo / ".sim2real/runtime.lock"),
              str(python_path),
              str(trainer_path),
              "--logdir",
              str(run_dir.parent),
              "--run_name",
              run_dir.name,
          ]
      )
      (run_dir / "lanzador.pid").write_text(f"{process.pid}\n", encoding="utf-8")
      try:
        command = "\n".join(
            (
                f"source {shlex.quote(str(launcher))}",
                f"REPO_ROOT={shlex.quote(str(fake_repo))}",
                f"LOGS_DIR={shlex.quote(str(run_dir.parent))}",
                f"ULTIMA_EJECUCION={shlex.quote(str(run_dir.parent / 'ultima_ejecucion.txt'))}",
                f"parar_entrenamiento --ejecucion {shlex.quote(run_dir.name)}",
            )
        )
        completed = subprocess.run(
            ["bash", "-c", command], text=True, capture_output=True, timeout=20
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        process.wait(timeout=5)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["estado"], "cancelado")
        self.assertEqual(state["perfil_ppo"], "ligero")
        self.assertEqual(state["fase_curriculum_recompensa"], 2)
        self.assertTrue((run_dir / "parada_total_solicitada").is_file())
      finally:
        if process.poll() is None:
          process.terminate()
          try:
            process.wait(timeout=5)
          except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

  def test_curriculum_counts_only_reported_steps(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      fake_repo = Path(temporary) / "repo"
      scripts_dir = fake_repo / "scripts"
      lib_dir = scripts_dir / "lib"
      lib_dir.mkdir(parents=True)
      shutil.copy2(ROOT / "scripts/lib/platform.sh", lib_dir / "platform.sh")
      shutil.copy2(
          ROOT / "scripts/curriculo_automatico_sim2real.sh",
          scripts_dir / "curriculo_automatico_sim2real.sh",
      )
      (scripts_dir / "curriculo_automatico_sim2real.sh").chmod(0o755)
      python_path = fake_repo / ".venvs/cpu/bin/python"
      python_path.parent.mkdir(parents=True)
      python_path.symlink_to(sys.executable)
      trainer_path = fake_repo / "sim2real_mjx/entrenar_ppo_mjx.py"
      _write_executable(
          trainer_path,
          r'''
          #!/usr/bin/env python3
          import csv
          import json
          import os
          from pathlib import Path
          import sys
          import time

          def value(name):
              return sys.argv[sys.argv.index(name) + 1]

          logdir = Path(value("--logdir"))
          run = logdir / value("--run_name")
          steps = int(value("--fake-steps"))
          phase = int(value("--fake-phase"))
          (run / "entrenamiento.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
          (run / "estado.json").write_text(json.dumps({
              "estado": "entrenando",
              "pid": os.getpid(),
              "perfil_ppo": "depuracion",
              "fase_curriculum_recompensa": phase,
          }), encoding="utf-8")
          time.sleep(0.15)
          progress = run / "progreso.csv"
          exists = progress.exists()
          with progress.open("a", newline="", encoding="utf-8") as handle:
              writer = csv.DictWriter(handle, fieldnames=["num_steps", "source"])
              if not exists:
                  writer.writeheader()
              writer.writerow({"num_steps": steps, "source": "eval"})
          (run / "estado.json").write_text(json.dumps({
              "estado": "terminado",
              "pid": os.getpid(),
              "perfil_ppo": "depuracion",
              "fase_curriculum_recompensa": phase,
              "ultimo_paso_confirmado": steps,
          }), encoding="utf-8")
          ''',
      )
      _write_executable(
          scripts_dir / "sim2real.sh",
          r'''
          #!/usr/bin/env bash
          set -euo pipefail
          repo_root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
          logs="${repo_root}/logs_sim2real_mjx"
          command="$1"
          shift
          [[ "${command}" == "entrenar" ]] || exit 2
          run=""; requested=""; phase="1"
          while [[ $# -gt 0 ]]; do
            case "$1" in
              --nombre-ejecucion) run="$2"; shift 2 ;;
              --num-timesteps) requested="$2"; shift 2 ;;
              --fase-recompensa) phase="$2"; shift 2 ;;
              --perfil-ppo) shift 2 ;;
              --segundo-plano|--omitir-prueba-mjx|--anexar-csv|--setup|--desde-cero) shift ;;
              *) shift ;;
            esac
          done
          run_dir="${logs}/${run}"
          mkdir -p "${run_dir}" "${repo_root}/.sim2real"
          touch "${repo_root}/.sim2real/runtime.lock"
          counter="${run_dir}/bloques_lanzados.txt"
          launches=0
          [[ ! -f "${counter}" ]] || launches="$(wc -l < "${counter}")"
          actual="${requested}"
          if [[ "${launches}" -eq 0 ]]; then actual=4; fi
          printf '%s\n' "${requested}" >> "${counter}"
          nohup flock --shared "${repo_root}/.sim2real/runtime.lock" \
            "${repo_root}/.venvs/cpu/bin/python" \
            "${repo_root}/sim2real_mjx/entrenar_ppo_mjx.py" \
            --logdir "${logs}" --run_name "${run}" \
            --fake-steps "${actual}" --fake-phase "${phase}" \
            </dev/null >/dev/null 2>&1 &
          launcher_pid=$!
          printf '%s\n' "${launcher_pid}" > "${run_dir}/lanzador.pid"
          ''',
      )
      completed = subprocess.run(
          [
              "bash",
              str(scripts_dir / "curriculo_automatico_sim2real.sh"),
              "--perfil-ppo",
              "depuracion",
              "--nombre-ejecucion",
              "conteo-real",
              "--fase-inicial",
              "1",
              "--pasos-totales",
              "9",
              "--pasos-por-bloque",
              "9",
              "--sin-preparacion",
              "--sin-reinicio-inicial",
          ],
          cwd=fake_repo,
          text=True,
          capture_output=True,
          timeout=20,
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)
      run_dir = fake_repo / "logs_sim2real_mjx/conteo-real"
      launches = (run_dir / "bloques_lanzados.txt").read_text(encoding="utf-8").splitlines()
      self.assertEqual(launches, ["9", "5"])
      state = json.loads(
          (run_dir / "curriculo_automatico_estado.json").read_text(encoding="utf-8")
      )
      self.assertEqual(state["estado"], "terminado")
      self.assertEqual(state["pasos_confirmados_por_supervisor"], 9)
      self.assertFalse((run_dir / "curriculo_automatico.pid").exists())

  def test_reward_metric_dictionary_has_no_duplicate_literal_keys(self) -> None:
    trainer = ROOT / "sim2real_mjx/entrenar_ppo_mjx.py"
    parsed = ast.parse(trainer.read_text(encoding="utf-8"))
    duplicates: list[str] = []
    for node in ast.walk(parsed):
      if not isinstance(node, ast.Dict):
        continue
      keys = [key.value for key in node.keys if isinstance(key, ast.Constant)]
      duplicates.extend(key for key in set(keys) if keys.count(key) > 1)
    self.assertNotIn("vector_gravedad_estabilidad_reward_ponderado", duplicates)

  def test_background_training_keeps_nohup_sighup_ignored(self) -> None:
    source = (ROOT / "sim2real_mjx/entrenar_ppo_mjx.py").read_text(
        encoding="utf-8"
    )
    self.assertIn(
        "if signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:", source
    )
    self.assertNotIn(
        "for signal_number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):",
        source,
    )


if __name__ == "__main__":
  unittest.main()
