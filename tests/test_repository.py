from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
  def test_required_files_exist(self) -> None:
    required = (
        "pyproject.toml",
        "uv.lock",
        ".gitattributes",
        ".sim2real-repository",
        "scripts/install.sh",
        "scripts/bootstrap_ubuntu.sh",
        "scripts/curriculo_automatico_sim2real.sh",
        "scripts/doctor.sh",
        "scripts/install_windows.ps1",
        "scripts/sim2real.sh",
        "scripts/test_static.sh",
        "scripts/lib/platform.sh",
        "tests/test_install_windows.ps1",
        "requirements/amd-rocm70-py312.txt",
        "sim2real_mjx/curriculo_recompensas.py",
        "sim2real_mjx/entorno_robot_mjx.py",
        "sim2real_mjx/xmls/ROBOT_MJX.xml",
    )
    missing = [path for path in required if not (ROOT / path).is_file()]
    self.assertEqual(missing, [], f"Faltan archivos: {missing}")

  def test_heavy_runtime_data_is_ignored(self) -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (
        ".venvs/",
        "external/",
        "logs_sim2real_mjx/",
        "checkpoints*/",
        "modelos_antiguos_sim2real/",
        ".sim2real/",
    ):
      self.assertIn(pattern, ignore)

  def test_scientific_versions_and_git_pin(self) -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    self.assertEqual(project["project"]["requires-python"], ">=3.12,<3.13")
    dependencies = "\n".join(project["project"]["dependencies"])
    for pin in (
        "jax==0.6.2",
        "jaxlib==0.6.2",
        "mujoco==3.6.0",
        "mujoco-mjx==3.6.0",
        "brax==0.14.2",
        "flax==0.11.2",
        "playground @ git+https://github.com/google-deepmind/mujoco_playground.git@9c2dce4a3519cd4bb9d299bf28a6ef3f5086844b",
    ):
      self.assertIn(pin, dependencies)
    self.assertEqual(project["project"]["optional-dependencies"]["nvidia"], ["jax[cuda12]==0.6.2"])
    self.assertNotIn("amd", project["project"]["optional-dependencies"])

  def test_amd_overlay_is_complete_and_hashed(self) -> None:
    lines = [
        line.strip()
        for line in (ROOT / "requirements/amd-rocm70-py312.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    self.assertEqual(len(lines), 3)
    self.assertTrue(any("jaxlib-0.6.2-cp312" in line for line in lines))
    self.assertTrue(any("jax_rocm7_pjrt-0.6.0" in line for line in lines))
    self.assertTrue(any("jax_rocm7_plugin-0.6.0-cp312" in line for line in lines))
    for line in lines:
      self.assertRegex(line, r"#sha256=[0-9a-f]{64}$")

  def test_no_absolute_old_phase_paths(self) -> None:
    offenders: list[str] = []
    needles = ("5_Mujoco_Playground_MJX_JAX_JAXPPO_WSL", "/home/ubuntuvirtual")
    for folder in ("scripts", "sim2real_mjx", "docs"):
      for path in (ROOT / folder).rglob("*"):
        if path.is_file() and path.suffix in {".sh", ".py", ".ps1", ".md"}:
          text = path.read_text(encoding="utf-8", errors="replace")
          if any(needle in text for needle in needles):
            offenders.append(str(path.relative_to(ROOT)))
    self.assertEqual(offenders, [], f"Rutas historicas hardcodeadas: {offenders}")

  def test_no_previous_robot_brand_remains(self) -> None:
    forbidden = ("taran" + "tulin").encode("ascii")
    ignored_roots = {".git", ".venvs", ".sim2real", "logs_sim2real_mjx", "__pycache__"}
    offenders: list[str] = []

    for path in ROOT.rglob("*"):
      relative = path.relative_to(ROOT)
      if any(part in ignored_roots for part in relative.parts):
        continue
      relative_bytes = relative.as_posix().lower().encode("utf-8", errors="ignore")
      if forbidden in relative_bytes:
        offenders.append(relative.as_posix())
        continue
      if path.is_file() and forbidden in path.read_bytes().lower():
        offenders.append(relative.as_posix())

    self.assertEqual(offenders, [], f"Quedan referencias a la marca anterior: {offenders}")

  def test_no_runtime_dependency_on_external_checkout(self) -> None:
    offenders: list[str] = []
    for path in (ROOT / "scripts").rglob("*"):
      if path.is_file() and path.suffix in {".sh", ".py", ".ps1"}:
        if "external/mujoco_playground" in path.read_text(encoding="utf-8", errors="replace"):
          offenders.append(str(path.relative_to(ROOT)))
    self.assertEqual(offenders, [])

  def test_shell_syntax(self) -> None:
    scripts = sorted((ROOT / "scripts").rglob("*.sh"))
    completed = subprocess.run(
        ["bash", "-n", *map(str, scripts)],
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertEqual(completed.returncode, 0, completed.stderr)

  def test_python_syntax(self) -> None:
    python_files = sorted((ROOT / "sim2real_mjx").rglob("*.py"))
    python_files += sorted((ROOT / "scripts").rglob("*.py"))
    for path in python_files:
      with self.subTest(path=path.relative_to(ROOT)):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

  def test_environment_does_not_share_a_mutable_default_config(self) -> None:
    source_path = ROOT / "sim2real_mjx/entorno_robot_mjx.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), source_path.as_posix())
    environment_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EntornoRobotMJX"
    )
    initializer = next(
        node
        for node in environment_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    positional_names = [argument.arg for argument in initializer.args.args]
    defaults_by_name = dict(
        zip(
            positional_names[-len(initializer.args.defaults):],
            initializer.args.defaults,
            strict=True,
        )
    )
    config_default = defaults_by_name["config"]
    self.assertIsInstance(config_default, ast.Constant)
    self.assertIsNone(config_default.value)

  def test_rl_layer_has_real_spanish_names_without_python_bridges(self) -> None:
    obsolete_modules = (
        "sim2real_mjx/reward_curriculum.py",
        "sim2real_mjx/sim2real_mjx_env.py",
    )
    for relative_path in obsolete_modules:
      with self.subTest(relative_path=relative_path):
        self.assertFalse(
            (ROOT / relative_path).exists(),
            f"Sigue existiendo el modulo puente {relative_path}",
        )

    package_source = (ROOT / "sim2real_mjx/__init__.py").read_text(encoding="utf-8")
    self.assertIn(
        "from sim2real_mjx.entorno_robot_mjx import EntornoRobotMJX",
        package_source,
    )
    self.assertIn(
        '__all__ = ["EntornoRobotMJX", "default_config"]', package_source
    )

    python_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "sim2real_mjx").rglob("*.py")
    )
    for obsolete_name in (
        "sim2real_mjx.reward_curriculum",
        "sim2real_mjx.sim2real_mjx_env",
        "aplicar_fase_curriculum_recompensa",
        "DEBUG_REWARD_VERSION",
        "ppo_sim2real_standup",
        "config_a_dict",
        "_crear_network_factory",
    ):
      with self.subTest(obsolete_name=obsolete_name):
        self.assertNotIn(obsolete_name, python_sources)

    trainer_source = (ROOT / "sim2real_mjx/entrenar_ppo_mjx.py").read_text(
        encoding="utf-8"
    )
    self.assertIn(
        "from sim2real_mjx.curriculo_recompensas import ", trainer_source
    )
    self.assertIn("aplicar_fase_curriculo_recompensa", trainer_source)
    self.assertIn("EntornoRobotMJX", trainer_source)

    hyperparameters_source = (ROOT / "sim2real_mjx/hiperparametros.py").read_text(
        encoding="utf-8"
    )
    for expected_name in (
        "configuracion_ppo_depuracion",
        "configuracion_ppo_ligera",
        "configuracion_ppo_ligera_rapida",
        "configuracion_ppo_completa",
        "configuracion_a_diccionario",
    ):
      with self.subTest(expected_name=expected_name):
        self.assertRegex(hyperparameters_source, rf"def {expected_name}\(")

  def test_scientific_layer_receives_the_resolved_accelerator(self) -> None:
    launcher = (ROOT / "scripts/sim2real.sh").read_text(encoding="utf-8")
    self.assertIn('export SIM2REAL_ACCELERATOR="${SIM2REAL_ACCELERATOR_ACTIVE}"', launcher)

    for relative_path in (
        "sim2real_mjx/entrenar_ppo_mjx.py",
        "sim2real_mjx/benchmark_mjx.py",
    ):
      with self.subTest(path=relative_path):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        self.assertIn('os.environ.get("SIM2REAL_ACCELERATOR", "auto")', source)
        self.assertNotIn("SIM2REAL_MJX_ACCELERATOR", source)

  def test_accelerator_names_and_wsl_amd_policy(self) -> None:
    lib = ROOT / "scripts/lib/platform.sh"
    command = f"""
      source {lib!s}
      for profile in auto nvidia amd intel cpu; do
        sim2real_validate_accelerator \"$profile\"
        sim2real_validate_accelerator_request \"$profile\"
      done
      sim2real_validate_accelerator_request compatible
      ! sim2real_validate_accelerator compatible >/dev/null 2>&1
      ! sim2real_validate_accelerator fake >/dev/null 2>&1
      ! sim2real_validate_accelerator_request fake >/dev/null 2>&1
    """
    completed = subprocess.run(
        ["bash", "-c", command], text=True, capture_output=True, check=False
    )
    self.assertEqual(completed.returncode, 0, completed.stderr)

    # Incluso con el hardware y ROCm simulados como operativos, AMD/JAX bajo
    # WSL2 sigue siendo una combinacion bloqueada. No existe una opcion oculta
    # que transforme el rechazo en una caida silenciosa a CPU.
    amd_wsl = subprocess.run(
        [
            "bash",
            "-c",
            f'''
              source {shlex.quote(str(lib))}
              sim2real_has_amd_hardware() {{ return 0; }}
              sim2real_has_rocm_runtime() {{ return 0; }}
              sim2real_check_profile_host amd wsl {shlex.quote(str(ROOT))}
            ''',
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertNotEqual(amd_wsl.returncode, 0)
    self.assertRegex(amd_wsl.stderr.lower(), r"amd|rocm")
    self.assertRegex(amd_wsl.stderr.lower(), r"wsl")

    source = lib.read_text(encoding="utf-8")
    self.assertNotIn("SIM2REAL_ALLOW_EXPERIMENTAL_AMD_WSL", source)
    self.assertNotIn("allow-amd-wsl", source)

  def test_bootstrap_and_local_installer_share_the_support_summary(self) -> None:
    bootstrap = ROOT / "scripts/bootstrap_ubuntu.sh"
    platform_library = ROOT / "scripts/lib/platform.sh"
    with tempfile.TemporaryDirectory() as temporary:
      bootstrap_summary = Path(temporary) / "bootstrap.txt"
      platform_summary = Path(temporary) / "platform.txt"
      completed = subprocess.run(
          [
              "bash",
              "-c",
              f'''
                source {shlex.quote(str(bootstrap))}
                sim2real_bootstrap_print_accelerator_support_summary >"$1"
                source {shlex.quote(str(platform_library))}
                sim2real_print_accelerator_support_summary >"$2"
                cmp -- "$1" "$2"
              ''',
              "support-summary-test",
              str(bootstrap_summary),
              str(platform_summary),
          ],
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)

  def test_vendor_environments_are_isolated_and_runtime_has_no_legacy_venv(self) -> None:
    lib = ROOT / "scripts/lib/platform.sh"
    with tempfile.TemporaryDirectory() as temporary:
      repository = Path(temporary) / "repo con espacios"
      repository.mkdir()
      command = f'''
        source {shlex.quote(str(lib))}
        for accelerator in nvidia amd cpu; do
          printf '%s|%s|%s\n' \
            "${{accelerator}}" \
            "$(sim2real_environment_dir "${{accelerator}}" {shlex.quote(str(repository))})" \
            "$(sim2real_python_executable "${{accelerator}}" {shlex.quote(str(repository))})"
        done
      '''
      completed = subprocess.run(
          ["bash", "-c", command], text=True, capture_output=True, check=False
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)
      self.assertEqual(
          completed.stdout.splitlines(),
          [
              f"nvidia|{repository}/.venvs/nvidia-cuda12|"
              f"{repository}/.venvs/nvidia-cuda12/bin/python",
              f"amd|{repository}/.venvs/amd-rocm70|"
              f"{repository}/.venvs/amd-rocm70/bin/python",
              f"cpu|{repository}/.venvs/cpu|{repository}/.venvs/cpu/bin/python",
          ],
      )

    runtime_scripts = (
        "scripts/install.sh",
        "scripts/doctor.sh",
        "scripts/sim2real.sh",
        "scripts/graficar_recompensas.sh",
        "scripts/ver_recompensas_en_directo.sh",
        "scripts/visualizar_sim2real.sh",
    )
    for relative_path in runtime_scripts:
      with self.subTest(relative_path=relative_path):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        self.assertNotIn("/.venv/", source)
        self.assertNotIn("${REPO_ROOT}/.venv", source)
        self.assertNotIn("${PWD}/.venv", source)

  def test_compatible_is_a_request_selector_never_a_saved_profile_or_slug(self) -> None:
    lib = ROOT / "scripts/lib/platform.sh"
    with tempfile.TemporaryDirectory() as temporary:
      repository = Path(temporary) / "repository"
      state = repository / ".sim2real"
      state.mkdir(parents=True)
      (state / "accelerator").write_text("compatible\n", encoding="utf-8")
      completed = subprocess.run(
          [
              "bash",
              "-c",
              r'''
                source "$1"
                repository="$2"
                sim2real_validate_accelerator_request compatible
                ! sim2real_validate_accelerator compatible >/dev/null 2>&1
                ! sim2real_environment_slug compatible >/dev/null 2>&1
                ! sim2real_saved_accelerator "${repository}" >/dev/null 2>&1
                ! sim2real_save_accelerator compatible "${repository}" >/dev/null 2>&1
              ''',
              "compatible-selector-test",
              str(lib),
              str(repository),
          ],
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)
      self.assertEqual((state / "accelerator").read_text(encoding="utf-8"), "compatible\n")

  def test_nvidia_driver_version_boundary(self) -> None:
    lib = ROOT / "scripts/lib/platform.sh"
    command = f'''
      source {shlex.quote(str(lib))}
      ! sim2real_nvidia_driver_version_supported 524.99
      sim2real_nvidia_driver_version_supported 525
      sim2real_nvidia_driver_version_supported 525.0.0
      ! sim2real_nvidia_driver_version_supported desconocida
    '''
    completed = subprocess.run(
        ["bash", "-c", command], text=True, capture_output=True, check=False
    )
    self.assertEqual(completed.returncode, 0, completed.stderr)

  def test_auto_detection_selects_only_ready_gpu_and_never_returns_cpu(self) -> None:
    lib = ROOT / "scripts/lib/platform.sh"
    body = r'''
      source "$1"
      sim2real_detect_host() { printf '%s\n' "${FAKE_HOST}"; }
      sim2real_os_version_id() { printf '%s\n' "${FAKE_OS_VERSION}"; }
      sim2real_nvidia_smi_command() {
        [[ "${FAKE_NVIDIA_READY}" == 1 ]] || return 1
        printf '%s\n' /bin/true
      }
      sim2real_nvidia_driver_version() { printf '%s\n' 550.120; }
      sim2real_has_nvidia_hardware() {
        [[ "${FAKE_NVIDIA_HARDWARE}" == 1 ]]
      }
      sim2real_has_amd_hardware() { [[ "${FAKE_AMD_HARDWARE}" == 1 ]]; }
      sim2real_has_rocm_runtime() { [[ "${FAKE_AMD_READY}" == 1 ]]; }
      sim2real_rocm_version() { printf '%s\n' 7.0.2; }
      sim2real_has_intel_hardware() { [[ "${FAKE_INTEL_HARDWARE}" == 1 ]]; }
      sim2real_detect_accelerator
    '''
    defaults = {
        "FAKE_HOST": "ubuntu",
        "FAKE_OS_VERSION": "24.04",
        "FAKE_NVIDIA_READY": "0",
        "FAKE_NVIDIA_HARDWARE": "0",
        "FAKE_AMD_READY": "0",
        "FAKE_AMD_HARDWARE": "0",
        "FAKE_INTEL_HARDWARE": "0",
    }
    scenarios = (
        ({"FAKE_NVIDIA_READY": "1", "FAKE_NVIDIA_HARDWARE": "1"}, "nvidia"),
        ({"FAKE_AMD_READY": "1", "FAKE_AMD_HARDWARE": "1"}, "amd"),
        (
            {
                "FAKE_OS_VERSION": "22.04",
                "FAKE_NVIDIA_READY": "1",
                "FAKE_NVIDIA_HARDWARE": "1",
                "FAKE_AMD_READY": "1",
                "FAKE_AMD_HARDWARE": "1",
            },
            "nvidia",
        ),
    )
    for overrides, expected in scenarios:
      with self.subTest(overrides=overrides):
        completed = subprocess.run(
            ["bash", "-c", body, "auto-test", str(lib)],
            env=os.environ | defaults | overrides,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), expected)

    rejected = (
        {},
        {"FAKE_NVIDIA_HARDWARE": "1"},
        {"FAKE_AMD_HARDWARE": "1"},
        {
            "FAKE_OS_VERSION": "22.04",
            "FAKE_AMD_READY": "1",
            "FAKE_AMD_HARDWARE": "1",
        },
        {"FAKE_INTEL_HARDWARE": "1"},
        {
            "FAKE_NVIDIA_READY": "1",
            "FAKE_NVIDIA_HARDWARE": "1",
            "FAKE_AMD_READY": "1",
            "FAKE_AMD_HARDWARE": "1",
        },
        {"FAKE_HOST": "wsl", "FAKE_AMD_READY": "1", "FAKE_AMD_HARDWARE": "1"},
    )
    for overrides in rejected:
      with self.subTest(rejected=overrides):
        completed = subprocess.run(
            ["bash", "-c", body, "auto-test", str(lib)],
            env=os.environ | defaults | overrides,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0, completed.stderr)
        self.assertNotEqual(completed.stdout.strip(), "cpu")
        self.assertRegex(completed.stderr.lower(), r"gpu|cpu|nvidia|amd|intel")

  def test_compatible_detection_prefers_supported_gpu_and_falls_back_visibly(self) -> None:
    lib = ROOT / "scripts/lib/platform.sh"
    body = r'''
      source "$1"
      sim2real_detect_host() { printf '%s\n' "${FAKE_HOST}"; }
      sim2real_os_version_id() { printf '%s\n' "${FAKE_OS_VERSION}"; }
      sim2real_nvidia_smi_command() {
        [[ "${FAKE_NVIDIA_READY}" == 1 ]] || return 1
        printf '%s\n' /bin/true
      }
      sim2real_nvidia_driver_version() { printf '%s\n' 550.120; }
      sim2real_has_nvidia_hardware() {
        [[ "${FAKE_NVIDIA_HARDWARE}" == 1 ]]
      }
      sim2real_has_amd_hardware() { [[ "${FAKE_AMD_HARDWARE}" == 1 ]]; }
      sim2real_has_rocm_runtime() { [[ "${FAKE_AMD_READY}" == 1 ]]; }
      sim2real_rocm_version() { printf '%s\n' 7.0.2; }
      sim2real_has_intel_hardware() { [[ "${FAKE_INTEL_HARDWARE}" == 1 ]]; }
      sim2real_detect_compatible_accelerator
    '''
    defaults = {
        "FAKE_HOST": "ubuntu",
        "FAKE_OS_VERSION": "24.04",
        "FAKE_NVIDIA_READY": "0",
        "FAKE_NVIDIA_HARDWARE": "0",
        "FAKE_AMD_READY": "0",
        "FAKE_AMD_HARDWARE": "0",
        "FAKE_INTEL_HARDWARE": "0",
    }
    scenarios = (
        (
            {"FAKE_HOST": "wsl", "FAKE_NVIDIA_READY": "1", "FAKE_NVIDIA_HARDWARE": "1"},
            "nvidia",
            False,
        ),
        (
            {"FAKE_AMD_READY": "1", "FAKE_AMD_HARDWARE": "1"},
            "amd",
            False,
        ),
        (
            {"FAKE_HOST": "wsl", "FAKE_INTEL_HARDWARE": "1"},
            "cpu",
            True,
        ),
    )
    for overrides, expected, expects_warning in scenarios:
      with self.subTest(overrides=overrides):
        completed = subprocess.run(
            ["bash", "-c", body, "compatible-test", str(lib)],
            env=os.environ | defaults | overrides,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), expected)
        if expects_warning:
          self.assertIn("AVISO", completed.stderr)
          self.assertIn("Intel", completed.stderr)
          self.assertIn("selecciona CPU automaticamente", completed.stderr)
        else:
          self.assertNotIn("AVISO", completed.stderr)

  def test_supported_host_rejects_architecture_and_ubuntu_version(self) -> None:
    lib = ROOT / "scripts/lib/platform.sh"
    simulated_hosts = (
        ("aarch64", "24.04", "Arquitectura no soportada"),
        ("x86_64", "20.04", "Version de Ubuntu no soportada"),
    )
    for architecture, ubuntu_version, expected_error in simulated_hosts:
      with self.subTest(architecture=architecture, ubuntu_version=ubuntu_version):
        command = f'''
          source {shlex.quote(str(lib))}
          sim2real_detect_host() {{ printf 'ubuntu\\n'; }}
          sim2real_os_id() {{ printf 'ubuntu\\n'; }}
          sim2real_os_version_id() {{ printf '{ubuntu_version}\\n'; }}
          uname() {{ printf '{architecture}\\n'; }}
          sim2real_require_supported_host
        '''
        completed = subprocess.run(
            ["bash", "-c", command], text=True, capture_output=True, check=False
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(expected_error, completed.stderr)

  def test_installer_reports_missing_accelerator_value(self) -> None:
    installer = ROOT / "scripts/install.sh"
    source = installer.read_text(encoding="utf-8")
    self.assertIn('ACCELERATOR="${SIM2REAL_ACCELERATOR:-auto}"', source)
    completed = subprocess.run(
        ["bash", str(installer), "--accelerator"],
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertNotEqual(completed.returncode, 0)
    self.assertIn("Falta el valor de --accelerator", completed.stderr)
    self.assertNotIn("unbound variable", completed.stderr)

  def _assert_installer_rejects_held_runtime_lock(self, mode: str) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      fixture_root = Path(temporary) / "repository"
      scripts_dir = fixture_root / "scripts"
      library_dir = scripts_dir / "lib"
      library_dir.mkdir(parents=True)
      shutil.copy2(ROOT / "scripts/install.sh", scripts_dir / "install.sh")
      shutil.copy2(ROOT / "scripts/lib/platform.sh", library_dir / "platform.sh")

      holder = subprocess.Popen(
          [
              "bash",
              "-c",
              r'''
                set -euo pipefail
                source "$1"
                sim2real_acquire_runtime_lock "$2" "$3"
                printf 'LOCKED\n'
                IFS= read -r _
              ''',
              "runtime-lock-holder",
              str(library_dir / "platform.sh"),
              mode,
              str(fixture_root),
          ],
          stdin=subprocess.PIPE,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
      )
      try:
        assert holder.stdout is not None
        ready = holder.stdout.readline().strip()
        if ready != "LOCKED":
          holder.terminate()
          _, holder_stderr = holder.communicate(timeout=5)
          self.fail(f"No se adquirio el lock {mode}: {holder_stderr}")

        completed = subprocess.run(
            [
                "bash",
                str(scripts_dir / "install.sh"),
                "--accelerator",
                "cpu",
                "--skip-system-packages",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertIn("Otro comando Sim2Real MJX-JAX", completed.stderr)
        self.assertFalse((fixture_root / ".venvs").exists())
      finally:
        if holder.poll() is None:
          holder.communicate("release\n", timeout=5)

  def test_concurrent_installation_is_rejected_by_exclusive_lock(self) -> None:
    self._assert_installer_rejects_held_runtime_lock("exclusive")

  def test_shared_runtime_lock_rejects_installation(self) -> None:
    self._assert_installer_rejects_held_runtime_lock("shared")

  def _make_transaction_recovery_fixture(
      self, temporary: str
  ) -> tuple[Path, dict[str, str]]:
    fixture_root = Path(temporary) / "repository"
    scripts_dir = fixture_root / "scripts"
    library_dir = scripts_dir / "lib"
    local_bin = Path(temporary) / "home/.local/bin"
    library_dir.mkdir(parents=True)
    local_bin.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/install.sh", scripts_dir / "install.sh")

    (library_dir / "platform.sh").write_text(
        r'''#!/usr/bin/env bash
sim2real_require_supported_host() { :; }
sim2real_require_workspace() { :; }
sim2real_acquire_runtime_lock() { :; }
sim2real_active_training() { return 1; }
sim2real_validate_accelerator() {
  [[ "$1" == auto || "$1" == nvidia || "$1" == amd || "$1" == intel || "$1" == cpu ]]
}
sim2real_validate_accelerator_request() {
  [[ "$1" == compatible ]] || sim2real_validate_accelerator "$1"
}
sim2real_detect_accelerator() { printf '%s\n' cpu; }
sim2real_detect_compatible_accelerator() { printf '%s\n' cpu; }
sim2real_resolve_accelerator() { printf '%s\n' "$1"; }
sim2real_detect_host() { printf '%s\n' ubuntu; }
sim2real_detect_virtualization() { printf '%s\n' none; }
sim2real_os_pretty_name() { printf '%s\n' 'Ubuntu de prueba'; }
sim2real_support_level() { printf '%s\n' prueba; }
sim2real_print_accelerator_support_summary() { printf '%s\n' 'resumen de compatibilidad'; }
sim2real_check_profile_host() {
  if [[ "${FAKE_PREFLIGHT_STATUS:-0}" != 0 ]]; then
    printf 'preflight rechazado para %s\n' "$1" >&2
    return "${FAKE_PREFLIGHT_STATUS}"
  fi
}
sim2real_environment_slug() {
  case "$1" in
    nvidia) printf '%s\n' nvidia-cuda12 ;;
    amd) printf '%s\n' amd-rocm70 ;;
    cpu) printf '%s\n' cpu ;;
    *) return 1 ;;
  esac
}
sim2real_environment_dir() {
  printf '%s/.venvs/%s\n' "$2" "$(sim2real_environment_slug "$1")"
}
sim2real_environments_root() { printf '%s/.venvs\n' "$1"; }
sim2real_python_executable() {
  printf '%s/bin/python\n' "$(sim2real_environment_dir "$1" "$2")"
}
sim2real_uv_extra_args() { :; }
sim2real_profile_file() { printf '%s/.sim2real/accelerator\n' "$1"; }
sim2real_saved_accelerator() {
  local profile
  profile="$(sim2real_profile_file "$1")"
  [[ -r "${profile}" ]] || return 1
  head -n 1 "${profile}"
}
sim2real_save_accelerator() {
  mkdir -p "$2/.sim2real"
  printf '%s\n' "$1" >"$2/.sim2real/accelerator.tmp"
  mv "$2/.sim2real/accelerator.tmp" "$2/.sim2real/accelerator"
}
''',
        encoding="utf-8",
    )
    doctor = scripts_dir / "doctor.sh"
    doctor.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    doctor.chmod(0o755)
    (fixture_root / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.0.0'\n", encoding="utf-8"
    )
    (fixture_root / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    fake_uv = local_bin / "uv"
    fake_uv.write_text(
        r'''#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then
  printf '%s\n' 'uv 0.11.8'
  exit 0
fi
if [[ "${1:-}" == "sync" ]]; then
  mkdir -p "${UV_PROJECT_ENVIRONMENT}"
  printf '%s\n' nueva >"${UV_PROJECT_ENVIRONMENT}/new-environment"
  exit "${FAKE_UV_SYNC_STATUS}"
fi
exit 0
''',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(Path(temporary) / "home"),
            "FAKE_UV_SYNC_STATUS": "73",
            "FAKE_PREFLIGHT_STATUS": "0",
        }
    )
    return fixture_root, environment

  def test_interrupted_transaction_restores_backup_and_profile(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      fixture_root, environment = self._make_transaction_recovery_fixture(temporary)
      environments_root = fixture_root / ".venvs"
      active_environment = environments_root / "cpu"
      transaction = environments_root / ".install-cpu.interrupted"
      backup = transaction / "previous"
      state_dir = fixture_root / ".sim2real"
      active_environment.mkdir(parents=True)
      backup.mkdir(parents=True)
      state_dir.mkdir(parents=True)
      (active_environment / "partial-environment").write_text(
          "descartar\n", encoding="utf-8"
      )
      (backup / "previous-environment").write_text(
          "conservar\n", encoding="utf-8"
      )
      (transaction / "initialized").touch()
      (transaction / "had-previous-environment").touch()
      (transaction / "previous-profile").write_text(
          "nvidia\n", encoding="utf-8"
      )
      (state_dir / "accelerator").write_text("cpu\n", encoding="utf-8")

      completed = subprocess.run(
          [
              "bash",
              str(fixture_root / "scripts/install.sh"),
              "--accelerator",
              "cpu",
              "--skip-system-packages",
          ],
          env=environment,
          text=True,
          capture_output=True,
          check=False,
          timeout=10,
      )

      self.assertEqual(completed.returncode, 73, completed.stderr)
      self.assertTrue((active_environment / "previous-environment").is_file())
      self.assertFalse((active_environment / "partial-environment").exists())
      self.assertFalse((active_environment / "new-environment").exists())
      self.assertEqual((state_dir / "accelerator").read_text(), "nvidia\n")
      self.assertEqual(list(environments_root.glob(".install-cpu.*")), [])

  def test_cpu_orphan_is_recovered_before_other_vendor_preflight(self) -> None:
    for requested_accelerator in ("amd", "nvidia"):
      with self.subTest(
          requested_accelerator=requested_accelerator
      ), tempfile.TemporaryDirectory() as temporary:
        fixture_root, environment = self._make_transaction_recovery_fixture(
            temporary
        )
        environment["FAKE_PREFLIGHT_STATUS"] = "81"
        environments_root = fixture_root / ".venvs"
        cpu_environment = environments_root / "cpu"
        transaction = environments_root / ".install-cpu.orphaned"
        backup = transaction / "previous"
        state_dir = fixture_root / ".sim2real"
        cpu_environment.mkdir(parents=True)
        backup.mkdir(parents=True)
        state_dir.mkdir(parents=True)
        (cpu_environment / "partial-environment").write_text(
            "descartar\n", encoding="utf-8"
        )
        recovered_marker = backup / "previous-environment"
        recovered_marker.write_text("recuperado\n", encoding="utf-8")
        (transaction / "initialized").touch()
        (transaction / "had-previous-environment").touch()
        (transaction / "previous-profile").write_text(
            "cpu\n", encoding="utf-8"
        )
        (state_dir / "accelerator").write_text("cpu\n", encoding="utf-8")

        completed = subprocess.run(
            [
                "bash",
                str(fixture_root / "scripts/install.sh"),
                "--accelerator",
                requested_accelerator,
                "--skip-system-packages",
            ],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )

        self.assertEqual(completed.returncode, 81, completed.stderr)
        self.assertIn(
            f"preflight rechazado para {requested_accelerator}", completed.stderr
        )
        self.assertEqual(
            (cpu_environment / recovered_marker.name).read_text(), "recuperado\n"
        )
        self.assertFalse((cpu_environment / "partial-environment").exists())
        self.assertFalse(transaction.exists())
        self.assertEqual((state_dir / "accelerator").read_text(), "cpu\n")
        requested_slug = {
            "amd": "amd-rocm70",
            "nvidia": "nvidia-cuda12",
        }[requested_accelerator]
        self.assertFalse((environments_root / requested_slug).exists())

  def test_cpu_orphan_does_not_replace_newer_active_profile(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      fixture_root, environment = self._make_transaction_recovery_fixture(temporary)
      environment["FAKE_PREFLIGHT_STATUS"] = "81"
      environments_root = fixture_root / ".venvs"
      cpu_environment = environments_root / "cpu"
      transaction = environments_root / ".install-cpu.orphaned"
      backup = transaction / "previous"
      state_dir = fixture_root / ".sim2real"
      cpu_environment.mkdir(parents=True)
      backup.mkdir(parents=True)
      state_dir.mkdir(parents=True)
      (cpu_environment / "partial-environment").write_text(
          "descartar\n", encoding="utf-8"
      )
      (backup / "previous-environment").write_text(
          "recuperado\n", encoding="utf-8"
      )
      (transaction / "initialized").touch()
      (transaction / "had-previous-environment").touch()
      (transaction / "previous-profile").write_text(
          "nvidia\n", encoding="utf-8"
      )
      (state_dir / "accelerator").write_text("amd\n", encoding="utf-8")

      completed = subprocess.run(
          [
              "bash",
              str(fixture_root / "scripts/install.sh"),
              "--accelerator",
              "nvidia",
              "--skip-system-packages",
          ],
          env=environment,
          text=True,
          capture_output=True,
          check=False,
          timeout=10,
      )

      self.assertEqual(completed.returncode, 81, completed.stderr)
      self.assertTrue((cpu_environment / "previous-environment").is_file())
      self.assertFalse((cpu_environment / "partial-environment").exists())
      self.assertFalse(transaction.exists())
      self.assertEqual((state_dir / "accelerator").read_text(), "amd\n")
      self.assertIn("Se conserva el perfil activo amd", completed.stdout)

  def test_committed_residue_keeps_valid_environment_and_cleans_backup(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      fixture_root, environment = self._make_transaction_recovery_fixture(temporary)
      environments_root = fixture_root / ".venvs"
      active_environment = environments_root / "cpu"
      transaction = environments_root / ".committed-cpu.finished"
      backup = transaction / "previous"
      state_dir = fixture_root / ".sim2real"
      active_environment.mkdir(parents=True)
      backup.mkdir(parents=True)
      state_dir.mkdir(parents=True)
      valid_marker = active_environment / "valid-environment"
      valid_marker.write_text("conservar\n", encoding="utf-8")
      (backup / "obsolete-backup").write_text("eliminar\n", encoding="utf-8")
      (transaction / "commit-complete").touch()
      (state_dir / "accelerator").write_text("cpu\n", encoding="utf-8")

      completed = subprocess.run(
          [
              "bash",
              str(fixture_root / "scripts/install.sh"),
              "--accelerator",
              "cpu",
              "--skip-system-packages",
          ],
          env=environment,
          text=True,
          capture_output=True,
          check=False,
          timeout=10,
      )

      self.assertEqual(completed.returncode, 73, completed.stderr)
      self.assertEqual(valid_marker.read_text(), "conservar\n")
      self.assertFalse((active_environment / "new-environment").exists())
      self.assertEqual((state_dir / "accelerator").read_text(), "cpu\n")
      self.assertEqual(list(environments_root.glob(".committed-cpu.*")), [])
      self.assertEqual(list(environments_root.rglob("obsolete-backup")), [])

  def test_installer_activates_profile_only_after_successful_doctor(self) -> None:
    """El perfil activo es un commit posterior a la validacion, no una intencion."""

    scenarios = (
        ("cpu", [], "0", 0, ["doctor", "save:cpu"]),
        ("cpu", [], "71", 71, ["doctor"]),
        ("compatible", [], "0", 0, ["doctor", "save:cpu"]),
    )
    for (
        requested_accelerator,
        extra_arguments,
        doctor_status,
        expected_status,
        expected_lifecycle,
    ) in scenarios:
      with self.subTest(
          requested_accelerator=requested_accelerator,
          extra_arguments=extra_arguments,
          doctor_status=doctor_status,
      ), tempfile.TemporaryDirectory() as temporary:
        fixture_root = Path(temporary) / "repository"
        scripts_dir = fixture_root / "scripts"
        library_dir = scripts_dir / "lib"
        local_bin = Path(temporary) / "home/.local/bin"
        library_dir.mkdir(parents=True)
        local_bin.mkdir(parents=True)
        shutil.copy2(ROOT / "scripts/install.sh", scripts_dir / "install.sh")

        events = Path(temporary) / "lifecycle.log"
        (library_dir / "platform.sh").write_text(
            r'''#!/usr/bin/env bash
sim2real_require_supported_host() { :; }
sim2real_require_workspace() { :; }
sim2real_acquire_runtime_lock() { :; }
sim2real_active_training() { return 1; }
sim2real_validate_accelerator() {
  [[ "$1" == auto || "$1" == nvidia || "$1" == amd || "$1" == intel || "$1" == cpu ]]
}
sim2real_validate_accelerator_request() {
  [[ "$1" == compatible ]] || sim2real_validate_accelerator "$1"
}
sim2real_detect_accelerator() { printf '%s\n' cpu; }
sim2real_detect_compatible_accelerator() { printf '%s\n' cpu; }
sim2real_resolve_accelerator() { printf '%s\n' "$1"; }
sim2real_detect_host() { printf '%s\n' ubuntu; }
sim2real_detect_virtualization() { printf '%s\n' none; }
sim2real_os_pretty_name() { printf '%s\n' 'Ubuntu de prueba'; }
sim2real_support_level() { printf '%s\n' prueba; }
sim2real_print_accelerator_support_summary() { printf '%s\n' 'resumen de compatibilidad'; }
sim2real_check_profile_host() { :; }
sim2real_has_nvidia_hardware() { return 1; }
sim2real_has_amd_hardware() { return 1; }
sim2real_has_intel_hardware() { return 1; }
sim2real_environment_slug() {
  case "$1" in
    nvidia) printf '%s\n' nvidia-cuda12 ;;
    amd) printf '%s\n' amd-rocm70 ;;
    cpu) printf '%s\n' cpu ;;
    *) return 1 ;;
  esac
}
sim2real_environment_dir() {
  printf '%s/.venvs/%s\n' "$2" "$(sim2real_environment_slug "$1")"
}
sim2real_environments_root() { printf '%s/.venvs\n' "$1"; }
sim2real_python_executable() {
  printf '%s/bin/python\n' "$(sim2real_environment_dir "$1" "$2")"
}
sim2real_uv_extra_args() { :; }
sim2real_configure_accelerator_env() { :; }
sim2real_profile_file() { printf '%s/.sim2real/accelerator\n' "$1"; }
sim2real_saved_accelerator() {
  local profile
  profile="$(sim2real_profile_file "$1")"
  [[ -r "${profile}" ]] || return 1
  head -n 1 "${profile}"
}
sim2real_save_accelerator() {
  printf 'save:%s\n' "$1" >>"${TEST_EVENTS}"
  mkdir -p "$2/.sim2real"
  printf '%s\n' "$1" >"$2/.sim2real/accelerator.tmp"
  mv "$2/.sim2real/accelerator.tmp" "$2/.sim2real/accelerator"
}
''',
            encoding="utf-8",
        )
        doctor = scripts_dir / "doctor.sh"
        doctor.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s\\n' doctor >>\"${TEST_EVENTS}\"\n"
            "exit \"${FAKE_DOCTOR_STATUS}\"\n",
            encoding="utf-8",
        )
        doctor.chmod(0o755)
        (fixture_root / "pyproject.toml").write_text(
            "[project]\nname='fixture'\nversion='0.0.0'\n",
            encoding="utf-8",
        )
        (fixture_root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        previous_environment = fixture_root / ".venvs/cpu"
        previous_environment.mkdir(parents=True)
        previous_sentinel = previous_environment / "previous-environment"
        previous_sentinel.write_text("keep me\n", encoding="utf-8")
        fake_uv = local_bin / "uv"
        fake_uv.write_text(
            r'''#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then
  printf '%s\n' 'uv 0.11.8'
  exit 0
fi
printf 'uv:%s\n' "$*" >>"${TEST_EVENTS}"
if [[ "${1:-}" == "sync" ]]; then
  [[ "${UV_PROJECT_ENVIRONMENT:-}" == "${TEST_REPOSITORY}/.venvs/cpu" ]] || exit 72
  mkdir -p "${UV_PROJECT_ENVIRONMENT}/bin"
  ln -sf "$(command -v python3)" "${UV_PROJECT_ENVIRONMENT}/bin/python"
fi
exit 0
''',
            encoding="utf-8",
        )
        fake_uv.chmod(0o755)

        environment = os.environ.copy()
        environment.update(
            {
                "HOME": str(Path(temporary) / "home"),
                "TEST_EVENTS": str(events),
                "TEST_REPOSITORY": str(fixture_root),
                "FAKE_DOCTOR_STATUS": doctor_status,
            }
        )
        completed = subprocess.run(
            [
                "bash",
                str(scripts_dir / "install.sh"),
                "--accelerator",
                requested_accelerator,
                "--skip-system-packages",
                *extra_arguments,
            ],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, expected_status, completed.stderr)
        lifecycle = events.read_text(encoding="utf-8").splitlines() if events.exists() else []
        lifecycle = [event for event in lifecycle if event == "doctor" or event.startswith("save:")]
        self.assertEqual(lifecycle, expected_lifecycle)
        active_profile = fixture_root / ".sim2real/accelerator"
        if expected_lifecycle == ["doctor", "save:cpu"]:
          self.assertEqual(active_profile.read_text(encoding="utf-8"), "cpu\n")
          self.assertFalse(previous_sentinel.exists())
        else:
          self.assertFalse(active_profile.exists())
          self.assertEqual(previous_sentinel.read_text(encoding="utf-8"), "keep me\n")

  def test_native_nvidia_support_label_does_not_claim_physical_validation(self) -> None:
    lib = ROOT / "scripts/lib/platform.sh"
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f"source {shlex.quote(str(lib))}; sim2real_support_level nvidia ubuntu",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertEqual(completed.returncode, 0, completed.stderr)
    label = completed.stdout.strip().lower()
    self.assertNotIn("validado-proyecto", label)
    self.assertIn("sin validacion fisica", label)

  def test_wsl2_and_uv_contract(self) -> None:
    platform_source = (ROOT / "scripts/lib/platform.sh").read_text(encoding="utf-8")
    self.assertIn("sim2real_detect_wsl_version", platform_source)
    self.assertIn("WSL1 no es compatible", platform_source)

    installer = (ROOT / "scripts/install.sh").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts/sim2real.sh").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/comprobaciones.yml").read_text(encoding="utf-8")
    for source in (installer, launcher):
      self.assertIn('0.11.8', source)
      self.assertIn('UV_NO_MODIFY_PATH=1', source)
    self.assertIn('version: "0.11.8"', workflow)

  def test_ubuntu_ci_and_public_cpu_bootstrap_run_the_full_acceptance(self) -> None:
    workflow = (ROOT / ".github/workflows/comprobaciones.yml").read_text(encoding="utf-8")
    for required in (
        "Instalación nueva de Ubuntu sobre CPU",
        "ubuntu-22.04",
        "ubuntu-24.04",
        "./scripts/install.sh --accelerator cpu",
        "./scripts/doctor.sh",
        "./scripts/sim2real.sh test-mjx --steps 10",
        "./scripts/sim2real.sh visualizar-modelo-preentrenado --solo-comprobar",
        "Autoinstalador público unificado",
        "SIM2REAL_PUBLIC_REPOSITORY: GuillermoRosale5/Sim2Real-MJX-JAX-sobre-Linux-WSL",
        "raw.githubusercontent.com/${SIM2REAL_PUBLIC_REPOSITORY}/${GITHUB_SHA}",
        "mktemp \"${RUNNER_TEMP}/sim2real-bootstrap.",
        "curl --proto '=https' --tlsv1.2 --fail --show-error --location",
        "bash -n \"${bootstrap_file}\"",
        "bash \"${bootstrap_file}\"",
        "--accelerator cpu",
        "--install-path \"${HOME}/sim2real-bootstrap-ci\"",
        "--repo-url \"https://github.com/${SIM2REAL_PUBLIC_REPOSITORY}.git\"",
    ):
      self.assertIn(required, workflow)
    self.assertNotIn("Instalación nativa nueva", workflow)

    public_job = workflow.split("  public-unified-bootstrap:", maxsplit=1)[1]
    self.assertNotIn("actions/checkout", public_job)
    self.assertNotRegex(public_job, re.compile(r"curl[^\n]*\|\s*bash"))
    self.assertLess(
        public_job.index("curl --proto"),
        public_job.index("bash -n \"${bootstrap_file}\""),
    )
    self.assertLess(
        public_job.index("bash -n \"${bootstrap_file}\""),
        public_job.index("bash \"${bootstrap_file}\""),
    )

  def test_windows_installer_has_public_windows_ci(self) -> None:
    workflow = (ROOT / ".github/workflows/comprobaciones.yml").read_text(encoding="utf-8")
    windows_job = workflow.split("  windows-installer:", maxsplit=1)[1].split(
        "\n  fresh-ubuntu-cpu-install:", maxsplit=1
    )[0]
    self.assertIn("runs-on: windows-latest", windows_job)
    self.assertIn("shell: powershell", windows_job)
    self.assertIn(r".\tests\test_install_windows.ps1", windows_job)

  def test_unified_bootstrap_accepts_wsl2_and_rejects_wsl1(self) -> None:
    if os.geteuid() == 0:
      self.skipTest("La proteccion contra root usa el EUID real")

    bootstrap = ROOT / "scripts/bootstrap_ubuntu.sh"
    command = r'''
      source "${TEST_BOOTSTRAP}"
      sim2real_bootstrap_is_wsl() { return 0; }
      sim2real_bootstrap_detect_wsl_version() {
        printf '%s\n' "${FAKE_WSL_VERSION}"
      }
      sim2real_bootstrap_os_id() { printf '%s\n' ubuntu; }
      sim2real_bootstrap_os_version() { printf '%s\n' 24.04; }
      uname() {
        case "${1:-}" in
          -s) printf '%s\n' Linux ;;
          -m) printf '%s\n' x86_64 ;;
          *) return 1 ;;
        esac
      }
      command() {
        if [[ "${1:-}" == -v &&
              ( "${2:-}" == sudo || "${2:-}" == apt-get ) ]]; then
          return 0
        fi
        builtin command "$@"
      }
      sim2real_bootstrap_validate_host
    '''
    base_env = os.environ.copy()
    base_env["TEST_BOOTSTRAP"] = str(bootstrap)

    wsl2 = subprocess.run(
        ["bash", "-c", command],
        env=base_env | {"FAKE_WSL_VERSION": "2"},
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertEqual(wsl2.returncode, 0, wsl2.stderr)

    wsl1 = subprocess.run(
        ["bash", "-c", command],
        env=base_env | {"FAKE_WSL_VERSION": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertEqual(wsl1.returncode, 10, wsl1.stderr)
    self.assertIn("WSL1 no es compatible", wsl1.stderr)

  def test_wsl_nvidia_bootstrap_never_invokes_linux_driver_management(self) -> None:
    bootstrap = ROOT / "scripts/bootstrap_ubuntu.sh"
    with tempfile.TemporaryDirectory() as temporary:
      temp_root = Path(temporary)
      calls = temp_root / "sudo-calls.txt"
      forbidden = temp_root / "linux-driver-was-invoked"
      env = os.environ.copy()
      env.update(
          {
              "TEST_BOOTSTRAP": str(bootstrap),
              "TEST_CALLS": str(calls),
              "TEST_FORBIDDEN": str(forbidden),
              "TEST_INSTALL": str(temp_root / "install"),
          }
      )
      command = r'''
        source "${TEST_BOOTSTRAP}"
        sim2real_bootstrap_validate_host() { :; }
        sim2real_bootstrap_host_kind() { printf '%s\n' wsl; }
        sim2real_bootstrap_add_wsl_driver_path() { :; }
        sim2real_bootstrap_resolve_install_path() { printf '%s\n' "$1"; }
        sim2real_bootstrap_validate_wsl_install_path() { :; }
        sudo() {
          printf '%s\n' "$*" >>"${TEST_CALLS}"
          case "$*" in
            *ubuntu-drivers*|*nvidia*|*mokutil*|*dkms*|*linux-modules*)
              : >"${TEST_FORBIDDEN}"
              return 97
              ;;
          esac
          return 0
        }
        apt-get() {
          printf 'direct-apt-get %s\n' "$*" >>"${TEST_CALLS}"
          return 0
        }
        ubuntu-drivers() { : >"${TEST_FORBIDDEN}"; return 97; }
        mokutil() { : >"${TEST_FORBIDDEN}"; return 97; }
        sim2real_bootstrap_resolve_accelerator() {
          [[ "$1" == nvidia && "$2" == wsl ]] || return 98
          printf '%s\n' nvidia
        }
        sim2real_bootstrap_nvidia_smi_command() { printf '%s\n' /bin/true; }
        sim2real_bootstrap_nvidia_driver_version() { printf '%s\n' 550.120; }
        sim2real_bootstrap_install_nvidia_management_packages() {
          : >"${TEST_FORBIDDEN}"
          return 97
        }
        sim2real_bootstrap_prepare_nvidia() {
          : >"${TEST_FORBIDDEN}"
          return 97
        }
        sim2real_bootstrap_sync_repository() { :; }
        sim2real_bootstrap_run_acceptance() { :; }

        sim2real_bootstrap_main \
          --accelerator nvidia \
          --install-path "${TEST_INSTALL}"
        [[ ! -e "${TEST_FORBIDDEN}" ]]
      '''
      completed = subprocess.run(
          ["bash", "-c", command],
          env=env,
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)
      sudo_calls = calls.read_text(encoding="utf-8")
      for forbidden_token in (
          "ubuntu-drivers",
          "nvidia",
          "mokutil",
          "dkms",
          "linux-modules",
      ):
        self.assertNotIn(forbidden_token, sudo_calls)
      for common_package in ("ca-certificates", "curl", "git", "pciutils"):
        self.assertIn(common_package, sudo_calls)
      self.assertIn("Driver NVIDIA de Windows visible en WSL2", completed.stdout)

  def test_native_nvidia_dispatch_keeps_ubuntu_driver_path_covered(self) -> None:
    bootstrap = ROOT / "scripts/bootstrap_ubuntu.sh"
    with tempfile.TemporaryDirectory() as temporary:
      temp_root = Path(temporary)
      calls = temp_root / "native-calls.txt"
      forbidden = temp_root / "wsl-validator-was-invoked"
      env = os.environ.copy()
      env.update(
          {
              "TEST_BOOTSTRAP": str(bootstrap),
              "TEST_CALLS": str(calls),
              "TEST_FORBIDDEN": str(forbidden),
              "TEST_INSTALL": str(temp_root / "install"),
          }
      )
      command = r'''
        source "${TEST_BOOTSTRAP}"
        sim2real_bootstrap_validate_host() { :; }
        sim2real_bootstrap_host_kind() { printf '%s\n' native; }
        sim2real_bootstrap_resolve_install_path() { printf '%s\n' "$1"; }
        sudo() { printf '%s\n' "$*" >>"${TEST_CALLS}"; }
        apt-get() {
          printf 'direct-apt-get %s\n' "$*" >>"${TEST_CALLS}"
          return 0
        }
        sim2real_bootstrap_resolve_accelerator() {
          [[ "$1" == nvidia && "$2" == native ]] || return 98
          printf '%s\n' nvidia
        }
        sim2real_bootstrap_validate_nvidia_wsl() {
          : >"${TEST_FORBIDDEN}"
          return 97
        }
        sim2real_bootstrap_prepare_nvidia() {
          printf '%s\n' prepare-native-nvidia >>"${TEST_CALLS}"
        }
        sim2real_bootstrap_sync_repository() { :; }
        sim2real_bootstrap_run_acceptance() { :; }

        sim2real_bootstrap_main \
          --accelerator nvidia \
          --install-path "${TEST_INSTALL}"
        [[ ! -e "${TEST_FORBIDDEN}" ]]
      '''
      completed = subprocess.run(
          ["bash", "-c", command],
          env=env,
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)
      native_calls = calls.read_text(encoding="utf-8")
      self.assertIn("ubuntu-drivers-common", native_calls)
      self.assertIn("mokutil", native_calls)
      self.assertIn("prepare-native-nvidia", native_calls)

  def test_virtualbox_without_gpu_passthrough_is_diagnosed(self) -> None:
    bootstrap = ROOT / "scripts/bootstrap_ubuntu.sh"
    common_setup = r'''
      source "${TEST_BOOTSTRAP}"
      sim2real_bootstrap_has_pci_display_vendor() { return 1; }
      sim2real_bootstrap_nvidia_driver_ready() { return 1; }
      sim2real_bootstrap_report_pci_graphics() { :; }
      sim2real_bootstrap_detect_virtualization() { printf '%s\n' oracle; }
    '''
    env = os.environ.copy()
    env["TEST_BOOTSTRAP"] = str(bootstrap)

    explicit_nvidia = subprocess.run(
        [
            "bash",
            "-c",
            common_setup
            + "sim2real_bootstrap_resolve_accelerator nvidia native\n",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertEqual(explicit_nvidia.returncode, 40, explicit_nvidia.stderr)
    diagnostic = explicit_nvidia.stderr.lower()
    self.assertIn("virtualbox", diagnostic)
    self.assertIn("no expone una gpu nvidia pci", diagnostic)
    self.assertIn("no puede crear ese acceso", diagnostic)
    self.assertIn("--accelerator cpu", diagnostic)

    automatic = subprocess.run(
        [
            "bash",
            "-c",
            common_setup + "sim2real_bootstrap_resolve_accelerator auto native\n",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertNotEqual(automatic.returncode, 0, automatic.stderr)
    self.assertNotEqual(automatic.stdout.strip(), "cpu")
    auto_diagnostic = automatic.stderr.lower()
    self.assertIn("virtualbox", auto_diagnostic)
    self.assertRegex(auto_diagnostic, r"gpu|acelerador")
    self.assertIn("--accelerator cpu", auto_diagnostic)

  def test_documented_bootstrap_main_matches_the_script(self) -> None:
    bootstrap = ROOT / "scripts/bootstrap_ubuntu.sh"
    expected_sha256 = hashlib.sha256(bootstrap.read_bytes()).hexdigest()
    release_path = (
        "Sim2Real-MJX-JAX-sobre-Linux-WSL/main/scripts/bootstrap_ubuntu.sh"
    )
    for relative_path in ("README.md", "docs/INSTALACION.md"):
      with self.subTest(path=relative_path):
        documentation = (ROOT / relative_path).read_text(encoding="utf-8")
        self.assertIn(release_path, documentation)
        self.assertIn(expected_sha256, documentation)
        self.assertNotIn("BOOTSTRAP_RELEASE_TODO", documentation)
        self.assertNotIn("BOOTSTRAP_SHA256_TODO", documentation)
        self.assertNotIn("BOOTSTRAP_SHA256_PENDIENTE", documentation)

  def test_documented_windows_installer_matches_the_script(self) -> None:
    installer_bytes = (ROOT / "scripts/install_windows.ps1").read_bytes()
    # Git normaliza el blob publicado a LF aunque el checkout de PowerShell use
    # CRLF. Get-FileHash comprueba el archivo que entrega raw.githubusercontent.
    published_bytes = installer_bytes.replace(b"\r\n", b"\n")
    expected_sha256 = hashlib.sha256(published_bytes).hexdigest()
    release_path = (
        "Sim2Real-MJX-JAX-sobre-Linux-WSL/main/scripts/install_windows.ps1"
    )
    for relative_path in ("README.md", "docs/INSTALACION.md"):
      with self.subTest(path=relative_path):
        documentation = (ROOT / relative_path).read_text(encoding="utf-8")
        self.assertIn(release_path, documentation)
        self.assertIn(expected_sha256, documentation)
        self.assertIn("Get-FileHash -Algorithm SHA256", documentation)

  def test_documented_bootstrap_is_one_group_before_sudo_prompts(self) -> None:
    for relative_path in ("README.md", "docs/INSTALACION.md"):
      with self.subTest(path=relative_path):
        documentation = (ROOT / relative_path).read_text(encoding="utf-8")
        raw_url = documentation.index(
            "Sim2Real-MJX-JAX-sobre-Linux-WSL/main/scripts/bootstrap_ubuntu.sh"
        )
        block_start = documentation.rfind("```bash", 0, raw_url)
        block_end = documentation.index("```", raw_url)
        self.assertNotEqual(block_start, -1)
        block = documentation[block_start:block_end]
        self.assertIn("```bash\n(\n  set -e\n  sudo -v", block)
        self.assertLess(block.index("sudo -v"), block.index("sudo apt-get update"))
        self.assertLess(block.index("sudo apt-get update"), block.index("curl --proto"))
        self.assertTrue(block.rstrip().endswith(")"))

  def test_clone_tutorials_create_the_parent_directory(self) -> None:
    for relative_path in ("README.md", "docs/INSTALACION.md"):
      with self.subTest(path=relative_path):
        lines = (ROOT / relative_path).read_text(encoding="utf-8").splitlines()
        clone_lines = [
            index
            for index, line in enumerate(lines)
            if "git clone " in line and "~/robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL" in line
        ]
        self.assertTrue(clone_lines, f"No hay ejemplo de clonacion en {relative_path}")
        for index in clone_lines:
          nearby_setup = "\n".join(lines[max(0, index - 8):index])
          self.assertIn("mkdir -p ~/robotica", nearby_setup)

  def test_wsl_rejects_windows_filesystem_at_custom_mount_root(self) -> None:
    lib = ROOT / "scripts/lib/platform.sh"
    with tempfile.TemporaryDirectory() as temp:
      temp_root = Path(temp)
      workspace = temp_root / "windows-custom-root/project"
      fake_bin = temp_root / "bin"
      workspace.mkdir(parents=True)
      fake_bin.mkdir()
      fake_findmnt = fake_bin / "findmnt"
      fake_findmnt.write_text("#!/usr/bin/env bash\nprintf '9p\\n'\n", encoding="utf-8")
      fake_findmnt.chmod(0o755)
      command = f'''
        source {shlex.quote(str(lib))}
        sim2real_detect_host() {{ printf 'wsl\\n'; }}
        ! sim2real_require_workspace {shlex.quote(str(workspace))}
      '''
      env = os.environ.copy()
      env["PATH"] = f"{fake_bin}:{env['PATH']}"
      completed = subprocess.run(
          ["bash", "-c", command],
          env=env,
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)

    source = lib.read_text(encoding="utf-8")
    for filesystem_type in ("drvfs", "9p", "v9fs", "virtiofs"):
      self.assertIn(filesystem_type, source)

  def test_powershell_bootstrap_passes_values_as_arguments(self) -> None:
    entrypoint = ROOT / "scripts/install_windows.ps1"
    bootstrap = entrypoint.read_text(encoding="utf-8")
    self.assertIn(
        '[ValidateSet("compatible", "auto", "nvidia", "amd", "intel", "cpu")]',
        bootstrap,
    )
    self.assertIn('[string]$Accelerator = "compatible"', bootstrap)
    self.assertIn("bash $linuxScript @ArgumentList", bootstrap)
    self.assertIn("SIM2REAL_TEMP_SCRIPT_WINDOWS/p", bootstrap)
    self.assertIn('[Text.UTF8Encoding]::new($false)', bootstrap)
    self.assertIn("ForEach-Object", bootstrap)
    self.assertIn("Write-Host $line", bootstrap)
    self.assertIn("__SIM2REAL_REPO_PATH__=", bootstrap)
    self.assertNotRegex(bootstrap, r"\$output\s*=\s*\$Script\s*\|")
    self.assertNotIn("AllowExperimentalAmdWsl", bootstrap)
    self.assertNotIn("eval echo", bootstrap)
    self.assertNotIn("repo_url='$RepoUrl'", bootstrap)
    self.assertNotIn("cd '$repoPath'", bootstrap)

  def test_bootstrap_rejects_foreign_git_repo_before_mutating_remote(self) -> None:
    bootstrap = (ROOT / "scripts/install_windows.ps1").read_text(encoding="utf-8")
    match = re.search(r"\$cloneScript = @'\n(.*?)\n'@", bootstrap, re.DOTALL)
    self.assertIsNotNone(match)
    clone_script = match.group(1)
    self.assertLess(
        clone_script.index(".sim2real-repository"),
        clone_script.index("remote set-url"),
    )

    with tempfile.TemporaryDirectory() as temp:
      foreign_repo = Path(temp) / "foreign-repository"
      foreign_repo.mkdir()
      subprocess.run(["git", "init", "--quiet", str(foreign_repo)], check=True)
      original_remote = "https://example.invalid/original.git"
      subprocess.run(
          ["git", "-C", str(foreign_repo), "remote", "add", "origin", original_remote],
          check=True,
      )
      completed = subprocess.run(
          [
              "bash",
              "-s",
              "--",
              "https://example.invalid/sim2real.git",
              str(foreign_repo),
          ],
          input=clone_script,
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertNotEqual(completed.returncode, 0)
      self.assertIn("no es Sim2Real MJX-JAX", completed.stderr)
      remote_after = subprocess.check_output(
          ["git", "-C", str(foreign_repo), "remote", "get-url", "origin"],
          text=True,
      ).strip()
      self.assertEqual(remote_after, original_remote)

  def test_single_entrypoints_and_warp_rejection(self) -> None:
    obsolete_bridges = (
        "docs/ACCELERATORS.md",
        "docs/ARCHITECTURE.md",
        "docs/COMMANDS.md",
        "docs/INSTALLATION.md",
        "scripts/diagnostico.sh",
        "scripts/instalar.sh",
        "scripts/instalar_linux.sh",
        "scripts/instalar_windows.ps1",
        "scripts/instalar_wsl.sh",
        "scripts/install_linux.sh",
        "scripts/install_wsl.sh",
        "scripts/pruebas_estaticas.sh",
        "scripts/script_instalador_wsl_jax_mujoco_playground.ps1",
        "scripts/sim2real_wsl.sh",
        "scripts/curriculum_auto_sim2real.sh",
    )
    for path in obsolete_bridges:
      with self.subTest(path=path):
        self.assertFalse((ROOT / path).exists(), f"Sigue existiendo el puente {path}")

    for path in (
        "scripts/install.sh",
        "scripts/doctor.sh",
        "scripts/test_static.sh",
        "scripts/sim2real.sh",
    ):
      with self.subTest(path=path):
        entrypoint = ROOT / path
        self.assertTrue(entrypoint.is_file())
        self.assertTrue(os.access(entrypoint, os.X_OK), f"No es ejecutable: {path}")

    installer = (ROOT / "scripts/install.sh").read_text(encoding="utf-8")
    self.assertIn("--skip-system-packages", installer)
    for obsolete_option in (
        "--no-system-packages",
        "--skip-doctor",
        "--skip-gpu-check",
    ):
      self.assertNotIn(obsolete_option, installer)

    windows_installer = (ROOT / "scripts/install_windows.ps1").read_text(
        encoding="utf-8"
    )
    self.assertNotIn("SkipDoctor", windows_installer)
    self.assertNotIn("AllowExperimentalAmdWsl", windows_installer)

    launcher = (ROOT / "scripts/sim2real.sh").read_text(encoding="utf-8")
    self.assertIn("main()", launcher)
    self.assertRegex(launcher, re.compile(r"validate_impl\(\).*?MJX-JAX", re.DOTALL))
    self.assertIn("curriculo-automatico)", launcher)
    self.assertNotIn("curriculum-auto", launcher)

    training_section = launcher.split("ejecutar_entrenamiento() {", 1)[1].split(
        "parar_entrenamiento() {", 1
    )[0]
    for spanish_option in (
        "--segundo-plano",
        "--nombre-ejecucion",
        "--continuar-ultimo",
        "--anexar-csv",
        "--omitir-prueba-mjx",
        "--desde-cero",
    ):
      self.assertIn(spanish_option, training_section)
    for obsolete_option in (
        "--background",
        "--resume-latest",
        "--append-csv",
        "--skip-test-mjx",
        "--reset-checkpoint",
    ):
      self.assertNotIn(obsolete_option, training_section)
    for technical_option in (
        "--setup",
        "--seed",
        "--num-envs",
        "--episode-length",
        "--load-checkpoint-path",
    ):
      self.assertIn(technical_option, training_section)

    supervisor_curriculo = (
        ROOT / "scripts/curriculo_automatico_sim2real.sh"
    ).read_text(encoding="utf-8")
    for spanish_option in (
        "--nombre-ejecucion",
        "--pasos-totales",
        "--pasos-por-bloque",
        "--sin-preparacion",
        "--sin-reinicio-inicial",
        "--sin-parada-al-solicitar",
    ):
      self.assertIn(spanish_option, supervisor_curriculo)
    for obsolete_option in (
        "--run-name",
        "--total-steps",
        "--chunk-steps",
        "--no-setup",
        "--no-reset-first",
        "--no-stop-on-request",
    ):
      self.assertNotIn(obsolete_option, supervisor_curriculo)
    self.assertIn("curriculo_automatico_estado.json", supervisor_curriculo)
    self.assertNotIn("curriculum_auto_estado.json", supervisor_curriculo)

    graficador = (ROOT / "scripts/graficar_recompensas.py").read_text(
        encoding="utf-8"
    )
    visor_directo = (ROOT / "scripts/ver_recompensas_en_directo.py").read_text(
        encoding="utf-8"
    )
    for spanish_option in (
        "--directorio-registros",
        "--directorio-ejecucion",
        "--ruta-csv",
        "--salida",
        "--eje-x",
        "--columnas",
        "--origen",
        "--segmento",
        "--suavizado",
        "--unidades-recompensa",
    ):
      self.assertIn(spanish_option, graficador)
      self.assertIn(spanish_option, visor_directo)
    self.assertIn("--modo-csv", graficador)
    for spanish_option in (
        "--intervalo",
        "--ultimas-filas",
        "--sin-ventana",
        "--una-vez",
    ):
      self.assertIn(spanish_option, visor_directo)
    for obsolete_option in (
        "--logs_dir",
        "--run_dir",
        "--csv_path",
        "--csv_mode",
        "--output",
        "--columns",
        "--source",
        "--segment",
        "--smooth",
        "--show",
        "--reward_units",
        "--unidades_recompensa",
    ):
      patron_opcion = re.compile(rf'["\']{re.escape(obsolete_option)}["\']')
      self.assertNotRegex(graficador, patron_opcion)
      self.assertNotRegex(visor_directo, patron_opcion)
    for obsolete_option in ("--interval", "--tail", "--no_window", "--once"):
      patron_opcion = re.compile(rf'["\']{re.escape(obsolete_option)}["\']')
      self.assertNotRegex(visor_directo, patron_opcion)

    visualizador_mjx = (
        ROOT / "scripts/visualizar_resultados_mjx.py"
    ).read_text(encoding="utf-8")
    for spanish_option in (
        "--directorio-registros",
        "--ruta-checkpoint",
        "--indice-checkpoint",
        "--longitud-episodio",
        "--espera",
        "--pausa-episodio",
        "--semilla",
        "--ruta-xml",
        "--postura-inicial",
        "--reinicio-automatico",
        "--congelar-al-terminar",
        "--solo-comprobar",
    ):
      self.assertIn(spanish_option, visualizador_mjx)
    self.assertIn('parser.add_argument("--impl"', visualizador_mjx)
    for obsolete_option in (
        "--logs_dir",
        "--checkpoint_path",
        "--checkpoint_index",
        "--episode_length",
        "--sleep",
        "--episode_pause",
        "--seed",
        "--xml_path",
        "--reset_preset",
        "--auto_reset",
        "--freeze_on_done",
        "--dry_run",
    ):
      self.assertNotIn(obsolete_option, visualizador_mjx)

    self.assertIn("visualizar-modelo-preentrenado)", launcher)
    self.assertNotIn("view-pretrained", launcher)

    for help_option in ("-h", "--help"):
      with self.subTest(help_option=help_option):
        completed = subprocess.run(
            ["bash", str(ROOT / "scripts/sim2real.sh"), help_option],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Uso:", completed.stdout)

    powershell_installer = (ROOT / "scripts/install_windows.ps1").read_text(
        encoding="utf-8"
    )
    self.assertIn("function Invoke-Wsl", powershell_installer)
    self.assertIn("$cloneScript = @'", powershell_installer)
    self.assertNotIn(
        "script_instalador_wsl_jax_mujoco_playground.ps1", powershell_installer
    )

  def test_run_paths_reject_traversal_and_symlinks(self) -> None:
    lib = ROOT / "scripts/lib/platform.sh"
    with tempfile.TemporaryDirectory() as temp:
      fake_repo = Path(temp) / "repo"
      outside = Path(temp) / "outside"
      fake_repo.mkdir()
      outside.mkdir()
      command = f"""
        source {lib!s}
        ! sim2real_safe_run_dir {fake_repo!s} ../../escape >/dev/null 2>&1
        mkdir -p {fake_repo!s}/logs_sim2real_mjx
        ln -s {outside!s} {fake_repo!s}/logs_sim2real_mjx/linked
        ! sim2real_safe_run_dir {fake_repo!s} linked >/dev/null 2>&1
      """
      completed = subprocess.run(
          ["bash", "-c", command], text=True, capture_output=True, check=False
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)

    trainer = (ROOT / "sim2real_mjx/entrenar_ppo_mjx.py").read_text(encoding="utf-8")
    self.assertIn("def _safe_run_dir", trainer)
    self.assertIn("run_dir.parent != base_logdir", trainer)

  def test_python_run_and_checkpoint_symlinks_are_rejected(self) -> None:
    trainer_path = ROOT / "sim2real_mjx/entrenar_ppo_mjx.py"
    parsed = ast.parse(trainer_path.read_text(encoding="utf-8"))
    selected_nodes: list[ast.stmt] = []
    for node in parsed.body:
      if isinstance(node, ast.Assign) and any(
          isinstance(target, ast.Name) and target.id == "_RUN_NAME_RE"
          for target in node.targets
      ):
        selected_nodes.append(node)
      elif isinstance(node, ast.FunctionDef) and node.name in {
          "_safe_run_dir",
          "_prepare_checkpoint_dir",
      }:
        selected_nodes.append(node)
    self.assertEqual(len(selected_nodes), 3)
    namespace = {"Path": Path, "re": re, "shutil": shutil}
    extracted = ast.fix_missing_locations(ast.Module(body=selected_nodes, type_ignores=[]))
    exec(compile(extracted, str(trainer_path), "exec"), namespace)

    with tempfile.TemporaryDirectory() as temp:
      logs = (Path(temp) / "logs").resolve()
      real_run = logs / "real-run"
      alias_run = logs / "alias-run"
      real_run.mkdir(parents=True)
      alias_run.symlink_to(real_run, target_is_directory=True)
      with self.assertRaisesRegex(ValueError, r"enlace simb[oó]lico"):
        namespace["_safe_run_dir"](logs, alias_run.name)

      checkpoint_target = Path(temp) / "checkpoint-target"
      checkpoint_target.mkdir()
      marker = checkpoint_target / "must-survive.txt"
      marker.write_text("safe\n", encoding="utf-8")
      (real_run / "checkpoints").symlink_to(
          checkpoint_target, target_is_directory=True
      )
      with self.assertRaisesRegex(ValueError, r"enlace simb[oó]lico"):
        namespace["_prepare_checkpoint_dir"](real_run, True)
      self.assertEqual(marker.read_text(encoding="utf-8"), "safe\n")

  def test_training_pid_identity_matches_exact_command(self) -> None:
    launcher = ROOT / "scripts/sim2real.sh"
    with tempfile.TemporaryDirectory() as temp:
      fake_repo = Path(temp) / "repo"
      run_dir = fake_repo / "logs_sim2real_mjx" / "valid-run"
      stale_run = fake_repo / "logs_sim2real_mjx" / "stale-run"
      python_path = fake_repo / ".venvs/nvidia-cuda12/bin/python"
      trainer = fake_repo / "sim2real_mjx/entrenar_ppo_mjx.py"
      guard_script = fake_repo / "scripts/proteccion_termica.sh"
      run_dir.mkdir(parents=True)
      stale_run.mkdir(parents=True)
      python_path.parent.mkdir(parents=True)
      state_dir = fake_repo / ".sim2real"
      state_dir.mkdir()
      (state_dir / "accelerator").write_text("nvidia\n", encoding="utf-8")
      trainer.parent.mkdir(parents=True)
      guard_script.parent.mkdir(parents=True)
      python_path.symlink_to(sys.executable)
      trainer.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
      guard_script.write_text(
          "#!/usr/bin/env bash\nwhile :; do sleep 1; done\n", encoding="utf-8"
      )
      guard_script.chmod(0o755)
      process = subprocess.Popen(
          [
              str(python_path),
              str(trainer),
              "--logdir",
              str(run_dir.parent),
              "--run_name",
              run_dir.name,
          ]
      )
      guard = subprocess.Popen(
          [
              str(guard_script),
              "--pid",
              str(process.pid),
              "--run-dir",
              str(run_dir),
          ]
      )
      innocent = subprocess.Popen(["sleep", "30"])
      try:
        (run_dir / "entrenamiento.pid").write_text(
            f"{process.pid}\n", encoding="utf-8"
        )
        (run_dir / "proteccion_termica.pid").write_text(
            f"{guard.pid}\n", encoding="utf-8"
        )
        (stale_run / "entrenamiento.pid").write_text(
            f"{innocent.pid}\n", encoding="utf-8"
        )
        (run_dir.parent / "ultima_ejecucion.txt").write_text(
            f"{stale_run}\n", encoding="utf-8"
        )
        command = (
            f"source {shlex.quote(str(launcher))}\n"
            f"REPO_ROOT={shlex.quote(str(fake_repo))}\n"
            f"LOGS_DIR={shlex.quote(str(run_dir.parent))}\n"
            f"ULTIMA_EJECUCION={shlex.quote(str(run_dir.parent / 'ultima_ejecucion.txt'))}\n"
            f"sim2real_training_pid_matches_run "
            f"{shlex.quote(str(fake_repo))} {shlex.quote(str(run_dir))} {process.pid}\n"
            f"sim2real_thermal_guard_pid_matches_run "
            f"{shlex.quote(str(fake_repo))} {shlex.quote(str(run_dir))} {guard.pid} {process.pid}\n"
            "active=\"$(ejecucion_activa_con_pid)\"\n"
            f"[[ \"${{active}}\" == $'{run_dir}\\t{process.pid}' ]]\n"
            "guard_output=\"$(iniciar_proteccion_termica_actual)\"\n"
            f"[[ \"${{guard_output}}\" == *'PID {guard.pid}'* ]]"
        )
        completed = subprocess.run(
            ["bash", "-c", command], text=True, capture_output=True, check=False
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
      finally:
        innocent.terminate()
        try:
          innocent.wait(timeout=5)
        except subprocess.TimeoutExpired:
          innocent.kill()
          innocent.wait(timeout=5)
        guard.terminate()
        try:
          guard.wait(timeout=5)
        except subprocess.TimeoutExpired:
          guard.kill()
          guard.wait(timeout=5)
        process.terminate()
        try:
          process.wait(timeout=5)
        except subprocess.TimeoutExpired:
          process.kill()
          process.wait(timeout=5)

  def test_stop_rejects_reused_pid_without_signalling(self) -> None:
    launcher = ROOT / "scripts/sim2real.sh"
    with tempfile.TemporaryDirectory() as temp:
      fake_repo = Path(temp) / "repo"
      run_dir = fake_repo / "logs_sim2real_mjx" / "stale-run"
      run_dir.mkdir(parents=True)
      env = os.environ.copy()
      env.update(
          {
              "TEST_LAUNCHER": str(launcher),
              "TEST_REPO": str(fake_repo),
              "TEST_RUN": str(run_dir),
          }
      )
      command = r'''
        source "${TEST_LAUNCHER}"
        REPO_ROOT="${TEST_REPO}"
        LOGS_DIR="${TEST_REPO}/logs_sim2real_mjx"
        ULTIMA_EJECUCION="${LOGS_DIR}/ultima_ejecucion.txt"
        sleep 30 &
        innocent_pid=$!
        cleanup() {
          kill "${innocent_pid}" >/dev/null 2>&1 || true
          wait "${innocent_pid}" >/dev/null 2>&1 || true
        }
        trap cleanup EXIT
        printf '%s\n' "${innocent_pid}" > "${TEST_RUN}/entrenamiento.pid"
        printf '%s\n' "${innocent_pid}" > "${TEST_RUN}/proteccion_termica.pid"
        printf '%s\n' "${TEST_RUN}" > "${ULTIMA_EJECUCION}"
        ! sim2real_thermal_guard_pid_matches_run \
          "${TEST_REPO}" "${TEST_RUN}" "${innocent_pid}" "${innocent_pid}"
        output="$(parar_entrenamiento 2>&1)"
        kill -0 "${innocent_pid}"
        [[ "${output}" == *"No hay PID guardado"* ]]
      '''
      completed = subprocess.run(
          ["bash", "-c", command],
          env=env,
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)

  def test_all_process_signals_use_identity_helpers(self) -> None:
    launcher = (ROOT / "scripts/sim2real.sh").read_text(encoding="utf-8")
    thermal = (ROOT / "scripts/proteccion_termica.sh").read_text(encoding="utf-8")
    for source in (launcher, thermal):
      self.assertNotRegex(source, re.compile(r"^\s*kill\s+(?!-0)", re.MULTILINE))
    self.assertIn("sim2real_signal_training_process", launcher)
    self.assertIn("sim2real_signal_thermal_guard", launcher)
    self.assertIn("sim2real_signal_training_process", thermal)


if __name__ == "__main__":
  unittest.main()
