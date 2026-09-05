from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "scripts/bootstrap_ubuntu.sh"
OFFICIAL_REPO = "https://github.com/GuillermoRosale5/Sim2Real-MJX-JAX-sobre-Linux-WSL.git"


def clean_test_environment(overrides: dict[str, str] | None = None) -> dict[str, str]:
  clean_env = os.environ.copy()
  inherited_state = {
      "BASH_ENV",
      "BASHOPTS",
      "CDPATH",
      "ENV",
      "GIT_ALTERNATE_OBJECT_DIRECTORIES",
      "GIT_CEILING_DIRECTORIES",
      "GIT_COMMON_DIR",
      "GIT_CONFIG_COUNT",
      "GIT_DIR",
      "GIT_DISCOVERY_ACROSS_FILESYSTEM",
      "GIT_INDEX_FILE",
      "GIT_OBJECT_DIRECTORY",
      "GIT_WORK_TREE",
      "SHELLOPTS",
      "WSL_DISTRO_NAME",
      "WSL_INTEROP",
  }
  for variable in tuple(clean_env):
    if (
        variable in inherited_state
        or variable.startswith("GIT_CONFIG_KEY_")
        or variable.startswith("GIT_CONFIG_VALUE_")
    ):
      clean_env.pop(variable, None)
  clean_env["GIT_CONFIG_NOSYSTEM"] = "1"
  clean_env["GIT_CONFIG_GLOBAL"] = os.devnull
  if overrides:
    clean_env.update(overrides)
  return clean_env


def run_bash(
    body: str,
    *arguments: str | Path,
    env: dict[str, str] | None = None,
    timeout: float = 10,
) -> subprocess.CompletedProcess[str]:
  command = [
      "bash",
      "-c",
      body,
      "bootstrap-test",
      str(BOOTSTRAP),
      *(str(argument) for argument in arguments),
  ]
  return subprocess.run(
      command,
      env=clean_test_environment(env),
      text=True,
      capture_output=True,
      check=False,
      timeout=timeout,
  )


class BootstrapUbuntuTests(unittest.TestCase):
  maxDiff = None

  def git(self, repository: Path, *arguments: str) -> str:
    env = os.environ.copy()
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertEqual(completed.returncode, 0, completed.stderr)
    return completed.stdout.strip()

  def make_repository(
      self,
      parent: Path,
      *,
      valid_identity: bool = True,
      origin: str = OFFICIAL_REPO,
  ) -> Path:
    repository = parent / "checkout"
    repository.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", "--initial-branch=main", str(repository)],
        text=True,
        capture_output=True,
        check=True,
    )
    self.git(repository, "config", "user.name", "Bootstrap tests")
    self.git(repository, "config", "user.email", "bootstrap@example.invalid")
    self.git(repository, "config", "commit.gpgsign", "false")
    self.git(repository, "config", "core.hooksPath", os.devnull)

    (repository / "payload.txt").write_text("contenido inicial\n", encoding="utf-8")
    if valid_identity:
      (repository / ".sim2real-repository").write_text(
          "sim2real-mjx-jax-linux-wsl-v1\n", encoding="utf-8"
      )
      (repository / "pyproject.toml").write_text(
          'name = "sim2real-mjx-jax"\n', encoding="utf-8"
      )
      scripts = repository / "scripts"
      scripts.mkdir()
      installer = scripts / "install.sh"
      installer.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
      installer.chmod(0o755)

    self.git(repository, "add", "--all")
    self.git(repository, "commit", "--quiet", "-m", "fixture")
    self.git(repository, "remote", "add", "origin", origin)
    return repository

  def run_sync_that_must_reject(
      self, repository: Path, expected_url: str, sentinel: Path
  ) -> subprocess.CompletedProcess[str]:
    return run_bash(
        r'''
          source "$1"
          repository="$2"
          sentinel="$3"
          expected_url="$4"
          git() {
            local argument
            for argument in "$@"; do
              case "${argument}" in
                fetch|merge|clone)
                  printf 'network-or-mutation-command: %s\n' "${argument}" >"${sentinel}"
                  return 97
                  ;;
              esac
            done
            command git "$@"
          }
          sim2real_bootstrap_sync_repository "${repository}" "${expected_url}"
        ''',
        repository,
        sentinel,
        expected_url,
    )

  def test_script_is_sourceable_without_enabling_shell_options_or_side_effects(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      home = Path(temporary) / "home"
      home.mkdir()
      completed = run_bash(
          r'''
            set +e
            set +u
            set +o pipefail
            trap_before="$(trap -p EXIT)"
            source "$1"
            trap_after="$(trap -p EXIT)"
            [[ "${trap_before}" == "${trap_after}" ]]
            [[ "$-" != *e* && "$-" != *u* ]]
            [[ "$(set -o | awk '$1 == "pipefail" {print $2}')" == "off" ]]
            declare -F sim2real_bootstrap_main >/dev/null
            declare -F sim2real_bootstrap_validate_checkout >/dev/null
            declare -F sim2real_bootstrap_sync_repository >/dev/null
          ''',
          env={"HOME": str(home)},
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)
      self.assertEqual(list(home.iterdir()), [])

    help_result = subprocess.run(
        ["bash", str(BOOTSTRAP), "--help"],
        env=clean_test_environment(),
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    self.assertEqual(help_result.returncode, 0, help_result.stderr)
    self.assertIn("--accelerator", help_result.stdout)

  def test_support_summary_is_visible_and_matches_the_public_policy(self) -> None:
    summary = run_bash(
        'source "$1"; sim2real_bootstrap_print_accelerator_support_summary'
    )
    self.assertEqual(summary.returncode, 0, summary.stderr)
    for expected in (
        "MODO UNIVERSAL RECOMENDADO",
        "--accelerator compatible",
        "puede preparar el driver de una NVIDIA visible por PCI",
        "selecciona CPU",
        "automaticamente con un aviso visible",
        "VALIDADO DE PRINCIPIO A FIN EN EL PROYECTO",
        "NVIDIA sobre WSL2 con Ubuntu 24.04",
        "IMPLEMENTADO, PERO SIN VALIDACION FISICA EN ESTE PROYECTO",
        "--accelerator nvidia",
        "AMD sobre Ubuntu 24.04 nativo",
        "ROCm 7.0.0-7.0.2",
        "--accelerator amd",
        "NO SOPORTADOS CON GPU",
        "Usuarios AMD en WSL2",
        "Usuarios Intel",
        "--accelerator cpu",
    ):
      self.assertIn(expected, summary.stdout)
    self.assertRegex(
        summary.stdout,
        re.compile(
            r"Usuarios AMD en WSL2:.*--accelerator cpu;.*"
            r"GPU AMD.*no esta admitida",
            re.DOTALL,
        ),
    )
    self.assertRegex(
        summary.stdout,
        re.compile(
            r"Usuarios Intel:.*--accelerator cpu;.*"
            r"GPU Intel.*no esta admitida",
            re.DOTALL,
        ),
    )

    help_result = subprocess.run(
        ["bash", str(BOOTSTRAP), "--help"],
        env=clean_test_environment(),
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    self.assertEqual(help_result.returncode, 0, help_result.stderr)
    self.assertIn(summary.stdout, help_result.stdout)

  def test_sourced_main_does_not_leak_shell_state(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      home = Path(temporary) / "home"
      home.mkdir()
      completed = run_bash(
          r'''
            set +e
            set +u
            set +o pipefail
            umask 077
            trap ':' EXIT
            flags_before="$-"
            umask_before="$(umask)"
            path_before="${PATH}"
            trap_before="$(trap -p EXIT)"
            fake_install_path="$2/project"
            source "$1"
            sim2real_bootstrap_validate_host() { :; }
            sim2real_bootstrap_host_kind() { printf '%s\n' native; }
            sim2real_bootstrap_resolve_install_path() {
              printf '%s\n' "${fake_install_path}"
            }
            sim2real_bootstrap_install_base_packages() { :; }
            sim2real_bootstrap_resolve_accelerator() { printf '%s\n' cpu; }
            sim2real_bootstrap_detect_virtualization() { printf '%s\n' none; }
            sim2real_bootstrap_sync_repository() { :; }
            sim2real_bootstrap_run_acceptance() { :; }
            sim2real_bootstrap_main --accelerator cpu >/dev/null
            [[ "$-" == "${flags_before}" ]]
            [[ "$(umask)" == "${umask_before}" ]]
            [[ "${PATH}" == "${path_before}" ]]
            [[ "$(trap -p EXIT)" == "${trap_before}" ]]
          ''',
          home,
          env={"HOME": str(home)},
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)
      self.assertEqual(list(home.iterdir()), [])

  def test_argument_parser_rejects_missing_invalid_and_unknown_options(self) -> None:
    cases = (
        (),
        ("--accelerator",),
        ("--accelerator=",),
        ("--accelerator", "gpu"),
        ("--accelerator", "cpu", "--install-path"),
        ("--accelerator", "cpu", "--install-path="),
        ("--accelerator", "cpu", "--repo-url"),
        ("--accelerator", "cpu", "--repo-url", "--malicious-option"),
        ("--accelerator", "cpu", "--unknown"),
    )
    for arguments in cases:
      with self.subTest(arguments=arguments):
        completed = subprocess.run(
            ["bash", str(BOOTSTRAP), *arguments],
            env=clean_test_environment(),
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 2, completed.stderr)

  def test_argument_parser_accepts_every_public_accelerator_profile(self) -> None:
    """Aceptar el nombre no implica prometer que el host pueda ejecutarlo."""

    body = r'''
      source "$1"
      shift
      sim2real_bootstrap_validate_host() { :; }
      sim2real_bootstrap_host_kind() { printf '%s\n' native; }
      sim2real_bootstrap_resolve_install_path() { printf '%s\n' /home/test/project; }
      sim2real_bootstrap_install_base_packages() { :; }
      sim2real_bootstrap_resolve_accelerator() {
        printf 'REQUESTED=<%s>\n' "$1" >&2
        # Se devuelve CPU para aislar esta prueba al parser. Las politicas de
        # elegibilidad de cada perfil se prueban por separado.
        printf '%s\n' cpu
      }
      sim2real_bootstrap_detect_virtualization() { printf '%s\n' none; }
      sim2real_bootstrap_sync_repository() { :; }
      sim2real_bootstrap_run_acceptance() { :; }
      sim2real_bootstrap_main "$@"
    '''
    for accelerator in ("compatible", "auto", "nvidia", "amd", "intel", "cpu"):
      with self.subTest(accelerator=accelerator):
        completed = run_bash(body, "--accelerator", accelerator)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(f"REQUESTED=<{accelerator}>", completed.stderr)

  def test_argument_values_are_forwarded_without_shell_reinterpretation(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      sentinel = Path(temporary) / "argument-was-executed"
      requested_path = f"{temporary}/ruta con espacios;$(touch {sentinel})"
      fork_url = "https://github.com/example-owner/Sim2Real-MJX-JAX-sobre-Linux-WSL"
      completed = run_bash(
          r'''
            source "$1"
            shift
            sim2real_bootstrap_validate_host() { :; }
            sim2real_bootstrap_resolve_install_path() { printf '%s\n' "$1"; }
            sim2real_bootstrap_install_base_packages() { :; }
            sim2real_bootstrap_resolve_accelerator() { printf '%s\n' "$1"; }
            sim2real_bootstrap_sync_repository() {
              printf 'SYNC_PATH=<%s>\nSYNC_URL=<%s>\n' "$1" "$2"
            }
            sim2real_bootstrap_run_acceptance() {
              printf 'ACCEPT_PATH=<%s>\nACCEPT_ACCELERATOR=<%s>\n' "$1" "$2"
            }
            sim2real_bootstrap_main "$@"
          ''',
          "--accelerator=cpu",
          f"--install-path={requested_path}",
          "--repo-url",
          fork_url,
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)
      self.assertIn(f"SYNC_PATH=<{requested_path}>", completed.stdout)
      self.assertIn(f"SYNC_URL=<{fork_url}>", completed.stdout)
      self.assertIn("ACCEPT_ACCELERATOR=<cpu>", completed.stdout)
      self.assertFalse(sentinel.exists(), "Se reinterpretaron metacaracteres del argumento")

  def test_repo_url_validation_accepts_only_the_expected_https_shape(self) -> None:
    accepted = (
        OFFICIAL_REPO,
        OFFICIAL_REPO.removesuffix(".git"),
        "https://github.com/a/Sim2Real-MJX-JAX-sobre-Linux-WSL.git",
        "https://github.com/owner-2.example_name/Sim2Real-MJX-JAX-sobre-Linux-WSL",
    )
    rejected = (
        "",
        "http://github.com/owner/Sim2Real-MJX-JAX-sobre-Linux-WSL.git",
        "https://github.com.evil.invalid/owner/Sim2Real-MJX-JAX-sobre-Linux-WSL.git",
        "https://user@github.com/owner/Sim2Real-MJX-JAX-sobre-Linux-WSL.git",
        "https://github.com/-owner/Sim2Real-MJX-JAX-sobre-Linux-WSL.git",
        "https://github.com/owner/otro-repositorio.git",
        "https://github.com/owner/Sim2Real-MJX-JAX-sobre-Linux-WSL.git/",
        "https://github.com/owner/Sim2Real-MJX-JAX-sobre-Linux-WSL.git?ref=main",
        "https://github.com/owner/Sim2Real-MJX-JAX-sobre-Linux-WSL#main",
        "https://github.com/owner/Sim2Real-MJX-JAX-sobre-Linux-WSL.git\n--upload-pack=malicioso",
        "git@github.com:owner/Sim2Real-MJX-JAX-sobre-Linux-WSL.git",
        "ssh://github.com/owner/Sim2Real-MJX-JAX-sobre-Linux-WSL.git",
        "file:///tmp/Sim2Real-MJX-JAX-sobre-Linux-WSL.git",
        "ext::sh -c touch /tmp/no",
        "--upload-pack=malicioso",
    )
    for repo_url in accepted:
      with self.subTest(repo_url=repo_url, expected="accepted"):
        completed = run_bash(
            'source "$1"; sim2real_bootstrap_validate_repo_url "$2"', repo_url
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
    for repo_url in rejected:
      with self.subTest(repo_url=repo_url, expected="rejected"):
        completed = run_bash(
            'source "$1"; sim2real_bootstrap_validate_repo_url "$2"', repo_url
        )
        self.assertNotEqual(completed.returncode, 0)

    normalized_with_suffix = run_bash(
        'source "$1"; sim2real_bootstrap_normalize_repo_url "$2"', OFFICIAL_REPO
    )
    normalized_without_suffix = run_bash(
        'source "$1"; sim2real_bootstrap_normalize_repo_url "$2"',
        OFFICIAL_REPO.removesuffix(".git"),
    )
    self.assertEqual(normalized_with_suffix.returncode, 0)
    self.assertEqual(normalized_with_suffix.stdout, normalized_without_suffix.stdout)

  def test_install_path_stays_beneath_home_and_rejects_symlinks(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      base = Path(temporary)
      home = base / "home"
      outside = base / "outside"
      home.mkdir()
      outside.mkdir()
      env = {"HOME": str(home)}

      accepted = {
          "": home / "robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL",
          "~/robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL": home / "robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL",
          "~/ruta con espacios/proyecto": home / "ruta con espacios/proyecto",
          str(home / "absoluta/proyecto"): home / "absoluta/proyecto",
      }
      for requested, expected in accepted.items():
        with self.subTest(requested=requested, expected="accepted"):
          completed = run_bash(
              'source "$1"; sim2real_bootstrap_resolve_install_path "$2"',
              requested,
              env=env,
          )
          self.assertEqual(completed.returncode, 0, completed.stderr)
          self.assertEqual(Path(completed.stdout.strip()), expected)

      rejected = {
          "~": 10,
          "ruta/relativa": 2,
          str(outside / "proyecto"): 10,
          "~/../outside/proyecto": 10,
      }
      for requested, expected_code in rejected.items():
        with self.subTest(requested=requested, expected="rejected"):
          completed = run_bash(
              'source "$1"; sim2real_bootstrap_resolve_install_path "$2"',
              requested,
              env=env,
          )
          self.assertEqual(completed.returncode, expected_code, completed.stderr)

      symlink = home / "enlace"
      symlink.symlink_to(outside, target_is_directory=True)
      completed = run_bash(
          'source "$1"; sim2real_bootstrap_resolve_install_path "$2"',
          "~/enlace/proyecto",
          env=env,
      )
      self.assertEqual(completed.returncode, 10, completed.stderr)
      self.assertIn("enlace simbolico", completed.stderr)

  def test_wsl_install_path_rejects_windows_filesystems_before_clone(self) -> None:
    body = r'''
      source "$1"
      sim2real_bootstrap_wsl_filesystem_type() {
        printf '%s\n' "${FAKE_FILESYSTEM}"
      }
      sim2real_bootstrap_validate_wsl_install_path "$2"
    '''
    for filesystem_type in ("drvfs", "9p", "v9fs", "virtiofs"):
      with self.subTest(filesystem_type=filesystem_type):
        completed = run_bash(
            body,
            "/home/user/robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL",
            env={"FAKE_FILESYSTEM": filesystem_type},
        )
        self.assertEqual(completed.returncode, 10, completed.stderr)
        self.assertIn("filesystem Linux", completed.stderr)

    accepted = run_bash(
        body,
        "/home/user/robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL",
        env={"FAKE_FILESYSTEM": "ext4"},
    )
    self.assertEqual(accepted.returncode, 0, accepted.stderr)

    unknown_mnt = run_bash(body, "/mnt/c/Sim2Real-MJX-JAX-sobre-Linux-WSL", env={"FAKE_FILESYSTEM": ""})
    self.assertEqual(unknown_mnt.returncode, 10, unknown_mnt.stderr)
    self.assertIn("/mnt", unknown_mnt.stderr)

  def test_host_validation_is_hermetic_for_supported_and_rejected_hosts(self) -> None:
    if os.geteuid() == 0:
      self.skipTest("La comprobacion dinamica usa el EUID real; el guard root se prueba estaticamente")

    body = r'''
      source "$1"
      sim2real_bootstrap_is_wsl() { [[ "${FAKE_WSL}" == "1" ]]; }
      sim2real_bootstrap_detect_wsl_version() { printf '%s\n' "${FAKE_WSL_VERSION}"; }
      sim2real_bootstrap_os_id() { printf '%s\n' "${FAKE_OS_ID}"; }
      sim2real_bootstrap_os_version() { printf '%s\n' "${FAKE_OS_VERSION}"; }
      uname() {
        case "${1:-}" in
          -s) printf '%s\n' "${FAKE_UNAME_S}" ;;
          -m) printf '%s\n' "${FAKE_ARCH}" ;;
          *) return 1 ;;
        esac
      }
      command() {
        if [[ "${1:-}" == "-v" && ( "${2:-}" == "sudo" || "${2:-}" == "apt-get" ) ]]; then
          [[ "${FAKE_MISSING_TOOL:-}" != "${2:-}" ]]
          return
        fi
        builtin command "$@"
      }
      sim2real_bootstrap_validate_host
    '''
    defaults = {
        "FAKE_WSL": "0",
        "FAKE_WSL_VERSION": "none",
        "FAKE_OS_ID": "ubuntu",
        "FAKE_OS_VERSION": "24.04",
        "FAKE_UNAME_S": "Linux",
        "FAKE_ARCH": "x86_64",
        "FAKE_MISSING_TOOL": "",
    }
    for version in ("22.04", "24.04"):
      with self.subTest(version=version, expected="accepted"):
        env = defaults | {"FAKE_OS_VERSION": version}
        completed = run_bash(body, env=env)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    for version in ("22.04", "24.04"):
      with self.subTest(version=version, host="wsl2", expected="accepted"):
        env = defaults | {
            "FAKE_WSL": "1",
            "FAKE_WSL_VERSION": "2",
            "FAKE_OS_VERSION": version,
        }
        completed = run_bash(body, env=env)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    rejected_overrides = (
        {"FAKE_WSL": "1", "FAKE_WSL_VERSION": "1"},
        {"FAKE_OS_ID": "debian"},
        {"FAKE_OS_VERSION": "20.04"},
        {"FAKE_OS_VERSION": "24.10"},
        {"FAKE_UNAME_S": "Darwin"},
        {"FAKE_ARCH": "aarch64"},
        {"FAKE_MISSING_TOOL": "sudo"},
        {"FAKE_MISSING_TOOL": "apt-get"},
    )
    for overrides in rejected_overrides:
      with self.subTest(overrides=overrides, expected="rejected"):
        completed = run_bash(body, env=defaults | overrides)
        self.assertEqual(completed.returncode, 10, completed.stderr)

  def test_clean_checkout_accepts_equivalent_https_origin(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      repository = self.make_repository(
          Path(temporary), origin=OFFICIAL_REPO.removesuffix(".git")
      )
      completed = run_bash(
          'source "$1"; sim2real_bootstrap_validate_checkout "$2" "$3"',
          repository,
          OFFICIAL_REPO,
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)

  def test_foreign_repository_is_rejected_before_fetch_or_mutation(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      base = Path(temporary)
      repository = self.make_repository(
          base,
          valid_identity=False,
          origin="https://example.invalid/original.git",
      )
      sentinel = base / "forbidden-git-command"
      head_before = self.git(repository, "rev-parse", "HEAD")
      origin_before = self.git(repository, "remote", "get-url", "origin")
      status_before = self.git(repository, "status", "--porcelain=v1", "--untracked-files=all")

      completed = self.run_sync_that_must_reject(repository, OFFICIAL_REPO, sentinel)

      self.assertEqual(completed.returncode, 30, completed.stderr)
      self.assertFalse(sentinel.exists())
      self.assertEqual(self.git(repository, "rev-parse", "HEAD"), head_before)
      self.assertEqual(self.git(repository, "remote", "get-url", "origin"), origin_before)
      self.assertEqual(
          self.git(repository, "status", "--porcelain=v1", "--untracked-files=all"),
          status_before,
      )

  def test_dirty_repository_is_rejected_before_fetch_for_all_dirty_states(self) -> None:
    for dirty_state in ("modified", "staged", "untracked"):
      with self.subTest(dirty_state=dirty_state), tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        repository = self.make_repository(base)
        if dirty_state == "modified":
          (repository / "payload.txt").write_text("modificado\n", encoding="utf-8")
        elif dirty_state == "staged":
          (repository / "staged.txt").write_text("preparado\n", encoding="utf-8")
          self.git(repository, "add", "staged.txt")
        else:
          (repository / "untracked.txt").write_text("sin seguimiento\n", encoding="utf-8")

        sentinel = base / "forbidden-git-command"
        head_before = self.git(repository, "rev-parse", "HEAD")
        status_before = self.git(
            repository, "status", "--porcelain=v1", "--untracked-files=all"
        )
        completed = self.run_sync_that_must_reject(repository, OFFICIAL_REPO, sentinel)

        self.assertEqual(completed.returncode, 30, completed.stderr)
        self.assertFalse(sentinel.exists())
        self.assertEqual(self.git(repository, "rev-parse", "HEAD"), head_before)
        self.assertEqual(
            self.git(repository, "status", "--porcelain=v1", "--untracked-files=all"),
            status_before,
        )

  def test_origin_and_branch_mismatch_are_rejected_before_fetch(self) -> None:
    scenarios = ("origin", "branch", "detached")
    for scenario in scenarios:
      with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        repository = self.make_repository(base)
        expected_url = OFFICIAL_REPO
        if scenario == "origin":
          expected_url = "https://github.com/another-owner/Sim2Real-MJX-JAX-sobre-Linux-WSL.git"
        elif scenario == "branch":
          self.git(repository, "switch", "--quiet", "-c", "feature")
        else:
          self.git(repository, "checkout", "--quiet", "--detach")

        sentinel = base / "forbidden-git-command"
        head_before = self.git(repository, "rev-parse", "HEAD")
        origin_before = self.git(repository, "remote", "get-url", "origin")
        branch_before = self.git(repository, "symbolic-ref", "--quiet", "--short", "HEAD") if scenario != "detached" else ""

        completed = self.run_sync_that_must_reject(repository, expected_url, sentinel)

        self.assertEqual(completed.returncode, 30, completed.stderr)
        self.assertFalse(sentinel.exists())
        self.assertEqual(self.git(repository, "rev-parse", "HEAD"), head_before)
        self.assertEqual(self.git(repository, "remote", "get-url", "origin"), origin_before)
        if scenario != "detached":
          self.assertEqual(
              self.git(repository, "symbolic-ref", "--quiet", "--short", "HEAD"),
              branch_before,
          )
        else:
          symbolic_ref = subprocess.run(
              ["git", "-C", str(repository), "symbolic-ref", "--quiet", "HEAD"],
              text=True,
              capture_output=True,
              check=False,
          )
          self.assertNotEqual(symbolic_ref.returncode, 0)

  def test_nvidia_driver_version_boundary(self) -> None:
    accepted = ("525", "525.0", "525.0.0", "580.82.07", "999")
    rejected = ("", "0", "52", "524", "524.99", "525beta", "525.0-rc1", "desconocida")
    for version in accepted:
      with self.subTest(version=version, expected="accepted"):
        completed = run_bash(
            'source "$1"; sim2real_bootstrap_nvidia_driver_version_supported "$2"',
            version,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
    for version in rejected:
      with self.subTest(version=version, expected="rejected"):
        completed = run_bash(
            'source "$1"; sim2real_bootstrap_nvidia_driver_version_supported "$2"',
            version,
        )
        self.assertNotEqual(completed.returncode, 0)

  def test_nvidia_driver_version_falls_back_to_standard_smi_header(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      fake_bin = Path(temporary) / "bin"
      fake_bin.mkdir()
      nvidia_smi = fake_bin / "nvidia-smi"
      nvidia_smi.write_text(
          """#!/usr/bin/env bash
case "${1:-}" in
  -L) echo 'GPU 0: NVIDIA Example (UUID: GPU-test)' ;;
  --query-gpu=driver_version) exit 1 ;;
  *) echo '| NVIDIA-SMI 535.183.01   Driver Version: 535.183.01   CUDA Version: 12.2 |' ;;
esac
""",
          encoding="utf-8",
      )
      nvidia_smi.chmod(0o755)
      completed = run_bash(
          r'''
            set -o pipefail
            source "$1"
            sim2real_bootstrap_is_wsl() { return 1; }
            sim2real_bootstrap_nvidia_driver_version
          ''',
          env={"PATH": f"{fake_bin}:{os.environ['PATH']}"},
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)
      self.assertEqual(completed.stdout.strip(), "535.183.01")

  def test_wsl_packages_exclude_linux_driver_stack(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      base = Path(temporary)
      wsl_log = base / "wsl-sudo.log"
      native_log = base / "native-sudo.log"
      body = r'''
        log_file="$2"
        fake_host="$3"
        source "$1"
        sudo() { printf '%s\n' "$*" >>"${log_file}"; }
        sim2real_bootstrap_host_kind() { printf '%s\n' "${fake_host}"; }
        sim2real_bootstrap_install_base_packages "${fake_host}"
        if [[ "${fake_host}" == "native" ]]; then
          sim2real_bootstrap_install_nvidia_management_packages
        fi
      '''
      wsl = run_bash(body, wsl_log, "wsl")
      self.assertEqual(wsl.returncode, 0, wsl.stderr)
      wsl_commands = wsl_log.read_text(encoding="utf-8")
      self.assertIn("apt-get update", wsl_commands)
      self.assertIn("ca-certificates", wsl_commands)
      for forbidden in ("ubuntu-drivers", "mokutil", "nvidia-driver"):
        self.assertNotIn(forbidden, wsl_commands)

      native = run_bash(body, native_log, "native")
      self.assertEqual(native.returncode, 0, native.stderr)
      native_commands = native_log.read_text(encoding="utf-8")
      self.assertIn("ubuntu-drivers-common", native_commands)
      self.assertIn("mokutil", native_commands)

  def test_wsl_nvidia_failure_never_calls_native_driver_management(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      forbidden = Path(temporary) / "native-driver-command"
      completed = run_bash(
          r'''
            forbidden="$2"
            source "$1"
            sim2real_bootstrap_validate_host() { :; }
            sim2real_bootstrap_host_kind() { printf '%s\n' wsl; }
            sim2real_bootstrap_add_wsl_driver_path() { :; }
            sim2real_bootstrap_resolve_install_path() { printf '%s\n' /home/test/project; }
            sim2real_bootstrap_validate_wsl_install_path() { :; }
            sim2real_bootstrap_install_base_packages() { :; }
            sim2real_bootstrap_resolve_accelerator() { printf '%s\n' nvidia; }
            sim2real_bootstrap_detect_virtualization() { printf '%s\n' none; }
            sim2real_bootstrap_validate_nvidia_wsl() { return 40; }
            sim2real_bootstrap_install_nvidia_management_packages() {
              : >"${forbidden}"
            }
            sim2real_bootstrap_prepare_nvidia() { : >"${forbidden}"; }
            sim2real_bootstrap_write_driver_state() { : >"${forbidden}"; }
            sim2real_bootstrap_request_reboot() { : >"${forbidden}"; }
            ubuntu-drivers() { : >"${forbidden}"; }
            sudo() { : >"${forbidden}"; }
            sim2real_bootstrap_sync_repository() { : >"${forbidden}"; }
            sim2real_bootstrap_run_acceptance() { : >"${forbidden}"; }
            sim2real_bootstrap_main --accelerator nvidia
          ''',
          forbidden,
      )
      self.assertEqual(completed.returncode, 40, completed.stderr)
      self.assertFalse(forbidden.exists(), "WSL alcanzo una operacion nativa o posterior")

      for function_name in (
          "sim2real_bootstrap_install_nvidia_management_packages",
          "sim2real_bootstrap_prepare_nvidia",
      ):
        with self.subTest(direct_guard=function_name):
          guarded = run_bash(
              r'''
                forbidden="$2"
                source "$1"
                sim2real_bootstrap_host_kind() { printf '%s\n' wsl; }
                sudo() { : >"${forbidden}"; }
                ubuntu-drivers() { : >"${forbidden}"; }
                sim2real_bootstrap_write_driver_state() { : >"${forbidden}"; }
                sim2real_bootstrap_request_reboot() { : >"${forbidden}"; }
                "$3"
              ''',
              forbidden,
              function_name,
          )
          self.assertEqual(guarded.returncode, 40, guarded.stderr)
          self.assertFalse(forbidden.exists(), "El guard WSL permitio una operacion nativa")

  def test_wsl_nvidia_driver_good_old_and_missing_are_explicit(self) -> None:
    body = r'''
      source "$1"
      sim2real_bootstrap_nvidia_smi_command() {
        [[ "${FAKE_DRIVER}" != "missing" ]] || return 1
        printf '%s\n' /bin/true
      }
      sim2real_bootstrap_nvidia_driver_version() {
        case "${FAKE_DRIVER}" in
          good) printf '%s\n' 535.183.01 ;;
          old) printf '%s\n' 524.99 ;;
          unknown) return 1 ;;
          missing) return 1 ;;
        esac
      }
      sim2real_bootstrap_validate_nvidia_wsl
    '''
    expected_codes = {"good": 0, "old": 40, "unknown": 40, "missing": 40}
    for scenario, expected_code in expected_codes.items():
      with self.subTest(scenario=scenario):
        completed = run_bash(body, env={"FAKE_DRIVER": scenario})
        self.assertEqual(completed.returncode, expected_code, completed.stderr)
        combined = f"{completed.stdout}\n{completed.stderr}"
        self.assertIn("Windows", combined)
        if scenario != "good":
          self.assertNotIn("usara CPU", combined)

  def test_recommended_nvidia_package_is_parsed_from_typical_ubuntu_output(self) -> None:
    completed = run_bash(
        r'''
          source "$1"
          ubuntu-drivers() {
            cat <<'EOF'
== /sys/devices/pci0000:00/0000:01:00.0 ==
vendor   : NVIDIA Corporation
model    : Example GPU
driver   : nvidia-driver-535 - distro non-free
driver   : nvidia-driver-550 - distro non-free recommended
driver   : xserver-xorg-video-nouveau - distro free builtin
EOF
          }
          sim2real_bootstrap_recommended_nvidia_package
        ''',
    )
    self.assertEqual(completed.returncode, 0, completed.stderr)
    self.assertEqual(completed.stdout.strip(), "nvidia-driver-550")

    ambiguous = run_bash(
        r'''
          source "$1"
          ubuntu-drivers() {
            printf '%s\n' \
              'driver : nvidia-driver-550 - distro non-free recommended' \
              'driver : nvidia-driver-580-open - distro non-free recommended'
          }
          sim2real_bootstrap_recommended_nvidia_package
        ''',
    )
    self.assertNotEqual(ambiguous.returncode, 0)

  def test_driver_boot_state_prevents_reinstallation_before_and_after_reboot(self) -> None:
    saved_boot_id = "11111111-1111-1111-1111-111111111111"
    scenarios = (
        (saved_boot_id, 20),
        ("22222222-2222-2222-2222-222222222222", 40),
    )
    body = r'''
      source "$1"
      FAKE_STATE_FILE="$2"
      FAKE_FORBIDDEN_CALL="$3"
      sim2real_bootstrap_host_kind() { printf '%s\n' native; }
      sim2real_bootstrap_driver_state_file() { printf '%s\n' "${FAKE_STATE_FILE}"; }
      sim2real_bootstrap_nvidia_driver_ready() { return 1; }
      sim2real_bootstrap_current_boot_id() { printf '%s\n' "${FAKE_BOOT_ID}"; }
      sim2real_bootstrap_request_reboot() { exit 20; }
      sim2real_bootstrap_recommended_nvidia_package() {
        : >"${FAKE_FORBIDDEN_CALL}"
        printf '%s\n' nvidia-driver-550
      }
      sudo() {
        : >"${FAKE_FORBIDDEN_CALL}"
        return 97
      }
      sim2real_bootstrap_prepare_nvidia
    '''
    for current_boot_id, expected_code in scenarios:
      with self.subTest(current_boot_id=current_boot_id), tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        state_file = base / "driver-install-boot-id"
        sentinel = base / "driver-was-reinstalled"
        state_file.write_text(f"{saved_boot_id}\n", encoding="utf-8")

        completed = run_bash(
            body,
            state_file,
            sentinel,
            env={"FAKE_BOOT_ID": current_boot_id},
        )

        self.assertEqual(completed.returncode, expected_code, completed.stderr)
        self.assertFalse(sentinel.exists(), "Se intento reinstalar el driver NVIDIA")
        self.assertEqual(state_file.read_text(encoding="utf-8"), f"{saved_boot_id}\n")

  def test_accelerator_resolution_uses_fake_hardware_only(self) -> None:
    body = r'''
      source "$1"
      sim2real_bootstrap_has_pci_display_vendor() {
        [[ " ${FAKE_VENDORS:-} " == *" $1 "* ]]
      }
      sim2real_bootstrap_os_version() {
        printf '%s\n' "${FAKE_OS_VERSION:-24.04}"
      }
      sim2real_bootstrap_amd_runtime_ready() {
        [[ "${FAKE_AMD_READY:-0}" == 1 && "${FAKE_OS_VERSION:-24.04}" == "24.04" ]]
      }
      sim2real_bootstrap_nvidia_driver_ready() {
        [[ "${FAKE_NVIDIA_READY:-0}" == 1 ]]
      }
      sim2real_bootstrap_resolve_accelerator "$2" "$3"
    '''
    cases = (
        ("cpu", "", "native", "24.04", "0", "0", "cpu", True),
        ("compatible", "0x10de", "native", "24.04", "0", "0", "nvidia", True),
        ("compatible", "0x1002", "native", "24.04", "1", "0", "amd", True),
        ("compatible", "", "wsl", "24.04", "0", "1", "nvidia", True),
        ("auto", "0x10de", "native", "24.04", "0", "0", "nvidia", True),
        ("auto", "0x1002", "native", "24.04", "1", "0", "amd", True),
        ("auto", "0x10de 0x1002", "native", "24.04", "1", "1", "", False),
        ("auto", "0x10de 0x1002", "native", "24.04", "1", "0", "amd", True),
        ("auto", "0x10de 0x1002", "native", "24.04", "0", "1", "nvidia", True),
        ("nvidia", "0x10de", "native", "24.04", "0", "0", "nvidia", True),
        ("nvidia", "", "native", "24.04", "0", "0", "", False),
        ("amd", "0x1002", "native", "24.04", "1", "0", "amd", True),
        ("auto", "0x1002", "native", "24.04", "0", "0", "", False),
        ("amd", "0x1002", "native", "24.04", "0", "0", "", False),
        ("auto", "0x1002", "native", "22.04", "1", "0", "", False),
        ("amd", "0x1002", "native", "22.04", "1", "0", "", False),
        ("auto", "", "native", "24.04", "0", "0", "", False),
        ("auto", "0x8086", "native", "24.04", "0", "0", "", False),
        ("intel", "0x8086", "native", "24.04", "0", "0", "", False),
        ("amd", "0x1002", "wsl", "24.04", "1", "0", "", False),
        ("intel", "0x8086", "wsl", "24.04", "0", "0", "", False),
    )
    for requested, vendors, host, os_version, amd_ready, nvidia_ready, expected, succeeds in cases:
      with self.subTest(
          requested=requested,
          vendors=vendors,
          host=host,
          os_version=os_version,
          amd_ready=amd_ready,
          nvidia_ready=nvidia_ready,
      ):
        completed = run_bash(
            body,
            requested,
            host,
            env={
                "FAKE_VENDORS": vendors,
                "FAKE_OS_VERSION": os_version,
                "FAKE_AMD_READY": amd_ready,
                "FAKE_NVIDIA_READY": nvidia_ready,
            },
        )
        if succeeds:
          self.assertEqual(completed.returncode, 0, completed.stderr)
          self.assertEqual(completed.stdout.strip(), expected)
        else:
          self.assertNotEqual(completed.returncode, 0, completed.stderr)
          self.assertNotEqual(completed.stdout.strip(), "cpu")
          if vendors == "0x1002" and host == "native":
            if os_version == "22.04":
              self.assertIn("solo se admite en Ubuntu 24.04", completed.stderr)
            elif amd_ready == "0":
              self.assertIn("ROCm no esta listo", completed.stderr)

  def test_legacy_venv_process_is_only_accepted_for_safe_shutdown(self) -> None:
    platform_library = ROOT / "scripts/lib/platform.sh"
    with tempfile.TemporaryDirectory() as temporary:
      repository = Path(temporary) / "repository"
      logs_root = repository / "logs_sim2real_mjx"
      run_dir = logs_root / "legacy-run"
      trainer = repository / "sim2real_mjx/entrenar_ppo_mjx.py"
      legacy_python = repository / ".venv/bin/python"
      run_dir.mkdir(parents=True)
      trainer.parent.mkdir(parents=True)
      legacy_python.parent.mkdir(parents=True)
      trainer.write_text(
          "import time\ntime.sleep(60)\n",
          encoding="utf-8",
      )
      legacy_python.symlink_to(sys.executable)

      process = subprocess.Popen(
          [
              str(legacy_python),
              str(trainer),
              "--logdir",
              str(logs_root),
              "--run_name",
              run_dir.name,
          ],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
      )
      try:
        (run_dir / "entrenamiento.pid").write_text(
            f"{process.pid}\n", encoding="utf-8"
        )
        completed = run_bash(
            r'''
              source "$2"
              repository="$3"
              run_dir="$4"
              pid="$5"
              sim2real_training_pid_matches_run \
                "${repository}" "${run_dir}" "${pid}"
              active="$(sim2real_active_training "${repository}")"
              printf 'ACTIVE=%s\n' "${active}"
              printf 'RUNTIME=%s\n' \
                "$(sim2real_python_executable cpu "${repository}")"
              sim2real_signal_training_process \
                "${repository}" "${run_dir}" "${pid}" TERM
            ''',
            platform_library,
            repository,
            run_dir,
            str(process.pid),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(f"ACTIVE={run_dir}\t{process.pid}", completed.stdout)
        self.assertIn(
            f"RUNTIME={repository}/.venvs/cpu/bin/python", completed.stdout
        )
        self.assertNotIn(f"RUNTIME={legacy_python}", completed.stdout)
        self.assertEqual(process.wait(timeout=5), -15)
        self.assertFalse((repository / ".sim2real/accelerator").exists())
      finally:
        if process.poll() is None:
          process.terminate()
          process.wait(timeout=5)

  def test_wsl_compatible_can_fall_back_while_auto_remains_fail_closed(self) -> None:
    body = r'''
      source "$1"
      sim2real_bootstrap_nvidia_driver_ready() {
        [[ "${FAKE_DRIVER_READY}" == "1" ]]
      }
      sim2real_bootstrap_resolve_accelerator "$2" wsl
    '''
    compatible_ready = run_bash(body, "compatible", env={"FAKE_DRIVER_READY": "1"})
    self.assertEqual(compatible_ready.returncode, 0, compatible_ready.stderr)
    self.assertEqual(compatible_ready.stdout.strip(), "nvidia")
    self.assertNotIn("AVISO", compatible_ready.stderr)

    compatible_missing = run_bash(
        body, "compatible", env={"FAKE_DRIVER_READY": "0"}
    )
    self.assertEqual(compatible_missing.returncode, 0, compatible_missing.stderr)
    self.assertEqual(compatible_missing.stdout.strip(), "cpu")
    self.assertIn("AVISO", compatible_missing.stderr)
    self.assertIn("WSL2 no expone una NVIDIA operativa", compatible_missing.stderr)
    self.assertIn("seleccionara CPU automaticamente", compatible_missing.stderr)

    auto_ready = run_bash(body, "auto", env={"FAKE_DRIVER_READY": "1"})
    self.assertEqual(auto_ready.returncode, 0, auto_ready.stderr)
    self.assertEqual(auto_ready.stdout.strip(), "nvidia")

    auto_missing = run_bash(body, "auto", env={"FAKE_DRIVER_READY": "0"})
    self.assertNotEqual(auto_missing.returncode, 0, auto_missing.stderr)
    self.assertNotEqual(auto_missing.stdout.strip(), "cpu")
    self.assertIn("WSL2", auto_missing.stderr)
    self.assertRegex(auto_missing.stderr, r"NVIDIA.*driver Windows|driver.*NVIDIA.*Windows")

    explicit = run_bash(body, "nvidia", env={"FAKE_DRIVER_READY": "0"})
    self.assertEqual(explicit.returncode, 0, explicit.stderr)
    self.assertEqual(explicit.stdout.strip(), "nvidia")

  def test_incompatible_profiles_are_blocked_before_apt_clone_or_acceptance(self) -> None:
    body = r'''
      sentinels="$2"
      requested="$3"
      fake_host="$4"
      source "$1"
      sim2real_bootstrap_validate_host() { :; }
      sim2real_bootstrap_host_kind() { printf '%s\n' "${fake_host}"; }
      sim2real_bootstrap_resolve_install_path() { printf '%s\n' /home/test/project; }
      sim2real_bootstrap_validate_wsl_install_path() { :; }
      sim2real_bootstrap_detect_virtualization() { printf '%s\n' none; }
      sim2real_bootstrap_install_base_packages() { : >"${sentinels}/apt"; }
      sim2real_bootstrap_has_pci_display_vendor() { return 0; }
      sim2real_bootstrap_nvidia_driver_ready() { return 1; }
      sim2real_bootstrap_nvidia_smi_command() { return 1; }
      sim2real_bootstrap_sync_repository() { : >"${sentinels}/clone"; }
      sim2real_bootstrap_run_acceptance() { : >"${sentinels}/acceptance"; }
      sim2real_bootstrap_main --accelerator "${requested}"
    '''
    scenarios = (
        ("amd", "wsl", "AMD/JAX"),
        ("intel", "native", "Intel"),
        ("intel", "wsl", "Intel"),
        ("nvidia", "wsl", "nvidia-smi"),
    )
    for accelerator, host, expected_diagnostic in scenarios:
      with self.subTest(accelerator=accelerator, host=host), tempfile.TemporaryDirectory() as temporary:
        sentinels = Path(temporary)
        completed = run_bash(body, sentinels, accelerator, host)
        self.assertEqual(completed.returncode, 40, completed.stderr)
        for operation in ("apt", "clone", "acceptance"):
          self.assertFalse(
              (sentinels / operation).exists(),
              f"{operation} se ejecuto antes de rechazar {accelerator} en {host}",
          )
        self.assertNotIn("Acelerador resuelto: cpu", completed.stdout)
        self.assertIn(expected_diagnostic, completed.stderr)
        self.assertIn("VALIDADO DE PRINCIPIO A FIN EN EL PROYECTO", completed.stderr)
        self.assertIn("NO SOPORTADOS CON GPU", completed.stderr)
        self.assertIn("Usuarios AMD en WSL2", completed.stderr)
        self.assertIn("Usuarios Intel", completed.stderr)
        self.assertIn("--accelerator cpu", completed.stderr)

  def test_explicit_compatible_uses_cpu_with_warning_and_continues_on_wsl(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      sentinels = Path(temporary)
      completed = run_bash(
          r'''
            sentinels="$2"
            source "$1"
            sim2real_bootstrap_validate_host() { :; }
            sim2real_bootstrap_host_kind() { printf '%s\n' wsl; }
            sim2real_bootstrap_add_wsl_driver_path() { :; }
            sim2real_bootstrap_resolve_install_path() {
              printf '%s\n' /home/test/project
            }
            sim2real_bootstrap_validate_wsl_install_path() { :; }
            sim2real_bootstrap_nvidia_driver_ready() { return 1; }
            sim2real_bootstrap_install_base_packages() {
              : >"${sentinels}/apt"
            }
            sim2real_bootstrap_sync_repository() {
              : >"${sentinels}/clone"
            }
            sim2real_bootstrap_run_acceptance() {
              [[ "$2" == cpu ]]
              : >"${sentinels}/acceptance"
            }
            sim2real_bootstrap_main --accelerator compatible
          ''',
          sentinels,
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)
      self.assertIn("Acelerador solicitado: compatible", completed.stdout)
      self.assertIn("Acelerador resuelto: cpu", completed.stdout)
      self.assertIn("AVISO", completed.stderr)
      self.assertIn("WSL2 no expone una NVIDIA operativa", completed.stderr)
      for operation in ("apt", "clone", "acceptance"):
        self.assertTrue((sentinels / operation).is_file(), operation)

  def test_explicit_cpu_reaches_base_packages_clone_and_acceptance(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      sentinels = Path(temporary)
      completed = run_bash(
          r'''
            sentinels="$2"
            source "$1"
            sim2real_bootstrap_validate_host() { :; }
            sim2real_bootstrap_host_kind() { printf '%s\n' wsl; }
            sim2real_bootstrap_add_wsl_driver_path() { :; }
            sim2real_bootstrap_resolve_install_path() {
              printf '%s\n' /home/test/project
            }
            sim2real_bootstrap_validate_wsl_install_path() { :; }
            sim2real_bootstrap_install_base_packages() {
              : >"${sentinels}/apt"
            }
            sim2real_bootstrap_sync_repository() {
              : >"${sentinels}/clone"
            }
            sim2real_bootstrap_run_acceptance() {
              [[ "$2" == cpu ]]
              : >"${sentinels}/acceptance"
            }
            sim2real_bootstrap_main --accelerator cpu
          ''',
          sentinels,
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)
      self.assertIn("Acelerador resuelto: cpu", completed.stdout)
      for operation in ("apt", "clone", "acceptance"):
        self.assertTrue((sentinels / operation).is_file(), operation)

  def test_virtual_machine_without_gpu_has_explicit_passthrough_diagnosis(self) -> None:
    body = r'''
      source "$1"
      sim2real_bootstrap_has_pci_display_vendor() { return 1; }
      sim2real_bootstrap_nvidia_driver_ready() { return 1; }
      sim2real_bootstrap_report_pci_graphics() { :; }
      sim2real_bootstrap_detect_virtualization() { printf '%s\n' oracle; }
      sim2real_bootstrap_resolve_accelerator "$2" native
    '''
    explicit = run_bash(body, "nvidia")
    self.assertEqual(explicit.returncode, 40, explicit.stderr)
    self.assertIn("VirtualBox", explicit.stderr)
    self.assertIn("GPU NVIDIA PCI", explicit.stderr)
    self.assertNotIn("continuara", explicit.stderr)

    automatic = run_bash(body, "auto")
    self.assertNotEqual(automatic.returncode, 0, automatic.stderr)
    self.assertNotEqual(automatic.stdout.strip(), "cpu")
    self.assertIn("VirtualBox", automatic.stderr)
    self.assertIn("--accelerator cpu", automatic.stderr)

    platform_name = run_bash(
        r'''
          source "$1"
          sim2real_bootstrap_detect_virtualization() { printf '%s\n' oracle; }
          sim2real_bootstrap_platform_description native
        ''',
    )
    self.assertEqual(platform_name.returncode, 0, platform_name.stderr)
    self.assertEqual(platform_name.stdout.strip(), "Ubuntu en VM (VirtualBox/Oracle)")

  def test_pci_detection_consumes_long_output_and_accepts_display_subclasses(self) -> None:
    body = r'''
      set -o pipefail
      source "$1"
      fake_class="$2"
      fake_vendor="$3"
      tail_lines="$4"
      lspci() {
        printf '0000:01:00.0 %s: %s:2684 (rev a1)\n' \
          "${fake_class}" "${fake_vendor}"
        awk -v count="${tail_lines}" 'BEGIN {
          for (i = 0; i < count; i++)
            print "0000:00:00.0 0600: 8086:0000"
        }'
      }
      sim2real_bootstrap_has_pci_display_vendor "0x${fake_vendor,,}"
    '''
    cases = (
        ("0300", "ABCD", 1, True),
        ("0301", "ABCE", 1, True),
        ("0302", "ABCF", 200_000, True),
        ("0380", "ABD0", 1, True),
        ("03ff", "ABD1", 1, True),
        ("0403", "ABD2", 1, False),
    )
    for pci_class, vendor, tail_lines, expected in cases:
      with self.subTest(pci_class=pci_class, expected=expected):
        completed = run_bash(body, pci_class, vendor, str(tail_lines))
        if expected:
          self.assertEqual(completed.returncode, 0, completed.stderr)
        else:
          self.assertNotEqual(completed.returncode, 0)

  def test_root_sudo_and_repository_updates_avoid_destructive_patterns(self) -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    self.assertRegex(source, r"if \(\( EUID == 0 \)\); then")

    main = source[source.index("sim2real_bootstrap_main()") :]
    host_validation = main.index("sim2real_bootstrap_validate_host")
    accelerator_resolution = main.index(
        'sim2real_bootstrap_resolve_accelerator "${requested_accelerator}"'
    )
    wsl_nvidia_preflight = main.index("sim2real_bootstrap_validate_nvidia_wsl")
    base_packages = main.index("sim2real_bootstrap_install_base_packages")
    repository_sync = main.index("sim2real_bootstrap_sync_repository")
    self.assertLess(host_validation, accelerator_resolution)
    self.assertLess(accelerator_resolution, wsl_nvidia_preflight)
    self.assertLess(wsl_nvidia_preflight, base_packages)
    self.assertLess(base_packages, repository_sync)

    sudo_commands = [
        line.strip()
        for line in source.splitlines()
        if re.match(r"^\s*sudo(?:\s|$)", line)
    ]
    self.assertGreaterEqual(len(sudo_commands), 3)
    for command in sudo_commands:
      self.assertRegex(
          command,
          r"^sudo (?:DEBIAN_FRONTEND=noninteractive )?(?:apt-get|ubuntu-drivers)\b",
      )
    self.assertIn("sudo ubuntu-drivers install", sudo_commands)
    for forbidden in (
        "git reset",
        "git clean",
        "git checkout",
        "remote set-url",
        "rm -rf",
        "sudo ./scripts/install.sh",
        "sudo bash",
        "eval ",
    ):
      self.assertNotIn(forbidden, source)
    self.assertIn("status --porcelain=v1 --untracked-files=all", source)
    self.assertIn("core.hooksPath=/dev/null", source)
    self.assertRegex(source, re.compile(r"merge\s+\\\s*--ff-only", re.DOTALL))


if __name__ == "__main__":
  unittest.main()
