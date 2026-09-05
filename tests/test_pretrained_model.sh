#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL_REL="modelo_preentrenado/modelo_referencia_fase2_45932544"
MODEL_DIR="${REPO_ROOT}/${MODEL_REL}"
CHECKPOINT_REL="checkpoints/000045932544"
CHECKPOINT_DIR="${MODEL_DIR}/${CHECKPOINT_REL}"
PRIMARY_SHARD_REL="${CHECKPOINT_REL}/ocdbt.process_0/d/3c4621901660b3872a4ce38008f147ea"
PRIMARY_SHARD_SHA256="8148c5c2c8edad0eda0dd9167e219d6ad1aeef9872ae73ba4434144f2687145e"

fail() {
  echo "ERROR: contrato de la red preentrenada: $*" >&2
  exit 1
}

command -v python3 >/dev/null 2>&1 || fail "python3 no esta disponible."
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum no esta disponible."

[[ -d "${MODEL_DIR}" ]] || fail "falta ${MODEL_REL}."
[[ ! -L "${MODEL_DIR}" ]] || fail "el directorio del modelo no puede ser un enlace simbolico."
[[ -d "${CHECKPOINT_DIR}" ]] || fail "falta el checkpoint ${CHECKPOINT_REL}."
[[ ! -L "${CHECKPOINT_DIR}" ]] || fail "el checkpoint no puede ser un enlace simbolico."

SYMLINK_PATH="$(find "${MODEL_DIR}" -type l -print -quit)"
[[ -z "${SYMLINK_PATH}" ]] || fail "el bundle contiene un enlace simbolico: ${SYMLINK_PATH}."

EXPECTED_FILES=(
  "README.md"
  "SHA256SUMS"
  "checkpoints/000045932544/_CHECKPOINT_METADATA"
  "checkpoints/000045932544/_METADATA"
  "checkpoints/000045932544/_sharding"
  "checkpoints/000045932544/array_metadatas/process_0"
  "checkpoints/000045932544/d/bfb4b84f05877bd0546ce34867d6de42"
  "checkpoints/000045932544/manifest.ocdbt"
  "checkpoints/000045932544/ocdbt.process_0/d/3c4621901660b3872a4ce38008f147ea"
  "checkpoints/000045932544/ocdbt.process_0/d/5045d5d247cafa822934b44fc95b40ca"
  "checkpoints/000045932544/ocdbt.process_0/d/6cc32dbfa1810f9febe1d4dc751ae286"
  "checkpoints/000045932544/ocdbt.process_0/d/8657c0b4c9331108dd230e8c4d0a0727"
  "checkpoints/000045932544/ocdbt.process_0/manifest.ocdbt"
  "checkpoints/000045932544/ppo_network_config.json"
  "config_curriculum_recompensa.json"
  "config_entorno.json"
  "hiperparametros.json"
  "model.json"
)

EXPECTED_LAYOUT_FILE="$(mktemp)"
ACTUAL_LAYOUT_FILE="$(mktemp)"
EXPECTED_MANIFEST_FILE="$(mktemp)"
ACTUAL_MANIFEST_FILE="$(mktemp)"
CORRUPTED_MODEL_ROOT=""
cleanup() {
  rm -f -- \
    "${EXPECTED_LAYOUT_FILE}" \
    "${ACTUAL_LAYOUT_FILE}" \
    "${EXPECTED_MANIFEST_FILE}" \
    "${ACTUAL_MANIFEST_FILE}"
  if [[ -n "${CORRUPTED_MODEL_ROOT}" && -d "${CORRUPTED_MODEL_ROOT}" ]]; then
    rm -rf -- "${CORRUPTED_MODEL_ROOT}"
  fi
}
trap cleanup EXIT

printf '%s\n' "${EXPECTED_FILES[@]}" | LC_ALL=C sort >"${EXPECTED_LAYOUT_FILE}"
(
  cd "${MODEL_DIR}"
  find . -type f -printf '%P\n' | LC_ALL=C sort
) >"${ACTUAL_LAYOUT_FILE}"
if ! diff -u "${EXPECTED_LAYOUT_FILE}" "${ACTUAL_LAYOUT_FILE}"; then
  fail "el contenido del bundle no coincide con la version publicada."
fi

[[ -f "${MODEL_DIR}/SHA256SUMS" ]] || fail "falta SHA256SUMS."
if ! (
  cd "${MODEL_DIR}"
  sha256sum --quiet --strict --check SHA256SUMS
); then
  fail "SHA256SUMS no valida todos los archivos declarados."
fi

(
  cd "${MODEL_DIR}"
  find . -type f \
    ! -path './README.md' \
    ! -path './SHA256SUMS' \
    -printf '%P\n' | LC_ALL=C sort
) >"${EXPECTED_MANIFEST_FILE}"

if ! awk '
  NF != 2 || length($1) != 64 || $1 !~ /^[0-9a-f]+$/ { exit 1 }
  { print $2 }
' "${MODEL_DIR}/SHA256SUMS" | LC_ALL=C sort >"${ACTUAL_MANIFEST_FILE}"; then
  fail "SHA256SUMS tiene un formato no valido."
fi
if ! diff -u "${EXPECTED_MANIFEST_FILE}" "${ACTUAL_MANIFEST_FILE}"; then
  fail "SHA256SUMS no cubre exactamente el bundle (salvo README.md y SHA256SUMS)."
fi

ACTUAL_PRIMARY_SHARD_SHA256="$(sha256sum "${MODEL_DIR}/${PRIMARY_SHARD_REL}" | awk '{print $1}')"
[[ "${ACTUAL_PRIMARY_SHARD_SHA256}" == "${PRIMARY_SHARD_SHA256}" ]] || \
  fail "el shard principal no es el publicado para el paso 45932544."

python3 - "${MODEL_DIR}" "${REPO_ROOT}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
repository = Path(sys.argv[2])
expected_xml = "sim2real_mjx/xmls/ROBOT_POSE_IDEAL.xml"


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: contrato de la red preentrenada: {message}")


def load(relative_path: str):
    path = root / relative_path
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"no se puede leer {relative_path}: {exc}")


def expect(actual, expected, label: str) -> None:
    if actual != expected:
        fail(f"{label}: se esperaba {expected!r}, se encontro {actual!r}.")


model = load("model.json")
environment = load("config_entorno.json")
curriculum = load("config_curriculum_recompensa.json")
hyperparameters = load("hiperparametros.json")
network = load("checkpoints/000045932544/ppo_network_config.json")

expect(model.get("model_id"), "modelo_referencia_fase2_45932544", "model_id")
expect(model.get("reward_phase"), 2, "fase del modelo")
expect(model.get("seed"), 42, "seed")
expect(model.get("episode_length"), 1500, "duracion de episodio del modelo")
expect(model.get("checkpoint_step"), 45932544, "paso del checkpoint")
expect(model.get("target_training_steps"), 100000000, "objetivo de entrenamiento")
expect(model.get("eval_episode_reward"), 158.104767, "recompensa de evaluacion")
expect(model.get("checkpoint_relative_path"), "checkpoints/000045932544", "ruta del checkpoint")
expect(model.get("xml_relative_path"), expected_xml, "XML del modelo")

for metadata_key, relative_path in {
    "environment_source_sha256": "sim2real_mjx/entorno_robot_mjx.py",
    "hyperparameters_source_sha256": "sim2real_mjx/hiperparametros.py",
    "xml_sha256": expected_xml,
}.items():
    source_hash = hashlib.sha256((repository / relative_path).read_bytes()).hexdigest()
    expect(model.get(metadata_key), source_hash, f"hash de {relative_path}")

expect(environment.get("fase_curriculum_recompensa"), 2, "fase del entorno")
expect(environment.get("episode_length"), 1500, "duracion de episodio del entorno")
expect(environment.get("xml_path"), expected_xml, "XML del entorno")
expect(curriculum.get("fase"), 2, "fase del curriculum")
expect(curriculum.get("ajustes", {}).get("xml_path"), expected_xml, "XML del curriculum")
expect(hyperparameters.get("episode_length"), 1500, "duracion de episodio PPO")

network_factory = network.get("network_factory_kwargs", {})
observation_state = network.get("observation_size", {}).get("state", {})
expect(network.get("action_size"), 12, "numero de acciones")
expect(network.get("normalize_observations"), True, "normalizacion de observaciones")
expect(observation_state.get("shape"), [59], "dimension de observacion")
expect(network_factory.get("policy_hidden_layer_sizes"), [256, 256], "capas de la politica")
expect(network_factory.get("value_hidden_layer_sizes"), [256, 256], "capas del critico")
expect(network_factory.get("activation"), "silu", "activacion de la red")
expect(network_factory.get("distribution_type"), "tanh_normal", "distribucion de acciones")

windows_absolute = re.compile(r"^[A-Za-z]:[\\/]")


def inspect_strings(value, source: Path, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            inspect_strings(child, source, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            inspect_strings(child, source, f"{location}[{index}]")
    elif isinstance(value, str):
        if value.startswith(("/", "\\")) or windows_absolute.match(value):
            fail(f"ruta absoluta en {source.relative_to(root)} ({location}): {value!r}.")


json_paths = set(root.rglob("*.json"))
json_paths.update(
    {
        root / "checkpoints/000045932544/_CHECKPOINT_METADATA",
        root / "checkpoints/000045932544/_METADATA",
        root / "checkpoints/000045932544/_sharding",
        root / "checkpoints/000045932544/array_metadatas/process_0",
    }
)
for json_path in sorted(json_paths):
    try:
        with json_path.open(encoding="utf-8") as stream:
            inspect_strings(json.load(stream), json_path)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"JSON no valido en {json_path.relative_to(root)}: {exc}")
PY

if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  RUNTIME_CHECKPOINT_PROBE="checkpoints/prueba-runtime/000000000001/estado"
  if ! git -C "${REPO_ROOT}" check-ignore --quiet --no-index -- "${RUNTIME_CHECKPOINT_PROBE}"; then
    fail "los checkpoints arbitrarios de ejecucion ya no estan ignorados por Git."
  fi
  if ! git -C "${REPO_ROOT}" check-ignore --quiet --no-index -- \
    "modelo_preentrenado/modelo-no-publicado/checkpoints/000000000001/pesos"; then
    fail "un modelo futuro no curado podria versionarse por accidente."
  fi

  for published_path in \
    "${MODEL_REL}/model.json" \
    "${MODEL_REL}/${PRIMARY_SHARD_REL}"; do
    if git -C "${REPO_ROOT}" check-ignore --quiet --no-index -- "${published_path}"; then
      fail "Git esta ignorando parte de la red publicada: ${published_path}."
    fi
  done
fi

LAUNCHER="${REPO_ROOT}/scripts/sim2real.sh"
[[ -f "${LAUNCHER}" ]] || fail "falta el lanzador principal."

LAUNCHER_ARGS="$(
  bash -s -- "${LAUNCHER}" <<'BASH'
set -euo pipefail
source "$1"
ULTIMA_EJECUCION="/tmp/una-ejecucion-local-que-no-debe-usarse/ultima_ejecucion.txt"
verificar_modelo_preentrenado() { :; }
visualizar_resultados() { printf '<%s>\n' "$@"; }
visualizar_modelo_preentrenado --solo-comprobar
BASH
)"
EXPECTED_LAUNCHER_ARGS="$(printf '<%s>\n' \
  '--solo-comprobar' \
  '--ruta-checkpoint' \
  "${CHECKPOINT_DIR}" \
  '--longitud-episodio' \
  '1500')"
[[ "${LAUNCHER_ARGS}" == "${EXPECTED_LAUNCHER_ARGS}" ]] || \
  fail "visualizar-modelo-preentrenado no fija exclusivamente la red publicada y el episodio 1500."

if bash -s -- "${LAUNCHER}" >/dev/null 2>&1 <<'BASH'
set -euo pipefail
source "$1"
verificar_modelo_preentrenado() { :; }
visualizar_resultados() { :; }
visualizar_modelo_preentrenado --ruta-checkpoint /tmp/red-equivocada
BASH
then
  fail "visualizar-modelo-preentrenado permite sustituir el checkpoint publicado."
fi

HELP_OUTPUT="$(
  bash -s -- "${LAUNCHER}" <<'BASH'
set -euo pipefail
source "$1"
verificar_modelo_preentrenado() { return 91; }
visualizar_resultados() { return 92; }
visualizar_modelo_preentrenado --help
BASH
)"
[[ "${HELP_OUTPUT}" == *"45.932.544"* && "${HELP_OUTPUT}" == *"ultima_ejecucion.txt"* ]] || \
  fail "la ayuda de visualizar-modelo-preentrenado no explica la identidad inmutable del modelo."

CORRUPTED_MODEL_ROOT="$(mktemp -d)"
cp -a -- "${MODEL_DIR}" "${CORRUPTED_MODEL_ROOT}/model"
printf '\n' >>"${CORRUPTED_MODEL_ROOT}/model/model.json"
if bash -s -- "${LAUNCHER}" "${CORRUPTED_MODEL_ROOT}/model" >/dev/null 2>&1 <<'BASH'
set -euo pipefail
source "$1"
DIRECTORIO_MODELO_PREENTRENADO="$2"
CHECKPOINT_PREENTRENADO="${DIRECTORIO_MODELO_PREENTRENADO}/checkpoints/000045932544"
verificar_modelo_preentrenado
BASH
then
  fail "el lanzador acepta una copia alterada de la red preentrenada."
fi

echo "PRETRAINED_MODEL_TESTS_OK"
