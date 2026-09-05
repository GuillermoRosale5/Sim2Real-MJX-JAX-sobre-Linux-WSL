#!/usr/bin/env python3
"""Visualiza SIM2REAL en una pose horneada en XML.

Por defecto abre la pose ideal:
  q1 =   0 deg
  q2 = +28 deg
  q3 = -93 deg
  altura cuerpo = 10.0 cm

Uso:
  python scripts/ver_pose_ideal.py
  python scripts/ver_pose_ideal.py --pose suelo2
  python scripts/ver_pose_ideal.py --pose ideal_mas_alta
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import mujoco
    import mujoco.viewer
except ModuleNotFoundError as exc:
    print(
        f"Falta {exc.name}. Ejecuta desde WSL con el entorno correcto:\n"
        "  python scripts/ver_pose_ideal.py",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


BASE_DIR = Path(__file__).parent.parent / "sim2real_mjx" / "xmls"

POSE_FILES = {
    "ideal": "ROBOT_POSE_IDEAL.xml",
    "suelo2": "ROBOT_POSE_SUELO2.xml",
    "ideal_mas_alta": "ROBOT_POSE_IDEAL_MAS_ALTA.xml",
}

parser = argparse.ArgumentParser()
parser.add_argument("--pose", choices=POSE_FILES, default="ideal")
args = parser.parse_args()

XML = BASE_DIR / POSE_FILES[args.pose]

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

altura_cm = data.qpos[2] * 100
print(f"XML: {XML.name}")
print(f"Pose ideal cargada - altura cuerpo: {altura_cm:.1f} cm")
print("Cierra la ventana para terminar.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()
