"""Entrenamiento PPO genérico con MJX/JAX/Brax."""

from __future__ import annotations

import argparse
import atexit
import csv
import datetime as dt
import functools
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import sys
import tempfile
import time
import traceback
from typing import Any
import warnings

import flax.linen as nn
import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from mujoco_playground import wrapper

from sim2real_mjx.hiperparametros import configuracion_a_diccionario
from sim2real_mjx.hiperparametros import configuracion_ppo_completa
from sim2real_mjx.hiperparametros import configuracion_ppo_depuracion
from sim2real_mjx.hiperparametros import configuracion_ppo_ligera
from sim2real_mjx.hiperparametros import configuracion_ppo_ligera_rapida
from sim2real_mjx.curriculo_recompensas import aplicar_fase_curriculo_recompensa
from sim2real_mjx.entorno_robot_mjx import EntornoRobotMJX
from sim2real_mjx.entorno_robot_mjx import VERSION_DIAGNOSTICO_RECOMPENSA
from sim2real_mjx.entorno_robot_mjx import default_config


warnings.filterwarnings(
    "ignore",
    message="overflow encountered in cast",
    category=RuntimeWarning,
)

CAMPOS_HISTORIAL_RECOMPENSAS = [
    "reward_total",
    "positive_reward",
    "supervivencia_reward_ponderado",
    "vector_gravedad_base_reward_ponderado",
    "vector_gravedad_reward_ponderado",
    "vector_gravedad_estabilidad_reward_ponderado",
    "poligono_CoG_reward_ponderado",
    "altura_reward_ponderado",
    "plano_CoG_arana_reward_ponderado",
    "contactos_suelo_reward_ponderado",
    "soporte_estatico_reward_ponderado",
    "apertura_efectores_q3_cerca_reward_ponderado",
    "q1_centrados_reward_ponderado",
    "q3_separados_centro_reward_ponderado",
    "simetria_patas_reward_ponderado",
    "contacto_invalido_penalty_ponderado",
    "contacto_invalido_persistente_penalty_ponderado",
    "efector_encima_q3_penalty_ponderado",
    "velocidad_vertical_cuerpo_penalty_ponderado",
    "velocidad_angular_cuerpo_penalty_ponderado",
    "control_penalty_ponderado",
    "cambio_accion_penalty_ponderado",
    "limite_articular_penalty_ponderado",
    "inclinacion_suave_penalty_ponderado",
    "penalties",
    "num_steps",
    "elapsed_seconds",
    "elapsed_hours",
    "percent",
    "source",
    "eval_episode_reward",
    "eval_episode_length",
    "eval_reward_per_step",
    "wall_steps_per_second",
    "steps_per_second",
    "training_steps_per_second",
    "salud_reward",
    "tiempo_vivo_reward",
    "verticalidad_reward",
    "cuerpo_paralelo_reward",
    "altura_legacy_reward",
    "alineacion_q1_reward",
    "contacto_pies_reward",
    "centro_soporte_reward",
    "q3_debajo_cuerpo_reward",
    "apertura_q3_reward",
    "pose_arana_baja_reward",
    "simetria_patas_legacy_reward",
    "supervivencia_reward_gate",
    "inclinacion_excesiva_penalty",
    "pies_aire_penalty",
    "contacto_suelo_invalido_penalty",
    "contacto_invalido_legacy_penalty",
    "efector_encima_q3_legacy_penalty",
    "terminal_failure_penalty_legacy",
    "vector_gravedad_penalty_legacy",
    "velocidad_penalty",
    "done_nonfinite",
    "done_unhealthy_z",
    "done_critical_z",
    "done_qpos_explosive",
    "done_qvel_explosive",
    "done_invalid_ground_contact",
    "done_terminal_failure",
    "done_supervivencia_fallida",
    "done_boca_arriba",
    "done_tilt_60",
    "done_invalid_contact",
    "done_time_limit",
    "state_z",
    "state_alive_reward_steps",
    "state_alive_reward_seconds",
    "state_episode_step",
    "state_episode_time_seconds",
    "state_episode_length_steps",
    "state_grace_steps",
    "state_uprightness",
    "state_tilt_degrees",
    "state_inclinacion_suave_penalty",
    "state_tilt_ratio",
    "state_excessive_tilt",
    "state_foot_contacts",
    "state_total_foot_clearance",
    "state_foot_clearance_ratio",
    "state_max_foot_clearance",
    "state_q3_body_z_margin_mean",
    "state_q3_radial_distance_mean",
    "state_low_spider_pose_reward",
    "state_low_spider_body_z_reward",
    "state_low_spider_q3_above_body_reward",
    "state_low_spider_foot_clearance_reward",
    "state_low_spider_support_reward",
    "state_low_spider_level_reward",
    "state_low_spider_stillness_reward",
    "state_low_spider_body_angvel_reward",
    "state_low_spider_cambio_accion_penalty",
    "state_low_spider_effector_below_q3_reward",
    "state_q3_above_body_margin_mean",
    "state_effector_distance_xy_mean",
    "state_effector_distance_xy_min",
    "state_excessive_foot_clearance",
    "state_all_feet_near_ground",
    "state_valid_foot_spread",
    "state_valid_support",
    "state_hard_gate",
    "state_valid_body_support",
    "state_enough_foot_contacts",
    "state_foot_contact_reward",
    "state_support_center_reward",
    "state_height_reward",
    "state_level_reward",
    "state_foot_clearance_reward",
    "state_effector_below_q3_reward",
    "state_stillness_reward",
    "state_suavidad_accion_reward",
    "state_quality_reward",
    "state_time_factor",
    "state_stability_reward_mature",
    "state_reward_per_step_unclipped",
    "state_reward_zeroed_by_bad_support",
    "state_supervivencia_reward",
    "state_vector_gravedad_reward",
    "state_vector_gravedad_lineal_reward_base",
    "state_vector_gravedad_lineal_reward",
    "state_upright_direction_gate",
    "state_vector_gravedad_estabilidad_reward",
    "state_contacto_invalido_penalty",
    "state_contacto_valido_reward",
    "state_filtro_contacto_soporte",
    "state_filtro_contacto_geometria",
    "state_filtro_contacto_altura",
    "state_contacto_invalido_persistente_penalty",
    "state_poligono_CoG_reward",
    "state_poligono_CoG_reward_filtrado",
    "state_altura_reward",
    "state_altura_reward_filtrada",
    "state_plano_CoG_arana_reward",
    "state_plano_CoG_arana_local_z",
    "state_contactos_suelo_reward",
    "state_contactos_suelo_reward_filtrado",
    "state_soporte_estatico_reward",
    "state_apertura_efectores_q3_cerca_reward",
    "state_apertura_efectores_reward",
    "state_q3_cerca_CoM_reward",
    "state_effector_radial_distance_mean",
    "state_q1_centrados_reward",
    "state_q3_separados_centro_reward",
    "state_simetria_patas_reward",
    "state_gravity_lateral_error",
    "state_g_body_x",
    "state_g_body_y",
    "state_g_body_z",
    "state_base_vz",
    "state_base_angvel_roll_pitch_norm",
    "state_limite_articular_penalty",
    "state_reward_raw",
    "state_reward_scaled",
    "state_fase_curriculum_recompensa",
    "state_invalid_ground_contact",
    "state_invalid_ground_contact_count",
    "state_invalid_ground_contact_steps",
    "state_num_valid_foot_contacts",
    "state_num_invalid_contacts",
    "state_body_or_chassis_touching_ground",
    "state_body_or_chassis_ground_contact_steps",
    "state_elbow_or_knee_touching_ground",
    "state_grace_period",
    "state_first_invalid_ground_geom_id",
    "state_first_invalid_non_q4_geom_id",
    "state_first_invalid_contact_geom1_id",
    "state_first_invalid_contact_geom2_id",
    "state_first_invalid_contact_index",
    "state_first_invalid_contact_dist",
    "state_effector_above_q3",
    "state_max_effector_q3_z_delta",
    "fase_curriculum_recompensa",
    "nombre_curriculum_recompensa",
    "debug_version",
    "debug_episode_length",
    "debug_action_scale",
    "debug_reward_scale",
    "debug_reward_clip_max",
    # Métricas nuevas: recompensas ponderadas y diagnóstico de postura y altura.
    "pose_articular_reward_ponderado",
    "velocidad_lineal_cero_reward_ponderado",
    "velocidad_angular_cero_reward_ponderado",
    "state_pose_articular_reward",
    "state_velocidad_lineal_cero_reward",
    "state_velocidad_angular_cero_reward",
    "state_k_c_curriculum_penalizaciones",
    "state_error_pose_medio",
    "state_error_altura",
    "state_sigma_pose_actual",
    "state_sigma_altura_actual",
    # Métricas del sistema de postura XML (currículo de recompensa v2026051903).
    "pose_imitacion_reward_ponderado",
    "pose_xml_final_reward_ponderado",
    "soporte_estatico_reward_filtrado_ponderado",
    "contactos_suelo_reward_filtrado_ponderado",
    "state_pose_imitacion_reward",
    "state_pose_xml_final_reward",
    "state_xml_pose_success_score",
    "state_xml_pose_success_score_with_support",
    "state_pose_imitacion_error_rms",
    "state_pose_imitacion_error_medio_deg",
    "state_pose_imitacion_sigma",
    "state_pose_imitacion_porcentaje",
    "state_target_body_z_referencia",
    "state_error_altura_xml",
    "state_support_gate",
    "state_pose_gate",
    "state_height_gate",
    "state_level_gate",
    "state_level_gate_strict",
    "state_height_gate_strict",
    "state_valid_contact_gate",
    "state_support_contact_gate",
    "cuerpo_paralelo_reward_ponderado",
    "apertura_efectores_xml_reward_ponderado",
    "q3_distancia_xml_reward_ponderado",
    "velocidad_articular_cero_reward_ponderado",
    "altura_exceso_penalty_ponderado",
    "altura_baja_penalty_ponderado",
    "rodilla_suelo_penalty_ponderado",
    "cuerpo_chasis_suelo_penalty_ponderado",
    "state_cuerpo_paralelo_reward",
    "state_altura_exceso_penalty",
    "state_altura_baja_penalty",
    "state_apertura_efectores_xml_reward",
    "state_q3_distancia_xml_reward",
    "state_velocidad_articular_cero_reward",
    "debug_reward_version",
    "debug_ref_body_z",
]

MAPA_METRICAS_RECOMPENSA = {
    "reward_total": ("reward_total",),
    "salud_reward": ("eval/episode_reward/salud", "episode/reward/salud"),
    "tiempo_vivo_reward": (
        "eval/episode_tiempo_vivo_reward",
        "episode/tiempo_vivo_reward",
    ),
    "verticalidad_reward": ("eval/episode_verticalidad_reward", "episode/verticalidad_reward"),
    "cuerpo_paralelo_reward": (
        "eval/episode_cuerpo_paralelo_reward",
        "episode/cuerpo_paralelo_reward",
    ),
    "altura_legacy_reward": ("eval/episode_altura_reward", "episode/altura_reward_ponderado"),
    "alineacion_q1_reward": (
        "eval/episode_alineacion_q1_reward",
        "episode/alineacion_q1_reward",
    ),
    "contacto_pies_reward": (
        "eval/episode_contacto_pies_reward",
        "episode/contacto_pies_reward",
    ),
    "centro_soporte_reward": (
        "eval/episode_centro_soporte_reward",
        "episode/centro_soporte_reward",
    ),
    "q3_debajo_cuerpo_reward": (
        "eval/episode_q3_debajo_cuerpo_reward",
        "episode/q3_debajo_cuerpo_reward",
    ),
    "apertura_q3_reward": (
        "eval/episode_apertura_q3_reward",
        "episode/apertura_q3_reward",
    ),
    "pose_arana_baja_reward": (
        "eval/episode_pose_arana_baja_reward",
        "episode/pose_arana_baja_reward",
    ),
    "simetria_patas_legacy_reward": (
        "eval/episode_simetria_patas_legacy_reward",
        "episode/simetria_patas_legacy_reward",
    ),
    "supervivencia_reward_gate": (
        "eval/episode_supervivencia_gate_reward",
        "episode/supervivencia_gate_reward",
    ),
    "supervivencia_reward_ponderado": ("supervivencia_reward_ponderado",),
    "vector_gravedad_base_reward_ponderado": ("vector_gravedad_base_reward_ponderado",),
    "vector_gravedad_reward_ponderado": ("vector_gravedad_reward_ponderado",),
    "vector_gravedad_estabilidad_reward_ponderado": (
        "vector_gravedad_estabilidad_reward_ponderado",
        "eval/episode_vector_gravedad_estabilidad_reward_ponderado",
        "episode/vector_gravedad_estabilidad_reward_ponderado",
    ),
    "poligono_CoG_reward_ponderado": ("poligono_CoG_reward_ponderado",),
    "altura_reward_ponderado": ("altura_reward_ponderado",),
    "plano_CoG_arana_reward_ponderado": ("plano_CoG_arana_reward_ponderado",),
    "contactos_suelo_reward_ponderado": ("contactos_suelo_reward_ponderado",),
    "soporte_estatico_reward_ponderado": (
        "eval/episode_soporte_estatico_reward",
        "episode/soporte_estatico_reward_ponderado",
    ),
    "apertura_efectores_q3_cerca_reward_ponderado": (
        "apertura_efectores_q3_cerca_reward_ponderado",
    ),
    "q1_centrados_reward_ponderado": ("q1_centrados_reward_ponderado",),
    "q3_separados_centro_reward_ponderado": ("q3_separados_centro_reward_ponderado",),
    "simetria_patas_reward_ponderado": ("simetria_patas_reward_ponderado",),
    "positive_reward": ("positive_reward",),
    "pies_aire_penalty": (
        "eval/episode_pies_aire_penalty",
        "episode/pies_aire_penalty",
    ),
    "contacto_suelo_invalido_penalty": (
        "eval/episode_contacto_suelo_invalido_penalty",
        "episode/contacto_suelo_invalido_penalty",
    ),
    "contacto_invalido_penalty_ponderado": ("contacto_invalido_penalty_ponderado",),
    "contacto_invalido_persistente_penalty_ponderado": (
        "contacto_invalido_persistente_penalty_ponderado",
    ),
    "efector_encima_q3_legacy_penalty": (
        "eval/episode_efector_encima_q3_penalty",
        "episode/efector_encima_q3_penalty_ponderado",
    ),
    "terminal_failure_penalty_legacy": (
        "eval/episode_terminal_failure_penalty",
        "episode/terminal_failure_penalty",
    ),
    "vector_gravedad_penalty_legacy": (
        "eval/episode_vector_gravedad_penalty",
        "episode/vector_gravedad_penalty",
    ),
    "efector_encima_q3_penalty_ponderado": ("efector_encima_q3_penalty_ponderado",),
    "velocidad_vertical_cuerpo_penalty_ponderado": ("velocidad_vertical_cuerpo_penalty_ponderado",),
    "velocidad_angular_cuerpo_penalty_ponderado": ("velocidad_angular_cuerpo_penalty_ponderado",),
    "inclinacion_excesiva_penalty": (
        "eval/episode_inclinacion_excesiva_penalty",
        "episode/inclinacion_excesiva_penalty",
    ),
    "control_penalty_ponderado": ("control_penalty_ponderado",),
    "velocidad_penalty": (
        "eval/episode_velocidad_penalty",
        "episode/velocidad_penalty",
    ),
    "cambio_accion_penalty_ponderado": ("cambio_accion_penalty_ponderado",),
    "limite_articular_penalty_ponderado": ("limite_articular_penalty_ponderado",),
    "penalties": ("penalties",),
    "done_nonfinite": ("eval/episode_done/nonfinite", "episode/done/nonfinite"),
    "done_unhealthy_z": (
        "eval/episode_done/unhealthy_z",
        "episode/done/unhealthy_z",
    ),
    "done_critical_z": (
        "eval/episode_done/critical_z",
        "episode/done/critical_z",
    ),
    "done_qpos_explosive": (
        "eval/episode_done/qpos_explosive",
        "episode/done/qpos_explosive",
    ),
    "done_qvel_explosive": (
        "eval/episode_done/qvel_explosive",
        "episode/done/qvel_explosive",
    ),
    "done_invalid_ground_contact": (
        "eval/episode_done/invalid_ground_contact",
        "episode/done/invalid_ground_contact",
    ),
    "done_terminal_failure": (
        "eval/episode_done/terminal_failure",
        "episode/done/terminal_failure",
    ),
    "done_supervivencia_fallida": (
        "eval/episode_done/supervivencia_fallida",
        "episode/done/supervivencia_fallida",
    ),
    "done_boca_arriba": (
        "eval/episode_done/boca_arriba",
        "episode/done/boca_arriba",
    ),
    "done_tilt_60": ("eval/episode_done/tilt_60", "episode/done/tilt_60"),
    "done_invalid_contact": (
        "eval/episode_done/invalid_contact",
        "episode/done/invalid_contact",
    ),
    "done_time_limit": (
        "eval/episode_done/time_limit",
        "episode/done/time_limit",
    ),
    "state_z": ("eval/episode_state/z", "episode/state/z"),
    "state_alive_reward_steps": (
        "eval/episode_state/alive_reward_steps",
        "episode/state/alive_reward_steps",
    ),
    "state_alive_reward_seconds": (
        "eval/episode_state/alive_reward_seconds",
        "episode/state/alive_reward_seconds",
    ),
    "state_episode_step": (
        "eval/episode_state/episode_step",
        "episode/state/episode_step",
    ),
    "state_episode_time_seconds": (
        "eval/episode_state/episode_time_seconds",
        "episode/state/episode_time_seconds",
    ),
    "state_episode_length_steps": (
        "eval/episode_state/episode_length_steps",
        "episode/state/episode_length_steps",
    ),
    "state_grace_steps": (
        "eval/episode_state/grace_steps",
        "episode/state/grace_steps",
    ),
    "state_uprightness": (
        "eval/episode_state/uprightness",
        "episode/state/uprightness",
    ),
    "state_tilt_degrees": (
        "eval/episode_state/tilt_degrees",
        "episode/state/tilt_degrees",
    ),
    "state_tilt_ratio": (
        "eval/episode_state/tilt_ratio",
        "episode/state/tilt_ratio",
    ),
    "state_excessive_tilt": (
        "eval/episode_state/excessive_tilt",
        "episode/state/excessive_tilt",
    ),
    "state_foot_contacts": (
        "eval/episode_state/foot_contacts",
        "episode/state/foot_contacts",
    ),
    "state_total_foot_clearance": (
        "eval/episode_state/total_foot_clearance",
        "episode/state/total_foot_clearance",
    ),
    "state_foot_clearance_ratio": (
        "eval/episode_state/foot_clearance_ratio",
        "episode/state/foot_clearance_ratio",
    ),
    "state_max_foot_clearance": (
        "eval/episode_state/max_foot_clearance",
        "episode/state/max_foot_clearance",
    ),
    "state_q3_body_z_margin_mean": (
        "eval/episode_state/q3_body_z_margin_mean",
        "episode/state/q3_body_z_margin_mean",
    ),
    "state_q3_radial_distance_mean": (
        "eval/episode_state/q3_radial_distance_mean",
        "episode/state/q3_radial_distance_mean",
    ),
    "state_low_spider_pose_reward": (
        "eval/episode_state/low_spider_pose_reward",
        "episode/state/low_spider_pose_reward",
    ),
    "state_low_spider_body_z_reward": (
        "eval/episode_state/low_spider_body_z_reward",
        "episode/state/low_spider_body_z_reward",
    ),
    "state_low_spider_q3_above_body_reward": (
        "eval/episode_state/low_spider_q3_above_body_reward",
        "episode/state/low_spider_q3_above_body_reward",
    ),
    "state_low_spider_foot_clearance_reward": (
        "eval/episode_state/low_spider_foot_clearance_reward",
        "episode/state/low_spider_foot_clearance_reward",
    ),
    "state_low_spider_support_reward": (
        "eval/episode_state/low_spider_support_reward",
        "episode/state/low_spider_support_reward",
    ),
    "state_low_spider_level_reward": (
        "eval/episode_state/low_spider_level_reward",
        "episode/state/low_spider_level_reward",
    ),
    "state_low_spider_stillness_reward": (
        "eval/episode_state/low_spider_stillness_reward",
        "episode/state/low_spider_stillness_reward",
    ),
    "state_low_spider_body_angvel_reward": (
        "eval/episode_state/low_spider_body_angvel_reward",
        "episode/state/low_spider_body_angvel_reward",
    ),
    "state_low_spider_cambio_accion_penalty": (
        "eval/episode_state/low_spider_cambio_accion_penalty",
        "episode/state/low_spider_cambio_accion_penalty",
    ),
    "state_low_spider_effector_below_q3_reward": (
        "eval/episode_state/low_spider_effector_below_q3_reward",
        "episode/state/low_spider_effector_below_q3_reward",
    ),
    "state_q3_above_body_margin_mean": (
        "eval/episode_state/q3_above_body_margin_mean",
        "episode/state/q3_above_body_margin_mean",
    ),
    "state_effector_distance_xy_mean": (
        "eval/episode_state/effector_distance_xy_mean",
        "episode/state/effector_distance_xy_mean",
    ),
    "state_effector_distance_xy_min": (
        "eval/episode_state/effector_distance_xy_min",
        "episode/state/effector_distance_xy_min",
    ),
    "state_excessive_foot_clearance": (
        "eval/episode_state/excessive_foot_clearance",
        "episode/state/excessive_foot_clearance",
    ),
    "state_all_feet_near_ground": (
        "eval/episode_state/all_feet_near_ground",
        "episode/state/all_feet_near_ground",
    ),
    "state_valid_foot_spread": (
        "eval/episode_state/valid_foot_spread",
        "episode/state/valid_foot_spread",
    ),
    "state_valid_support": (
        "eval/episode_state/valid_support",
        "episode/state/valid_support",
    ),
    "state_hard_gate": (
        "eval/episode_state/hard_gate",
        "episode/state/hard_gate",
    ),
    "state_valid_body_support": (
        "eval/episode_state/valid_body_support",
        "episode/state/valid_body_support",
    ),
    "state_enough_foot_contacts": (
        "eval/episode_state/enough_foot_contacts",
        "episode/state/enough_foot_contacts",
    ),
    "state_foot_contact_reward": (
        "eval/episode_state/foot_contact_reward",
        "episode/state/foot_contact_reward",
    ),
    "state_support_center_reward": (
        "eval/episode_state/support_center_reward",
        "episode/state/support_center_reward",
    ),
    "state_height_reward": (
        "eval/episode_state/height_reward",
        "episode/state/height_reward",
    ),
    "state_level_reward": (
        "eval/episode_state/level_reward",
        "episode/state/level_reward",
    ),
    "state_foot_clearance_reward": (
        "eval/episode_state/foot_clearance_reward",
        "episode/state/foot_clearance_reward",
    ),
    "state_effector_below_q3_reward": (
        "eval/episode_state/effector_below_q3_reward",
        "episode/state/effector_below_q3_reward",
    ),
    "state_stillness_reward": (
        "eval/episode_state/stillness_reward",
        "episode/state/stillness_reward",
    ),
    "state_suavidad_accion_reward": (
        "eval/episode_state/suavidad_accion_reward",
        "episode/state/suavidad_accion_reward",
    ),
    "state_quality_reward": (
        "eval/episode_state/quality_reward",
        "episode/state/quality_reward",
    ),
    "state_time_factor": (
        "eval/episode_state/time_factor",
        "episode/state/time_factor",
    ),
    "state_stability_reward_mature": (
        "eval/episode_state/stability_reward_mature",
        "episode/state/stability_reward_mature",
    ),
    "state_reward_per_step_unclipped": (
        "eval/episode_state/reward_per_step_unclipped",
        "episode/state/reward_per_step_unclipped",
    ),
    "state_reward_zeroed_by_bad_support": (
        "eval/episode_state/reward_zeroed_by_bad_support",
        "episode/state/reward_zeroed_by_bad_support",
    ),
    "state_supervivencia_reward": (
        "eval/episode_state/supervivencia_reward",
        "episode/state/supervivencia_reward",
    ),
    "state_vector_gravedad_reward": (
        "eval/episode_state/vector_gravedad_reward",
        "episode/state/vector_gravedad_reward",
    ),
    "state_vector_gravedad_lineal_reward_base": (
        "eval/episode_state/vector_gravedad_lineal_reward_base",
        "episode/state/vector_gravedad_lineal_reward_base",
    ),
    "state_upright_direction_gate": (
        "eval/episode_state/upright_direction_gate",
        "episode/state/upright_direction_gate",
    ),
    "state_contacto_invalido_penalty": (
        "eval/episode_state/contacto_invalido_penalty",
        "episode/state/contacto_invalido_penalty",
    ),
    "state_contacto_valido_reward": (
        "eval/episode_state/contacto_valido_reward",
        "episode/state/contacto_valido_reward",
    ),
    "state_filtro_contacto_soporte": (
        "eval/episode_state/filtro_contacto_soporte",
        "episode/state/filtro_contacto_soporte",
    ),
    "state_filtro_contacto_geometria": (
        "eval/episode_state/filtro_contacto_geometria",
        "episode/state/filtro_contacto_geometria",
    ),
    "state_filtro_contacto_altura": (
        "eval/episode_state/filtro_contacto_altura",
        "episode/state/filtro_contacto_altura",
    ),
    "state_contacto_invalido_persistente_penalty": (
        "eval/episode_state/contacto_invalido_persistente_penalty",
        "episode/state/contacto_invalido_persistente_penalty",
    ),
    "state_poligono_CoG_reward": (
        "eval/episode_state/poligono_CoG_reward",
        "episode/state/poligono_CoG_reward",
    ),
    "state_poligono_CoG_reward_filtrado": (
        "eval/episode_state/poligono_CoG_reward_filtrado",
        "episode/state/poligono_CoG_reward_filtrado",
    ),
    "state_altura_reward": (
        "eval/episode_state/altura_reward",
        "episode/state/altura_reward",
    ),
    "state_altura_reward_filtrada": (
        "eval/episode_state/altura_reward_filtrada",
        "episode/state/altura_reward_filtrada",
    ),
    "state_plano_CoG_arana_reward": (
        "eval/episode_state/plano_CoG_arana_reward",
        "episode/state/plano_CoG_arana_reward",
    ),
    "state_plano_CoG_arana_local_z": (
        "eval/episode_state/plano_CoG_arana_local_z",
        "episode/state/plano_CoG_arana_local_z",
    ),
    "state_contactos_suelo_reward": (
        "eval/episode_state/contactos_suelo_reward",
        "episode/state/contactos_suelo_reward",
    ),
    "state_contactos_suelo_reward_filtrado": (
        "eval/episode_state/contactos_suelo_reward_filtrado",
        "episode/state/contactos_suelo_reward_filtrado",
    ),
    "state_soporte_estatico_reward": (
        "eval/episode_state/soporte_estatico_reward",
        "episode/state/soporte_estatico_reward",
    ),
    "state_apertura_efectores_q3_cerca_reward": (
        "eval/episode_state/apertura_efectores_q3_cerca_reward",
        "episode/state/apertura_efectores_q3_cerca_reward",
    ),
    "state_apertura_efectores_reward": (
        "eval/episode_state/apertura_efectores_reward",
        "episode/state/apertura_efectores_reward",
    ),
    "state_q3_cerca_CoM_reward": (
        "eval/episode_state/q3_cerca_CoM_reward",
        "episode/state/q3_cerca_CoM_reward",
    ),
    "state_effector_radial_distance_mean": (
        "eval/episode_state/effector_radial_distance_mean",
        "episode/state/effector_radial_distance_mean",
    ),
    "state_q1_centrados_reward": (
        "eval/episode_state/q1_centrados_reward",
        "episode/state/q1_centrados_reward",
    ),
    "state_q3_separados_centro_reward": (
        "eval/episode_state/q3_separados_centro_reward",
        "episode/state/q3_separados_centro_reward",
    ),
    "state_simetria_patas_reward": (
        "eval/episode_state/simetria_patas_reward",
        "episode/state/simetria_patas_reward",
    ),
    "state_gravity_lateral_error": (
        "eval/episode_state/gravity_lateral_error",
        "episode/state/gravity_lateral_error",
    ),
    "state_g_body_x": ("eval/episode_state/g_body_x", "episode/state/g_body_x"),
    "state_g_body_y": ("eval/episode_state/g_body_y", "episode/state/g_body_y"),
    "state_g_body_z": ("eval/episode_state/g_body_z", "episode/state/g_body_z"),
    "state_base_vz": ("eval/episode_state/base_vz", "episode/state/base_vz"),
    "state_base_angvel_roll_pitch_norm": (
        "eval/episode_state/base_angvel_roll_pitch_norm",
        "episode/state/base_angvel_roll_pitch_norm",
    ),
    "state_limite_articular_penalty": (
        "eval/episode_state/limite_articular_penalty",
        "episode/state/limite_articular_penalty",
    ),
    "state_reward_raw": (
        "eval/episode_state/reward_raw",
        "episode/state/reward_raw",
    ),
    "state_reward_scaled": (
        "eval/episode_state/reward_scaled",
        "episode/state/reward_scaled",
    ),
    "state_fase_curriculum_recompensa": (
        "eval/episode_state/fase_curriculum_recompensa",
        "episode/state/fase_curriculum_recompensa",
    ),
    "state_invalid_ground_contact": (
        "eval/episode_state/invalid_ground_contact",
        "episode/state/invalid_ground_contact",
    ),
    "state_invalid_ground_contact_count": (
        "eval/episode_state/invalid_ground_contact_count",
        "episode/state/invalid_ground_contact_count",
    ),
    "state_invalid_ground_contact_steps": (
        "eval/episode_state/invalid_ground_contact_steps",
        "episode/state/invalid_ground_contact_steps",
    ),
    "state_num_valid_foot_contacts": (
        "eval/episode_state/num_valid_foot_contacts",
        "episode/state/num_valid_foot_contacts",
    ),
    "state_num_invalid_contacts": (
        "eval/episode_state/num_invalid_contacts",
        "episode/state/num_invalid_contacts",
    ),
    "state_body_or_chassis_touching_ground": (
        "eval/episode_state/body_or_chassis_touching_ground",
        "episode/state/body_or_chassis_touching_ground",
    ),
    "state_body_or_chassis_ground_contact_steps": (
        "eval/episode_state/body_or_chassis_ground_contact_steps",
        "episode/state/body_or_chassis_ground_contact_steps",
    ),
    "state_elbow_or_knee_touching_ground": (
        "eval/episode_state/elbow_or_knee_touching_ground",
        "episode/state/elbow_or_knee_touching_ground",
    ),
    "state_grace_period": (
        "eval/episode_state/grace_period",
        "episode/state/grace_period",
    ),
    "state_first_invalid_ground_geom_id": (
        "eval/episode_state/first_invalid_ground_geom_id",
        "episode/state/first_invalid_ground_geom_id",
    ),
    "state_first_invalid_non_q4_geom_id": (
        "eval/episode_state/first_invalid_non_q4_geom_id",
        "episode/state/first_invalid_non_q4_geom_id",
    ),
    "state_first_invalid_contact_geom1_id": (
        "eval/episode_state/first_invalid_contact_geom1_id",
        "episode/state/first_invalid_contact_geom1_id",
    ),
    "state_first_invalid_contact_geom2_id": (
        "eval/episode_state/first_invalid_contact_geom2_id",
        "episode/state/first_invalid_contact_geom2_id",
    ),
    "state_first_invalid_contact_index": (
        "eval/episode_state/first_invalid_contact_index",
        "episode/state/first_invalid_contact_index",
    ),
    "state_first_invalid_contact_dist": (
        "eval/episode_state/first_invalid_contact_dist",
        "episode/state/first_invalid_contact_dist",
    ),
    "state_effector_above_q3": (
        "eval/episode_state/effector_above_q3",
        "episode/state/effector_above_q3",
    ),
    "state_max_effector_q3_z_delta": (
        "eval/episode_state/max_effector_q3_z_delta",
        "episode/state/max_effector_q3_z_delta",
    ),
    "debug_version": ("eval/episode_debug/version", "episode/debug/version"),
    "debug_episode_length": (
        "eval/episode_debug/episode_length",
        "episode/debug/episode_length",
    ),
    "debug_action_scale": (
        "eval/episode_debug/action_scale",
        "episode/debug/action_scale",
    ),
    "debug_reward_scale": (
        "eval/episode_debug/reward_scale",
        "episode/debug/reward_scale",
    ),
    "debug_reward_clip_max": (
        "eval/episode_debug/reward_clip_max",
        "episode/debug/reward_clip_max",
    ),
    "pose_articular_reward_ponderado": (
        "eval/episode_pose_articular_reward_ponderado",
        "episode/pose_articular_reward_ponderado",
    ),
    "velocidad_lineal_cero_reward_ponderado": (
        "eval/episode_velocidad_lineal_cero_reward_ponderado",
        "episode/velocidad_lineal_cero_reward_ponderado",
    ),
    "velocidad_angular_cero_reward_ponderado": (
        "eval/episode_velocidad_angular_cero_reward_ponderado",
        "episode/velocidad_angular_cero_reward_ponderado",
    ),
    "state_pose_articular_reward": (
        "eval/episode_state/pose_articular_reward",
        "episode/state/pose_articular_reward",
    ),
    "state_velocidad_lineal_cero_reward": (
        "eval/episode_state/velocidad_lineal_cero_reward",
        "episode/state/velocidad_lineal_cero_reward",
    ),
    "state_velocidad_angular_cero_reward": (
        "eval/episode_state/velocidad_angular_cero_reward",
        "episode/state/velocidad_angular_cero_reward",
    ),
    "state_k_c_curriculum_penalizaciones": (
        "eval/episode_state/k_c_curriculum_penalizaciones",
        "episode/state/k_c_curriculum_penalizaciones",
    ),
    "state_error_pose_medio": (
        "eval/episode_state/error_pose_medio",
        "episode/state/error_pose_medio",
    ),
    "state_error_altura": (
        "eval/episode_state/error_altura",
        "episode/state/error_altura",
    ),
    "state_sigma_pose_actual": (
        "eval/episode_state/sigma_pose_actual",
        "episode/state/sigma_pose_actual",
    ),
    "state_sigma_altura_actual": (
        "eval/episode_state/sigma_altura_actual",
        "episode/state/sigma_altura_actual",
    ),
    # --- Métricas del sistema de postura XML (v2026051902) ---
    "pose_imitacion_reward_ponderado": (
        "eval/episode_pose_imitacion_reward_ponderado",
        "episode/pose_imitacion_reward_ponderado",
    ),
    "pose_xml_final_reward_ponderado": (
        "eval/episode_pose_xml_final_reward_ponderado",
        "episode/pose_xml_final_reward_ponderado",
    ),
    "cuerpo_paralelo_reward_ponderado": (
        "eval/episode_cuerpo_paralelo_reward_ponderado",
        "episode/cuerpo_paralelo_reward_ponderado",
    ),
    "apertura_efectores_xml_reward_ponderado": (
        "eval/episode_apertura_efectores_xml_reward_ponderado",
        "episode/apertura_efectores_xml_reward_ponderado",
    ),
    "q3_distancia_xml_reward_ponderado": (
        "eval/episode_q3_distancia_xml_reward_ponderado",
        "episode/q3_distancia_xml_reward_ponderado",
    ),
    "soporte_estatico_reward_filtrado_ponderado": (
        "eval/episode_soporte_estatico_reward_filtrado_ponderado",
        "episode/soporte_estatico_reward_filtrado_ponderado",
    ),
    "contactos_suelo_reward_filtrado_ponderado": (
        "eval/episode_contactos_suelo_reward_filtrado_ponderado",
        "episode/contactos_suelo_reward_filtrado_ponderado",
    ),
    "velocidad_articular_cero_reward_ponderado": (
        "eval/episode_velocidad_articular_cero_reward_ponderado",
        "episode/velocidad_articular_cero_reward_ponderado",
    ),
    "altura_exceso_penalty_ponderado": (
        "eval/episode_altura_exceso_penalty_ponderado",
        "episode/altura_exceso_penalty_ponderado",
    ),
    "altura_baja_penalty_ponderado": (
        "eval/episode_altura_baja_penalty_ponderado",
        "episode/altura_baja_penalty_ponderado",
    ),
    "rodilla_suelo_penalty_ponderado": (
        "eval/episode_rodilla_suelo_penalty_ponderado",
        "episode/rodilla_suelo_penalty_ponderado",
    ),
    "cuerpo_chasis_suelo_penalty_ponderado": (
        "eval/episode_cuerpo_chasis_suelo_penalty_ponderado",
        "episode/cuerpo_chasis_suelo_penalty_ponderado",
    ),
    "inclinacion_suave_penalty_ponderado": (
        "eval/episode_inclinacion_suave_penalty_ponderado",
        "episode/inclinacion_suave_penalty_ponderado",
    ),
    "state_pose_imitacion_reward": (
        "eval/episode_state/pose_imitacion_reward",
        "episode/state/pose_imitacion_reward",
    ),
    "state_pose_xml_final_reward": (
        "eval/episode_state/pose_xml_final_reward",
        "episode/state/pose_xml_final_reward",
    ),
    "state_xml_pose_success_score": (
        "eval/episode_state/xml_pose_success_score",
        "episode/state/xml_pose_success_score",
    ),
    "state_xml_pose_success_score_with_support": (
        "eval/episode_state/xml_pose_success_score_with_support",
        "episode/state/xml_pose_success_score_with_support",
    ),
    "state_pose_imitacion_error_rms": (
        "eval/episode_state/pose_imitacion_error_rms",
        "episode/state/pose_imitacion_error_rms",
    ),
    "state_pose_imitacion_error_medio_deg": (
        "eval/episode_state/pose_imitacion_error_medio_deg",
        "episode/state/pose_imitacion_error_medio_deg",
    ),
    "state_pose_imitacion_sigma": (
        "eval/episode_state/pose_imitacion_sigma",
        "episode/state/pose_imitacion_sigma",
    ),
    "state_pose_imitacion_porcentaje": (
        "eval/episode_state/pose_imitacion_porcentaje",
        "episode/state/pose_imitacion_porcentaje",
    ),
    "state_target_body_z_referencia": (
        "eval/episode_state/target_body_z_referencia",
        "episode/state/target_body_z_referencia",
    ),
    "state_error_altura_xml": (
        "eval/episode_state/error_altura_xml",
        "episode/state/error_altura_xml",
    ),
    "state_support_gate": (
        "eval/episode_state/support_gate",
        "episode/state/support_gate",
    ),
    "state_pose_gate": (
        "eval/episode_state/pose_gate",
        "episode/state/pose_gate",
    ),
    "state_height_gate": (
        "eval/episode_state/height_gate",
        "episode/state/height_gate",
    ),
    "state_level_gate": (
        "eval/episode_state/level_gate",
        "episode/state/level_gate",
    ),
    "state_level_gate_strict": (
        "eval/episode_state/level_gate_strict",
        "episode/state/level_gate_strict",
    ),
    "state_height_gate_strict": (
        "eval/episode_state/height_gate_strict",
        "episode/state/height_gate_strict",
    ),
    "state_valid_contact_gate": (
        "eval/episode_state/valid_contact_gate",
        "episode/state/valid_contact_gate",
    ),
    "state_support_contact_gate": (
        "eval/episode_state/support_contact_gate",
        "episode/state/support_contact_gate",
    ),
    "state_cuerpo_paralelo_reward": (
        "eval/episode_state/cuerpo_paralelo_reward",
        "episode/state/cuerpo_paralelo_reward",
    ),
    "state_vector_gravedad_lineal_reward": (
        "eval/episode_state/vector_gravedad_lineal_reward",
        "episode/state/vector_gravedad_lineal_reward",
    ),
    "state_vector_gravedad_estabilidad_reward": (
        "eval/episode_state/vector_gravedad_estabilidad_reward",
        "episode/state/vector_gravedad_estabilidad_reward",
    ),
    "state_altura_exceso_penalty": (
        "eval/episode_state/altura_exceso_penalty",
        "episode/state/altura_exceso_penalty",
    ),
    "state_altura_baja_penalty": (
        "eval/episode_state/altura_baja_penalty",
        "episode/state/altura_baja_penalty",
    ),
    "state_inclinacion_suave_penalty": (
        "eval/episode_state/inclinacion_suave_penalty",
        "episode/state/inclinacion_suave_penalty",
    ),
    "state_apertura_efectores_xml_reward": (
        "eval/episode_state/apertura_efectores_xml_reward",
        "episode/state/apertura_efectores_xml_reward",
    ),
    "state_q3_distancia_xml_reward": (
        "eval/episode_state/q3_distancia_xml_reward",
        "episode/state/q3_distancia_xml_reward",
    ),
    "state_velocidad_articular_cero_reward": (
        "eval/episode_state/velocidad_articular_cero_reward",
        "episode/state/velocidad_articular_cero_reward",
    ),
    "debug_reward_version": (
        "eval/episode_debug/reward_version",
        "episode/debug/reward_version",
    ),
    "debug_ref_body_z": (
        "eval/episode_debug/ref_body_z",
        "episode/debug/ref_body_z",
    ),
}

PHYSICAL_HISTORY_BASE_FIELDS = [
    "num_steps",
    "elapsed_seconds",
    "elapsed_hours",
    "percent",
    "source",
    "eval_episode_reward",
    "eval_episode_length",
    "eval_reward_per_step",
    "wall_steps_per_second",
    "steps_per_second",
    "training_steps_per_second",
]

PHYSICAL_METRIC_FIELDS = [
    "done/nonfinite",
    "done/unhealthy_z",
    "done/critical_z",
    "done/qpos_explosive",
    "done/qvel_explosive",
    "done/invalid_ground_contact",
    "done/terminal_failure",
    "done/supervivencia_fallida",
    "done/boca_arriba",
    "done/tilt_60",
    "done/invalid_contact",
    "done/time_limit",
    "final/body_x",
    "final/body_y",
    "final/body_z",
    "final/com_x",
    "final/com_y",
    "final/com_z",
    "final/body_tilt_degrees",
    "final/uprightness",
    "final/body_parallel_percent",
    "final/tilt_limit_percent",
    "final/effector_center_x",
    "final/effector_center_y",
    "final/effector_center_z",
    "final/effector_radius_from_com_mean",
    "final/effector_radius_from_com_min",
    "final/effector_radius_from_com_max",
    "final/effector_center_distance_to_com",
    "final/effector_radius_from_center_mean",
    "final/q3_body_z_margin_mean",
    "final/q3_above_body_margin_mean",
    "final/q3_below_body_percent",
    "final/q3_z_mean",
    "final/q3_z_min",
    "final/q3_z_max",
    "final/q3_radial_distance_mean",
    "final/q3_opening_percent",
    "final/effector_distance_xy_mean",
    "final/low_spider_pose_percent",
    "final/low_spider_body_z_percent",
    "final/low_spider_q3_above_body_percent",
    "final/low_spider_foot_clearance_percent",
    "final/low_spider_support_percent",
    "final/low_spider_level_percent",
    "final/low_spider_stillness_percent",
    "final/low_spider_body_angvel_percent",
    "final/low_spider_cambio_accion_percent",
    "final/low_spider_effector_below_q3_percent",
    "final/leg_symmetry_percent",
    "final/foot_contacts",
    "final/total_foot_clearance",
    "final/max_foot_clearance",
    "final/foot_clearance_percent",
    "final/hard_gate",
    "final/valid_body_support",
    "final/enough_foot_contacts",
    "final/foot_contact_reward",
    "final/support_center_reward",
    "final/height_reward",
    "final/level_reward",
    "final/foot_clearance_reward",
    "final/effector_below_q3_reward",
    "final/stillness_reward",
    "final/suavidad_accion_reward",
    "final/quality_reward",
    "final/time_factor",
    "final/reward_per_step_unclipped",
    "final/reward_zeroed_by_bad_support",
    "final/poligono_CoG_reward",
    "final/altura_reward",
    "final/plano_CoG_arana_reward",
    "final/contactos_suelo_reward",
    "final/soporte_estatico_reward",
    "final/apertura_efectores_q3_cerca_reward",
    "final/q1_centrados_reward",
    "final/q3_separados_centro_reward",
    "final/simetria_patas_reward",
    "final/gravity_lateral_error",
    "final/g_body_z",
    "final/reward_raw",
    "final/reward_scaled",
    "final/max_effector_q3_z_delta",
    "final/qvel_abs_max",
    "final/action_abs_mean",
    "final/action_abs_max",
    "final/first_invalid_ground_geom_id",
    "final/first_invalid_non_q4_geom_id",
    "final/first_invalid_contact_geom1_id",
    "final/first_invalid_contact_geom2_id",
    "final/first_invalid_contact_index",
    "final/first_invalid_contact_dist",
]

PHYSICAL_DATA_FIELDS = PHYSICAL_HISTORY_BASE_FIELDS + [
    name.replace("/", "_") for name in PHYSICAL_METRIC_FIELDS
]

PHYSICAL_METRIC_MAP = {
    name.replace("/", "_"): (f"eval/episode_{name}", f"episode/{name}")
    for name in PHYSICAL_METRIC_FIELDS
}

CLAVES_CONFIG_RECOMPENSA = [
    "reward_per_step_max",
    "reward_scale",
    "time_base",
    "time_tau",
    "terminal_failure_penalty",
    "supervivencia_reward_weight",
    "vector_gravedad_base_reward_weight",
    "vector_gravedad_reward_weight",
    "poligono_CoG_reward_weight",
    "altura_reward_weight",
    "contactos_suelo_reward_weight",
    "plano_CoG_arana_reward_weight",
    "soporte_estatico_reward_weight",
    "apertura_efectores_q3_cerca_reward_weight",
    "q1_centrados_reward_weight",
    "q3_separados_centro_reward_weight",
    "simetria_patas_reward_weight",
    "contacto_invalido_penalty_weight",
    "contacto_invalido_persistente_penalty_weight",
    "vector_gravedad_penalty_weight",
    "efector_encima_q3_penalty_weight",
    "velocidad_vertical_cuerpo_penalty_weight",
    "velocidad_angular_cuerpo_penalty_weight",
    "control_penalty_weight",
    "cambio_accion_penalty_weight",
    "limite_articular_penalty_weight",
    "velocidad_penalty_weight",
    "salud_reward",
    "healthy_z_min",
    "healthy_z_max",
    "max_episode_seconds",
    "critical_z_min",
    "critical_z_max",
    "initial_grace_steps",
    "reset_noise_scale",
    "reset_randomize_state",
    "reset_body_xy_noise",
    "reset_body_z_noise",
    "reset_roll_pitch_deg",
    "reset_yaw_deg",
    "reset_q1_noise_deg",
    "reset_q2_noise_deg",
    "reset_q3_noise_deg",
    "reset_step_count_max",
    "reset_episode_length_jitter_steps",
    "reset_grace_step_jitter_steps",
    "usar_filtros_contacto_recompensa",
    "action_scale",
    "verticalidad_reward_weight",
    "sigma_verticalidad_reward_deg",
    "cuerpo_paralelo_reward_weight",
    "support_gate_min",
    "max_survival_tilt_degrees",
    "altura_reward_weight_legacy",
    "altura_fuera_rango_penalty",
    "target_body_z",
    "target_height_min",
    "target_height_max",
    "target_height_sigma",
    "max_tilt_degrees",
    "inclinacion_excesiva_penalty",
    "inclinacion_excesiva_penalty_max",
    "terminate_on_excessive_tilt",
    "alineacion_q1_reward_weight",
    "q1_alignment_tolerance_deg",
    "contacto_pies_reward_weight",
    "centro_soporte_reward_weight",
    "support_center_margin",
    "plano_CoG_arana_target",
    "plano_CoG_arana_sigma",
    "min_foot_contacts_for_support",
    "valid_support_max_foot_clearance",
    "valid_support_min_foot_distance_xy",
    "max_total_foot_clearance",
    "foot_clearance_sigma",
    "pies_aire_penalty",
    "pies_aire_penalty_max",
    "terminate_on_excessive_foot_clearance",
    "contacto_suelo_invalido_penalty",
    "terminate_on_invalid_ground_contact",
    "terminate_on_persistent_invalid_ground_contact",
    "invalid_ground_contact_done_steps",
    "body_or_chassis_ground_contact_done_steps",
    "contacto_invalido_penalty_count_scale",
    "pasos_contacto_suelo_invalido_penalty",
    "efector_encima_q3_legacy_penalty",
    "efector_encima_q3_penalty_max",
    "effector_q3_z_margin",
    "effector_q3_z_scale",
    "q3_debajo_cuerpo_reward_weight",
    "q3_below_body_target_margin",
    "apertura_q3_reward_weight",
    "q3_opening_target_radius",
    "q3_opening_sigma",
    "q1_center_sigma_deg",
    "effector_opening_target_radius",
    "effector_opening_sigma",
    "q3_center_target_radius",
    "q3_center_sigma",
    "pose_arana_baja_reward_weight",
    "low_spider_target_body_z",
    "low_spider_body_z_sigma",
    "low_spider_target_q3_above_body",
    "low_spider_q3_sigma",
    "low_spider_foot_clearance_sigma",
    "low_spider_min_foot_distance_xy",
    "low_spider_support_sigma",
    "low_spider_tilt_sigma_deg",
    "low_spider_qvel_sigma",
    "low_spider_body_angvel_sigma",
    "low_spider_cambio_accion_sigma",
    "simetria_patas_reward_weight_legacy",
    "leg_symmetry_tolerance_deg",
    "control_cost_weight",
    "ctrl_cost_max",
    "velocidad_cost_weight",
    "velocity_cost_max",
    "cambio_accion_cost_weight",
    "cambio_accion_cost_max",
    "base_vertical_velocity_sigma",
    "base_angular_velocity_sigma",
    "joint_limit_margin_fraction",
    "qvel_sigma",
    "body_angvel_sigma",
    "cambio_accion_sigma",
    "tiempo_vivo_reward_weight",
    "tiempo_vivo_reward_growth",
    "tiempo_vivo_reward_max",
    "reward_clip_min",
    "reward_clip_max",
    # Sistema de postura XML (v2026051902).
    "pose_imitacion_reward_weight",
    "pose_imitacion_sigma",
    "pose_xml_final_reward_weight",
    "pose_xml_final_tilt_sigma_deg",
    "pose_xml_final_height_sigma",
    "usar_altura_referencia_xml",
    "usar_pose_referencia_xml",
    "include_pose_error",
    "vector_gravedad_sigma",
    "vector_gravedad_estabilidad_reward_weight",
    "cuerpo_paralelo_sigma",
    "altura_exceso_penalty_weight",
    "altura_exceso_tolerancia",
    "altura_exceso_sigma",
    "altura_baja_penalty_weight",
    "altura_baja_tolerancia",
    "altura_baja_sigma",
    "inclinacion_suave_penalty_weight",
    "inclinacion_suave_tolerancia_deg",
    "inclinacion_suave_sigma_deg",
    "apertura_efectores_xml_reward_weight",
    "apertura_efectores_xml_sigma",
    "q3_distancia_xml_reward_weight",
    "q3_distancia_xml_sigma",
    "velocidad_articular_cero_reward_weight",
    "velocidad_articular_cero_sigma",
    "rodilla_suelo_penalty_weight",
    "cuerpo_chasis_suelo_penalty_weight",
]


def _assert_linux_filesystem(path: Path) -> None:
  resolved = path.resolve()
  is_wsl = bool(os.environ.get("WSL_INTEROP")) or "microsoft" in os.uname().release.lower()
  if is_wsl and str(resolved).startswith("/mnt/"):
    raise RuntimeError(
        f"No ejecutes entrenamientos pesados en rutas montadas de Windows: {resolved}"
    )


def _assert_gpu() -> None:
  """Valida que JAX utilice el backend elegido para esta instalación."""
  print("Backend JAX:", jax.default_backend())
  print("Dispositivos JAX:", jax.devices())
  accelerator = os.environ.get("SIM2REAL_ACCELERATOR", "auto")
  expected = {
      "nvidia": "gpu",
      "amd": "gpu",
      "intel": "xpu",
      "cpu": "cpu",
  }.get(accelerator)
  if expected is None:
    raise RuntimeError(
        "SIM2REAL_ACCELERATOR debe estar resuelto antes de entrenar "
        "(nvidia, amd, intel o cpu)."
    )
  if jax.default_backend() != expected:
    raise RuntimeError(
        f"Backend JAX incorrecto para {accelerator}: "
        f"esperado {expected}, obtenido {jax.default_backend()}."
    )
  if accelerator == "cpu":
    print("AVISO: entrenamiento PPO en CPU permitido solo como prueba; sera muy lento.")


_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _safe_run_dir(base_logdir: Path, exp_name: str) -> Path:
  """Construye una ejecución hija directa e impide escapar del directorio."""
  if exp_name in {".", ".."} or not _RUN_NAME_RE.fullmatch(exp_name):
    raise ValueError(
        "run_name/suffix no válido: usa solo letras, números, punto, guion o "
        "guion bajo; no se admiten rutas"
    )
  candidate = base_logdir / exp_name
  if candidate.is_symlink():
    raise ValueError(f"La ejecución no puede ser un enlace simbólico: {candidate}")
  run_dir = candidate.resolve()
  if run_dir.parent != base_logdir:
    raise ValueError(f"La ejecución debe ser hija directa de {base_logdir}: {run_dir}")
  return run_dir


def _prepare_checkpoint_dir(run_dir: Path, reset_checkpoint: bool) -> Path:
  """Prepara checkpoints sin seguir enlaces simbolicos durante un reset."""
  ckpt_path = run_dir / "checkpoints"
  if ckpt_path.is_symlink():
    raise ValueError(f"La carpeta checkpoints no puede ser un enlace simbolico: {ckpt_path}")
  if reset_checkpoint and ckpt_path.exists():
    shutil.rmtree(ckpt_path)
  ckpt_path.mkdir(parents=True, exist_ok=True)
  return ckpt_path


_ACTIVACIONES_DISPONIBLES: dict[str, Any] = {
    "swish": nn.swish,
    "silu": nn.swish,   # SiLU y Swish son identicas en Flax
    "tanh": nn.tanh,
    "relu": nn.relu,
    "elu": nn.elu,
}


def _crear_fabrica_redes(ppo_config):
  """Crea la funcion de fabrica de redes PPO con la activacion configurada.

  Lee el campo network_factory.activacion_red para elegir la funcion de
  activacion. Si no existe, usa swish por defecto (recomendado para robots).
  """
  if not hasattr(ppo_config, "network_factory"):
    return ppo_networks.make_ppo_networks
  kwargs = dict(ppo_config.network_factory)
  nombre_activacion = kwargs.pop("activacion_red", "swish")
  activacion = _ACTIVACIONES_DISPONIBLES.get(nombre_activacion, nn.swish)
  kwargs["activation"] = activacion
  return functools.partial(ppo_networks.make_ppo_networks, **kwargs)


def _resolver_checkpoint(path_str: str | None) -> Path | None:
  if not path_str:
    return None
  candidate = Path(path_str).resolve()
  if not candidate.exists():
    raise FileNotFoundError(f"No existe el checkpoint o la ejecución: {candidate}")
  if candidate.is_dir() and (candidate / "checkpoints").is_dir():
    candidate = candidate / "checkpoints"
  if candidate.is_dir():
    children = [p for p in candidate.iterdir() if p.is_dir() and p.name.isdigit()]
    if children:
      return max(children, key=lambda p: p.stat().st_mtime)
  return candidate


def _ultimo_checkpoint(run_dir: Path) -> Path | None:
  ckpt_dir = run_dir / "checkpoints"
  if not ckpt_dir.is_dir():
    return None
  children = [p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()]
  if not children:
    return None
  return max(children, key=lambda p: p.stat().st_mtime)


def _guardar_json(path: Path, data) -> None:
  if path.is_symlink():
    raise ValueError(f"No se escribe JSON sobre un enlace simbolico: {path}")
  path.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary_name = tempfile.mkstemp(
      prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
  )
  temporary_path = Path(temporary_name)
  try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
      json.dump(data, temporary_file, indent=2, sort_keys=True)
      temporary_file.write("\n")
      temporary_file.flush()
      os.fsync(temporary_file.fileno())
    os.replace(temporary_path, path)
  finally:
    temporary_path.unlink(missing_ok=True)


def _guardar_comando(path: Path) -> None:
  env_keys = (
      "XLA_PYTHON_CLIENT_PREALLOCATE",
      "XLA_PYTHON_CLIENT_MEM_FRACTION",
      "XLA_PYTHON_CLIENT_ALLOCATOR",
      "TF_GPU_ALLOCATOR",
      "MUJOCO_GL",
      "JAX_DEFAULT_MATMUL_PRECISION",
      "PYTHONPATH",
  )
  lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
  for key in env_keys:
    value = os.environ.get(key)
    if value is not None:
      lines.append(f"export {key}={shlex.quote(value)}")
  lines.append(" ".join(shlex.quote(arg) for arg in sys.argv))
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  path.chmod(0o755)


def _guardar_estado(path: Path, estado: str, extra: dict | None = None) -> None:
  payload: dict[str, Any] = {}
  if path.is_file() and not path.is_symlink():
    try:
      previous = json.loads(path.read_text(encoding="utf-8"))
      if isinstance(previous, dict):
        payload.update(previous)
    except (OSError, ValueError, TypeError):
      pass
  if estado == "iniciando":
    for stale_key in (
        "error",
        "traceback",
        "senal",
        "motivo_cancelacion",
        "codigo_salida",
    ):
      payload.pop(stale_key, None)
  payload.update({
      "estado": estado,
      "pid": os.getpid(),
      "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
  })
  if extra:
    payload.update(extra)
  _guardar_json(path, payload)


class EntrenamientoCancelado(BaseException):
  """Interrumpe PPO de forma distinguible de un fallo del entrenamiento."""

  def __init__(self, motivo: str, signal_number: int | None = None):
    super().__init__(motivo)
    self.motivo = motivo
    self.signal_number = signal_number


def _parada_solicitada(run_dir: Path) -> bool:
  return any(
      (run_dir / name).exists()
      for name in ("parada_entrenamiento_solicitada", "parada_total_solicitada")
  )


def _informacion_checkpoint_conservado(
    run_dir: Path, ultimo_paso_confirmado: int | None
) -> dict[str, Any]:
  checkpoint = _ultimo_checkpoint(run_dir)
  if checkpoint is None:
    return {
        "ultimo_checkpoint_conservado": None,
        "paso_checkpoint_conservado": None,
        "checkpoint_final_seguro": False,
    }
  try:
    checkpoint_step = int(checkpoint.name)
  except ValueError:
    checkpoint_step = None
  return {
      "ultimo_checkpoint_conservado": checkpoint.as_posix(),
      "paso_checkpoint_conservado": checkpoint_step,
      # Brax guarda el checkpoint justo antes de llamar a progress_fn. Solo se
      # afirma que es final si ambos pasos coinciden; nunca se inventa un guardado
      # desde un manejador de senal mientras XLA puede estar ejecutando.
      "checkpoint_final_seguro": (
          checkpoint_step is not None
          and ultimo_paso_confirmado is not None
          and checkpoint_step == ultimo_paso_confirmado
      ),
  }


def _metric_value(
    metrics,
    keys: tuple[str, ...],
    episode_length_scale: float | None = None,
) -> str:
  for key in keys:
    if key in metrics and metrics[key] is not None:
      try:
        value = float(metrics[key])
        if not math.isfinite(value):
          return ""
        if episode_length_scale is not None and episode_length_scale > 0:
          value /= episode_length_scale
        return f"{value:.6f}"
      except (TypeError, ValueError):
        return ""
  return ""


def _field_to_env_metric_name(field: str) -> str | None:
  """Convierte una columna del CSV al nombre canonico exportado por el entorno."""

  field_to_metric = {
      "supervivencia_reward_ponderado": "supervivencia_reward_ponderado",
      "vector_gravedad_base_reward_ponderado": "vector_gravedad_base_reward_ponderado",
    "vector_gravedad_reward_ponderado": "vector_gravedad_reward_ponderado",
      "vector_gravedad_estabilidad_reward_ponderado": (
          "vector_gravedad_estabilidad_reward_ponderado"
      ),
      "cuerpo_paralelo_reward_ponderado": "cuerpo_paralelo_reward_ponderado",
      "poligono_CoG_reward_ponderado": "poligono_CoG_reward_ponderado",
      "altura_reward_ponderado": "altura_reward_ponderado",
      "apertura_efectores_xml_reward_ponderado": "apertura_efectores_xml_reward_ponderado",
      "q3_distancia_xml_reward_ponderado": "q3_distancia_xml_reward_ponderado",
      "plano_CoG_arana_reward_ponderado": "plano_CoG_arana_reward_ponderado",
      "contactos_suelo_reward_ponderado": "contactos_suelo_reward_ponderado",
      "contactos_suelo_reward_filtrado_ponderado": "contactos_suelo_reward_filtrado_ponderado",
      "soporte_estatico_reward_ponderado": "soporte_estatico_reward_ponderado",
      "soporte_estatico_reward_filtrado_ponderado": "soporte_estatico_reward_filtrado_ponderado",
      "apertura_efectores_q3_cerca_reward_ponderado": (
          "apertura_efectores_q3_cerca_reward_ponderado"
      ),
      "q1_centrados_reward_ponderado": "q1_centrados_reward_ponderado",
      "q3_separados_centro_reward_ponderado": "q3_separados_centro_reward_ponderado",
      "simetria_patas_reward_ponderado": "simetria_patas_reward_ponderado",
      "velocidad_lineal_cero_reward_ponderado": "velocidad_lineal_cero_reward_ponderado",
      "velocidad_angular_cero_reward_ponderado": "velocidad_angular_cero_reward_ponderado",
      "velocidad_articular_cero_reward_ponderado": "velocidad_articular_cero_reward_ponderado",
      "positive_reward": "positive_reward",
      "contacto_invalido_penalty_ponderado": "contacto_invalido_penalty_ponderado",
      "contacto_invalido_persistente_penalty_ponderado": (
          "contacto_invalido_persistente_penalty_ponderado"
      ),
      "efector_encima_q3_penalty_ponderado": "efector_encima_q3_penalty_ponderado",
      "altura_exceso_penalty_ponderado": "altura_exceso_penalty_ponderado",
      "altura_baja_penalty_ponderado": "altura_baja_penalty_ponderado",
      "rodilla_suelo_penalty_ponderado": "rodilla_suelo_penalty_ponderado",
      "cuerpo_chasis_suelo_penalty_ponderado": "cuerpo_chasis_suelo_penalty_ponderado",
      "inclinacion_suave_penalty_ponderado": "inclinacion_suave_penalty_ponderado",
      "velocidad_vertical_cuerpo_penalty_ponderado": "velocidad_vertical_cuerpo_penalty_ponderado",
      "velocidad_angular_cuerpo_penalty_ponderado": "velocidad_angular_cuerpo_penalty_ponderado",
      "control_penalty_ponderado": "control_penalty_ponderado",
      "cambio_accion_penalty_ponderado": "cambio_accion_penalty_ponderado",
      "limite_articular_penalty_ponderado": "limite_articular_penalty_ponderado",
      "penalties": "penalties",
      "reward_total": "reward_total",
  }
  return field_to_metric.get(field)


def _metric_key_candidates(field: str, keys: tuple[str, ...]) -> tuple[str, ...]:
  candidates = list(keys)
  env_metric_name = _field_to_env_metric_name(field)
  if env_metric_name is not None:
    candidates.extend(
        (
            env_metric_name,
            f"episode/{env_metric_name}",
            f"eval/episode_{env_metric_name}",
        )
    )
  return tuple(dict.fromkeys(candidates))


def _metric_needs_episode_average(field: str) -> bool:
  return field.startswith("state_")


def _row_has_any_value(row: dict[str, Any], fields: list[str]) -> bool:
  return any(row.get(field) not in (None, "") for field in fields)


def _guardar_config_recompensas(path: Path, env_config) -> None:
  fieldnames = ["debug_version_recompensa", *env_config.keys()]
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(
        {
            "debug_version_recompensa": VERSION_DIAGNOSTICO_RECOMPENSA,
            **{key: env_config.get(key, "") for key in env_config.keys()},
        }
    )


def _guardar_ultima_run(logdir: Path, run_dir: Path) -> None:
  pointer = logdir / "ultima_ejecucion.txt"
  if pointer.is_symlink():
    raise RuntimeError(f"ultima_ejecucion.txt no puede ser un enlace simbolico: {pointer}")
  temporary = logdir / f".ultima_run.{os.getpid()}.tmp"
  temporary.write_text(run_dir.as_posix() + "\n", encoding="utf-8")
  os.replace(temporary, pointer)


def _abrir_csv_log(
    path: Path,
    fieldnames: list[str],
    append: bool,
) -> tuple[Any, csv.DictWriter]:
  """Abre un CSV de entrenamiento y separa las ejecuciones por defecto."""

  if path.exists() and path.stat().st_size > 0:
    with path.open(newline="", encoding="utf-8") as existing_file:
      existing_header = next(csv.reader(existing_file), [])
    if append and existing_header == fieldnames:
      file = path.open("a", newline="", encoding="utf-8")
      return file, csv.DictWriter(file, fieldnames=fieldnames)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if existing_header != fieldnames:
      timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
      backup_path = path.with_name(
          f"{path.stem}.schema_mismatch_{timestamp}{path.suffix}"
      )
      reason = "cabecera antigua"
    else:
      backup_path = path.with_name(f"{path.stem}.previous_{timestamp}{path.suffix}")
      reason = "ejecución anterior"
    path.replace(backup_path)
    print(f"CSV de {reason} respaldado en: {backup_path}", flush=True)

  file = path.open("w", newline="", encoding="utf-8")
  writer = csv.DictWriter(file, fieldnames=fieldnames)
  writer.writeheader()
  file.flush()
  return file, writer


def _calcular_training_metrics_steps(ppo_config) -> int | None:
  if not ppo_config.get("log_training_metrics", False):
    return ppo_config.get("training_metrics_steps")
  if ppo_config.get("training_metrics_steps") is not None:
    return int(ppo_config.training_metrics_steps)

  interval = int(ppo_config.get("reward_log_policy_updates_interval", 5))
  updates_per_training_step = max(1, int(ppo_config.num_updates_per_batch))
  training_steps_between_logs = max(
      1, math.ceil(interval / updates_per_training_step)
  )
  env_steps_per_training_step = (
      int(ppo_config.batch_size)
      * int(ppo_config.unroll_length)
      * int(ppo_config.num_minibatches)
      * int(ppo_config.action_repeat)
  )
  return int(env_steps_per_training_step * training_steps_between_logs)


def _wrap_env_for_training_full_reset(
    env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn=None,
):
    return wrapper.wrap_for_brax_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=randomization_fn,
        full_reset=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_timesteps", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_evals", type=int, default=None)
    parser.add_argument("--episode_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--unroll_length", type=int, default=None)
    parser.add_argument("--num_minibatches", type=int, default=None)
    parser.add_argument("--num_updates_per_batch", type=int, default=None)
    parser.add_argument("--num_eval_envs", type=int, default=None)
    parser.add_argument(
        "--perfil-ppo",
        dest="perfil_ppo",
        choices=["depuracion", "ligero", "ligero_rapido", "completo"],
        default="ligero",
        help=(
            "Perfil de hiperparametros PPO: depuracion, ligero, "
            "ligero_rapido o completo."
        ),
    )
    parser.add_argument(
        "--intervalo-log-recompensas",
        type=int,
        default=None,
        dest="reward_log_policy_updates_interval",
    )
    parser.add_argument("--training_metrics_steps", type=int, default=None)
    parser.add_argument(
        "--sin-metricas-recompensa-entrenamiento",
        action="store_true",
        dest="no_training_reward_metrics",
    )
    parser.add_argument(
        "--metricas-recompensa-entrenamiento",
        action="store_true",
        dest="training_reward_metrics",
        help="Activa metricas de recompensa durante trayectorias de entrenamiento. Es mas lento.",
    )
    parser.add_argument(
        "--metricas-fisicas-completas",
        action="store_true",
        dest="full_physical_metrics",
        help="Calcula el bloque completo de metricas fisicas por paso. Usalo para depurar.",
    )
    parser.add_argument(
        "--curriculo-penalizaciones",
        dest="curriculo_penalizaciones",
        type=float,
        default=None,
        help="Valor fijo k_c para escalar penalizaciones en esta ejecucion, entre 0 y 1.",
    )
    parser.add_argument(
        "--fase-recompensa",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        dest="fase_recompensa",
      help="Fase del curriculo de recompensa: 0 historica, 1 mantener pose, 2 levantarse, 3 recuperarse de caidas.",
    )
    parser.add_argument("--impl", choices=["jax"], default="jax")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", default="logs_sim2real_mjx")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--suffix", default=None)
    parser.add_argument("--load_checkpoint_path", default=None)
    parser.add_argument(
        "--reset_checkpoint",
        action="store_true",
        help=(
            "Arranca desde cero aunque exista checkpoints/ en la run. "
            "Si se usa con --run_name fijo, borra solo los checkpoints de esa run."
        ),
    )
    parser.add_argument(
        "--append_csv",
        action="store_true",
        help=(
            "Anade filas a los CSV existentes si la cabecera coincide. "
            "Por defecto cada lanzamiento respalda los CSV previos y empieza limpio."
        ),
    )
    args = parser.parse_args()

    _assert_gpu()
    base_logdir = Path(args.logdir).resolve()
    _assert_linux_filesystem(base_logdir)
    base_logdir.mkdir(parents=True, exist_ok=True)

    now = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = args.run_name or f"EntornoRobotMJX-{now}"
    if args.suffix and not args.run_name:
        exp_name += f"-{args.suffix}"

    run_dir = _safe_run_dir(base_logdir, exp_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    _guardar_ultima_run(base_logdir, run_dir)

    pid_path = run_dir / "entrenamiento.pid"
    estado_path = run_dir / "estado.json"
    pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    estado_terminal = {"escrito": False}
    progreso_confirmado: dict[str, int | None] = {"pasos": None}
    previous_excepthook = sys.excepthook
    previous_signal_handlers: dict[int, Any] = {}

    def limpiar_pid_propio() -> None:
        try:
            if pid_path.is_file() and not pid_path.is_symlink():
                if pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                    pid_path.unlink()
        except OSError:
            pass

    def datos_cancelacion(
        motivo: str, signal_number: int | None = None
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "motivo_cancelacion": motivo,
            "ultimo_paso_confirmado": progreso_confirmado["pasos"],
        }
        if signal_number is not None:
            data["senal"] = signal.Signals(signal_number).name
            data["codigo_salida"] = 128 + signal_number
        data.update(
            _informacion_checkpoint_conservado(
                run_dir, progreso_confirmado["pasos"]
            )
        )
        return data

    def manejador_senal(signal_number, _frame) -> None:
        signal_name = signal.Signals(signal_number).name
        motivo = f"cancelacion solicitada mediante {signal_name}"
        _guardar_estado(
            estado_path,
            "cancelando",
            datos_cancelacion(motivo, signal_number),
        )
        raise EntrenamientoCancelado(motivo, signal_number)

    def manejador_excepcion(exc_type, exc_value, exc_traceback) -> None:
        if not estado_terminal["escrito"]:
            if issubclass(exc_type, (EntrenamientoCancelado, KeyboardInterrupt)):
                motivo = getattr(exc_value, "motivo", "interrupcion de teclado")
                signal_number = getattr(exc_value, "signal_number", None)
                _guardar_estado(
                    estado_path,
                    "cancelado",
                    datos_cancelacion(motivo, signal_number),
                )
                estado_terminal["escrito"] = True
            else:
                _guardar_estado(
                    estado_path,
                    "error",
                    {
                        "error": repr(exc_value),
                        "traceback": "".join(
                            traceback.format_exception(
                                exc_type, exc_value, exc_traceback, limit=12
                            )
                        ),
                        "ultimo_paso_confirmado": progreso_confirmado["pasos"],
                        **_informacion_checkpoint_conservado(
                            run_dir, progreso_confirmado["pasos"]
                        ),
                    },
                )
                estado_terminal["escrito"] = True
        limpiar_pid_propio()
        if not issubclass(exc_type, (EntrenamientoCancelado, KeyboardInterrupt)):
            previous_excepthook(exc_type, exc_value, exc_traceback)

    def finalizar_proceso() -> None:
        if not estado_terminal["escrito"]:
            _guardar_estado(
                estado_path,
                "error",
                {
                    "error": "finalizacion inesperada sin estado terminal",
                    "ultimo_paso_confirmado": progreso_confirmado["pasos"],
                    **_informacion_checkpoint_conservado(
                        run_dir, progreso_confirmado["pasos"]
                    ),
                },
            )
            estado_terminal["escrito"] = True
        limpiar_pid_propio()

    _guardar_estado(
        estado_path,
        "iniciando",
        {
            "run_dir": run_dir.as_posix(),
            "perfil_ppo": args.perfil_ppo,
            "fase_curriculum_recompensa": int(args.fase_recompensa),
        },
    )
    atexit.register(finalizar_proceso)
    sys.excepthook = manejador_excepcion
    for signal_number in (signal.SIGTERM, signal.SIGINT):
        previous_signal_handlers[signal_number] = signal.getsignal(signal_number)
        signal.signal(signal_number, manejador_senal)
    # ``nohup`` marca SIGHUP como ignorada antes de arrancar Python. No debemos
    # deshacer esa protección: en WSL la consola que lanzó el proceso termina
    # justo después de devolver el PID y envía SIGHUP a su grupo. En primer
    # plano sí instalamos el cierre controlado porque se conserva SIG_DFL.
    if signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:
        previous_signal_handlers[signal.SIGHUP] = signal.getsignal(signal.SIGHUP)
        signal.signal(signal.SIGHUP, manejador_senal)
    _guardar_comando(run_dir / "comando.sh")

    if _parada_solicitada(run_dir):
        raise EntrenamientoCancelado(
            "la parada fue solicitada durante el arranque del entrenamiento"
        )

    env_config = default_config()
    resumen_curriculum = aplicar_fase_curriculo_recompensa(
        env_config, args.fase_recompensa
    )
    env_config.impl = args.impl
    if args.perfil_ppo == "depuracion":
        ppo_config = configuracion_ppo_depuracion()
    elif args.perfil_ppo == "ligero":
        ppo_config = configuracion_ppo_ligera()
    elif args.perfil_ppo == "ligero_rapido":
        ppo_config = configuracion_ppo_ligera_rapida()
    elif args.perfil_ppo == "completo":
        ppo_config = configuracion_ppo_completa()
    else:
        raise ValueError(f"Perfil PPO no reconocido: {args.perfil_ppo}")

    overrides = {
        "num_timesteps": args.num_timesteps,
        "num_envs": args.num_envs,
        "num_evals": args.num_evals,
        "episode_length": args.episode_length,
        "batch_size": args.batch_size,
        "unroll_length": args.unroll_length,
        "num_minibatches": args.num_minibatches,
        "num_updates_per_batch": args.num_updates_per_batch,
        "num_eval_envs": args.num_eval_envs,
        "reward_log_policy_updates_interval": args.reward_log_policy_updates_interval,
        "training_metrics_steps": args.training_metrics_steps,
    }
    for key, value in overrides.items():
        if value is not None:
            ppo_config[key] = value
    env_config.episode_length = int(ppo_config.episode_length)
    env_config.action_repeat = int(ppo_config.action_repeat)
    if args.curriculo_penalizaciones is not None:
        env_config.curriculum_penalizaciones = max(
            0.0, min(float(args.curriculo_penalizaciones), 1.0)
        )
    env_config.log_full_metrics = bool(args.full_physical_metrics)
    if args.training_reward_metrics:
        ppo_config.log_training_metrics = True
    if args.no_training_reward_metrics:
        ppo_config.log_training_metrics = False
    ppo_config.training_metrics_steps = _calcular_training_metrics_steps(ppo_config)
    batch_size = int(ppo_config.batch_size)
    num_minibatches = int(ppo_config.num_minibatches)
    num_envs = int(ppo_config.num_envs)
    if batch_size % num_minibatches != 0:
        raise ValueError(
            "batch_size debe ser divisible por num_minibatches para que PPO "
            "actualice por trozos equilibrados."
        )
    rollout_batch = batch_size * num_minibatches
    if rollout_batch % num_envs != 0:
        raise ValueError(
            "Configuracion PPO incompatible: batch_size * num_minibatches "
            f"= {rollout_batch} debe ser divisible por num_envs={num_envs}. "
            "Prueba a ajustar --num_envs, --batch_size o --num_minibatches."
        )

    env = EntornoRobotMJX(config=env_config)
    eval_env = EntornoRobotMJX(config=env_config)
    env_config_dict = configuracion_a_diccionario(env_config)
    if hasattr(env, "_ref_body_z"):
        env_config_dict["target_body_z_referencia"] = float(env._ref_body_z)
    _guardar_json(run_dir / "config_entorno.json", env_config_dict)
    _guardar_json(
        run_dir / "hiperparametros.json", configuracion_a_diccionario(ppo_config)
    )
    _guardar_json(run_dir / "config_curriculum_recompensa.json", resumen_curriculum)
    _guardar_config_recompensas(run_dir / "config_recompensas.csv", env_config)

    print("Ejecución:", run_dir)
    print("XML entrenamiento:", env.xml_path)
    print(
        "Entorno Python:",
        Path(sys.modules[EntornoRobotMJX.__module__].__file__).resolve(),
    )
    print("Version de diagnostico de recompensa:", VERSION_DIAGNOSTICO_RECOMPENSA)
    print("Perfil PPO:", args.perfil_ppo)
    print(
        "Fase recompensa:",
        f"{env_config.fase_curriculum_recompensa} - {env_config.nombre_curriculum_recompensa}",
    )
    print("Observaciones:", env.observation_size)
    print("Acciones:", env.action_size)
    print("PID:", os.getpid())
    print("PPO:", ppo_config)
    _guardar_estado(
        estado_path,
        "entrenando",
        {
            "run_dir": run_dir.as_posix(),
            "perfil_ppo": args.perfil_ppo,
            "num_envs": int(ppo_config.num_envs),
            "num_timesteps": int(ppo_config.num_timesteps),
            "batch_size": int(ppo_config.batch_size),
            "unroll_length": int(ppo_config.unroll_length),
            "episode_length": int(ppo_config.episode_length),
            "log_full_metrics": bool(env_config.log_full_metrics),
            "curriculum_penalizaciones": float(env_config.curriculum_penalizaciones),
            "fase_curriculum_recompensa": int(env_config.fase_curriculum_recompensa),
            "nombre_curriculum_recompensa": str(env_config.nombre_curriculum_recompensa),
        },
    )

    if args.reset_checkpoint and args.load_checkpoint_path:
        raise ValueError("No uses --reset_checkpoint junto con --load_checkpoint_path.")
    ckpt_path = _prepare_checkpoint_dir(run_dir, args.reset_checkpoint)
    restore_checkpoint_path = None
    if not args.reset_checkpoint:
        restore_checkpoint_path = _resolver_checkpoint(args.load_checkpoint_path)
    if restore_checkpoint_path is None and args.run_name and not args.reset_checkpoint:
        restore_checkpoint_path = _ultimo_checkpoint(run_dir)
    if restore_checkpoint_path is not None:
        print("Restaurando:", restore_checkpoint_path)

    training_params = dict(ppo_config)
    network_factory = _crear_fabrica_redes(ppo_config)
    training_params.pop("network_factory", None)
    training_params.pop("reward_log_policy_updates_interval", None)
    num_eval_envs = training_params.pop("num_eval_envs", 2)

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        seed=args.seed,
        restore_checkpoint_path=restore_checkpoint_path,
        save_checkpoint_path=ckpt_path,
        wrap_env_fn=_wrap_env_for_training_full_reset,
        num_eval_envs=num_eval_envs,
        vision=False,
    )

    progress_path = run_dir / "progreso.csv"
    ruta_recompensas = run_dir / "recompensas.csv"
    physical_data_path = run_dir / "DATOS_fisicos_CSV.csv"
    start = time.monotonic()
    last_progress_sample: dict[str, float] = {
        "num_steps": 0.0,
        "elapsed_seconds": 0.0,
    }
    progress_fields = [
        "num_steps",
        "elapsed_seconds",
        "elapsed_hours",
        "percent",
        "source",
        "eval_episode_reward",
        "eval_episode_length",
        "eval_reward_per_step",
        "wall_steps_per_second",
        "steps_per_second",
        "training_steps_per_second",
    ]
    progress_file, writer = _abrir_csv_log(
        progress_path, progress_fields, append=args.append_csv
    )
    archivo_recompensas, escritor_recompensas = _abrir_csv_log(
        ruta_recompensas, CAMPOS_HISTORIAL_RECOMPENSAS, append=args.append_csv
    )
    physical_file, physical_writer = _abrir_csv_log(
        physical_data_path, PHYSICAL_DATA_FIELDS, append=args.append_csv
    )
    try:

        def progreso(num_steps, metrics):
            if _parada_solicitada(run_dir):
                raise EntrenamientoCancelado(
                    "parada solicitada en un punto seguro entre evaluaciones"
                )
            elapsed = time.monotonic() - start
            elapsed_hours = elapsed / 3600.0
            eval_episode_reward = metrics.get(
                "eval/episode_reward", metrics.get("episode/sum_reward")
            )
            length = metrics.get("eval/avg_episode_length", metrics.get("episode/length"))
            eval_reward_per_step = None
            if eval_episode_reward is not None and length is not None and float(length) > 0:
                eval_reward_per_step = float(eval_episode_reward) / float(length)
            episode_length_scale = None
            if length is not None and float(length) > 0:
                episode_length_scale = float(length)
            sps = metrics.get("training/sps", metrics.get("episode/sps"))
            source = "eval" if "eval/episode_reward" in metrics else "train"
            percent = 100.0 * num_steps / max(1, ppo_config.num_timesteps)
            step_delta = float(num_steps) - last_progress_sample["num_steps"]
            elapsed_delta = elapsed - last_progress_sample["elapsed_seconds"]
            wall_steps_per_second = None
            if step_delta > 0 and elapsed_delta > 0:
                wall_steps_per_second = step_delta / elapsed_delta
                last_progress_sample["num_steps"] = float(num_steps)
                last_progress_sample["elapsed_seconds"] = elapsed
            training_steps_per_second = None if sps is None else float(sps)
            steps_per_second = (
                training_steps_per_second
                if training_steps_per_second is not None
                else wall_steps_per_second
            )
            writer.writerow({
                "num_steps": int(num_steps),
                "elapsed_seconds": f"{elapsed:.3f}",
                "elapsed_hours": f"{elapsed_hours:.6f}",
                "percent": f"{percent:.3f}",
                "source": source,
                "eval_episode_reward": (
                    "" if eval_episode_reward is None else f"{float(eval_episode_reward):.6f}"
                ),
                "eval_episode_length": "" if length is None else f"{float(length):.3f}",
                "eval_reward_per_step": (
                    "" if eval_reward_per_step is None else f"{eval_reward_per_step:.6f}"
                ),
                "wall_steps_per_second": (
                    ""
                    if wall_steps_per_second is None
                    else f"{wall_steps_per_second:.3f}"
                ),
                "steps_per_second": (
                    "" if steps_per_second is None else f"{steps_per_second:.3f}"
                ),
                "training_steps_per_second": (
                    ""
                    if training_steps_per_second is None
                    else f"{training_steps_per_second:.3f}"
                ),
            })
            fila_recompensas = {
                "num_steps": int(num_steps),
                "elapsed_seconds": f"{elapsed:.3f}",
                "elapsed_hours": f"{elapsed_hours:.6f}",
                "percent": f"{percent:.3f}",
                "source": source,
                "eval_episode_reward": (
                    "" if eval_episode_reward is None else f"{float(eval_episode_reward):.6f}"
                ),
                "eval_episode_length": "" if length is None else f"{float(length):.3f}",
                "eval_reward_per_step": (
                    "" if eval_reward_per_step is None else f"{eval_reward_per_step:.6f}"
                ),
                "wall_steps_per_second": (
                    ""
                    if wall_steps_per_second is None
                    else f"{wall_steps_per_second:.3f}"
                ),
                "steps_per_second": (
                    "" if steps_per_second is None else f"{steps_per_second:.3f}"
                ),
                "training_steps_per_second": (
                    ""
                    if training_steps_per_second is None
                    else f"{training_steps_per_second:.3f}"
                ),
            }
            for field, metric_keys in MAPA_METRICAS_RECOMPENSA.items():
                scale = (
                    episode_length_scale
                    if _metric_needs_episode_average(field)
                    else None
                )
                fila_recompensas[field] = _metric_value(
                    metrics,
                    _metric_key_candidates(field, metric_keys),
                    scale,
                )
            positive_reward_value = fila_recompensas.get("positive_reward")
            penalties_value = fila_recompensas.get("penalties")
            if fila_recompensas.get("reward_total") in (None, ""):
                if positive_reward_value not in (None, "") and penalties_value not in (None, ""):
                    fila_recompensas["reward_total"] = (
                        f"{float(positive_reward_value) - float(penalties_value):.6f}"
                    )
                else:
                    fila_recompensas["reward_total"] = ""
            fila_recompensas["debug_version"] = str(VERSION_DIAGNOSTICO_RECOMPENSA)
            fila_recompensas["debug_episode_length"] = str(int(env_config.episode_length))
            fila_recompensas["debug_action_scale"] = f"{float(env_config.action_scale):.6f}"
            fila_recompensas["debug_reward_scale"] = f"{float(env_config.reward_scale):.6f}"
            fila_recompensas["debug_reward_clip_max"] = (
                f"{float(env_config.reward_clip_max):.6f}"
            )
            fila_recompensas["fase_curriculum_recompensa"] = str(
                int(env_config.fase_curriculum_recompensa)
            )
            fila_recompensas["nombre_curriculum_recompensa"] = str(
                env_config.nombre_curriculum_recompensa
            )
            if _row_has_any_value(
                fila_recompensas,
                [
                    "eval_episode_reward",
                    *[
                        field
                        for field in MAPA_METRICAS_RECOMPENSA
                        if not field.startswith("debug_")
                    ],
                ],
            ):
                escritor_recompensas.writerow(fila_recompensas)
            physical_row = {
                "num_steps": int(num_steps),
                "elapsed_seconds": f"{elapsed:.3f}",
                "elapsed_hours": f"{elapsed_hours:.6f}",
                "percent": f"{percent:.3f}",
                "source": source,
                "eval_episode_reward": (
                    "" if eval_episode_reward is None else f"{float(eval_episode_reward):.6f}"
                ),
                "eval_episode_length": "" if length is None else f"{float(length):.3f}",
                "eval_reward_per_step": (
                    "" if eval_reward_per_step is None else f"{eval_reward_per_step:.6f}"
                ),
                "wall_steps_per_second": (
                    ""
                    if wall_steps_per_second is None
                    else f"{wall_steps_per_second:.3f}"
                ),
                "steps_per_second": (
                    "" if steps_per_second is None else f"{steps_per_second:.3f}"
                ),
                "training_steps_per_second": (
                    ""
                    if training_steps_per_second is None
                    else f"{training_steps_per_second:.3f}"
                ),
            }
            for field, metric_keys in PHYSICAL_METRIC_MAP.items():
                physical_row[field] = _metric_value(
                    metrics,
                    _metric_key_candidates(field, metric_keys),
                )
            if _row_has_any_value(physical_row, list(PHYSICAL_METRIC_MAP.keys())):
                physical_writer.writerow(physical_row)
            progress_file.flush()
            archivo_recompensas.flush()
            physical_file.flush()
            progreso_confirmado["pasos"] = int(num_steps)
            _guardar_estado(
                estado_path,
                "entrenando",
                {
                    "ultimo_paso_confirmado": int(num_steps),
                    **_informacion_checkpoint_conservado(run_dir, int(num_steps)),
                },
            )
            if eval_episode_reward is not None:
                print(
                    f"{num_steps}/{ppo_config.num_timesteps} ({percent:.1f}%): "
                    f"recompensa_eval={float(eval_episode_reward):.3f}",
                    flush=True,
                )

        cancelacion: EntrenamientoCancelado | KeyboardInterrupt | None = None
        try:
            train_fn(environment=env, progress_fn=progreso, eval_env=eval_env)
        except (EntrenamientoCancelado, KeyboardInterrupt) as exc:
            cancelacion = exc
            motivo = getattr(exc, "motivo", "interrupcion de teclado")
            signal_number = getattr(exc, "signal_number", None)
            _guardar_estado(
                estado_path,
                "cancelado",
                datos_cancelacion(motivo, signal_number),
            )
            estado_terminal["escrito"] = True
        except BaseException as exc:
            _guardar_estado(
                estado_path,
                "error",
                {
                    "error": repr(exc),
                    "traceback": traceback.format_exc(limit=12),
                    "ultimo_paso_confirmado": progreso_confirmado["pasos"],
                    **_informacion_checkpoint_conservado(
                        run_dir, progreso_confirmado["pasos"]
                    ),
                },
            )
            estado_terminal["escrito"] = True
            raise
    finally:
        progress_file.close()
        archivo_recompensas.close()
        physical_file.close()

    if cancelacion is not None:
        limpiar_pid_propio()
        print(f"Entrenamiento cancelado: {run_dir}", flush=True)
        raise SystemExit(130)

    _guardar_estado(
        estado_path,
        "terminado",
        {
            "ultimo_paso_confirmado": progreso_confirmado["pasos"],
            **_informacion_checkpoint_conservado(
                run_dir, progreso_confirmado["pasos"]
            ),
        },
    )
    estado_terminal["escrito"] = True
    limpiar_pid_propio()
    print("Entrenamiento terminado:", run_dir)


if __name__ == "__main__":
  main()
