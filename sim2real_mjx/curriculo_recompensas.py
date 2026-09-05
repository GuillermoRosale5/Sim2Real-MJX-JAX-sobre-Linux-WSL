"""Fases de entrenamiento del robot.

Aqui no entrenamos nada directamente.
Este archivo solo decide qué recompensa, qué reinicio y qué castigos se usan en
cada fase.

La logica de las 3 fases:

1. Fase 1 – mantener_pose_xml
   El robot empieza muy cerca de la pose ideal del XML y aprende a quedarse
   ahi. La recompensa principal es pose_imitacion (imitacion de qpos0 del XML).
   Contacto y soporte no dominan. Penalizaciones de esfuerzo casi apagadas.

2. Fase 2 – llegar_a_pose_xml_desde_suelo
   El robot empieza mas lejos de la pose de referencia (suelo variado).
   Debe llegar a qpos0 del XML ideal. pose_imitacion sigue dominando, pero
   se suman altura, orientacion, contacto y soporte. Penalizaciones moderadas.

3. Fase 3 – recuperar_pose_xml_desde_caida
   El robot empieza desde posiciones extremas (caida variada). Debe volver
   a la pose del XML y quedar estable. Sigma mas estricto, penalizaciones
   mas fuertes.

REGLA CLAVE: La pose objetivo siempre es qpos0 del XML activo.
No hay angulos fijados a mano en este archivo. Si cambias el XML, la recompensa
se adapta sola al reconstruir el entorno.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml_collections import config_dict


INTERACCIONES_MINIMAS_PARA_SUBIR_DE_FASE = 5_000_000

_XML_POSE_IDEAL = (
    Path(__file__).parent / "xmls" / "ROBOT_POSE_IDEAL.xml"
).as_posix()


# Apagamos todo primero, luego cada fase enciende solo lo que interesa.
# Incluimos los campos nuevos del sistema de recompensa XML para que no queden
# encendidos accidentalmente si una fase anterior los activo.
PESOS_RECOMPENSA_APAGADOS: dict[str, float] = {
    # --- Recompensas positivas ---
    "pose_imitacion_reward_weight": 0.0,
    "pose_xml_final_reward_weight": 0.0,
    "supervivencia_reward_weight": 0.0,
    "vector_gravedad_reward_weight": 0.0,
    "vector_gravedad_base_reward_weight": 0.0,
    "vector_gravedad_estabilidad_reward_weight": 0.0,
    "cuerpo_paralelo_reward_weight": 0.0,
    "altura_reward_weight": 0.0,
    "altura_exceso_penalty_weight": 0.0,
    "altura_baja_penalty_weight": 0.0,
    "contactos_suelo_reward_weight": 0.0,
    "soporte_estatico_reward_weight": 0.0,
    "apertura_efectores_xml_reward_weight": 0.0,
    "q3_distancia_xml_reward_weight": 0.0,
    "velocidad_lineal_cero_reward_weight": 0.0,
    "velocidad_angular_cero_reward_weight": 0.0,
    "velocidad_articular_cero_reward_weight": 0.0,
    # --- Campos antiguos con peso 0, mantenidos por compatibilidad externa ---
    "pose_articular_reward_weight": 0.0,
    "q1_centrados_reward_weight": 0.0,
    "q3_separados_centro_reward_weight": 0.0,
    "apertura_efectores_q3_cerca_reward_weight": 0.0,
    "plano_CoG_arana_reward_weight": 0.0,
    "poligono_CoG_reward_weight": 0.0,
    "simetria_patas_reward_weight": 0.0,
    # --- Penalizaciones ---
    "contacto_invalido_penalty_weight": 0.0,
    "contacto_invalido_persistente_penalty_weight": 0.0,
    "rodilla_suelo_penalty_weight": 0.0,
    "cuerpo_chasis_suelo_penalty_weight": 0.0,
    "inclinacion_suave_penalty_weight": 0.0,
    "efector_encima_q3_penalty_weight": 0.0,
    "velocidad_vertical_cuerpo_penalty_weight": 0.0,
    "velocidad_angular_cuerpo_penalty_weight": 0.0,
    "control_penalty_weight": 0.0,
    "cambio_accion_penalty_weight": 0.0,
    "limite_articular_penalty_weight": 0.0,
    # --- Currículo de penalizaciones ---
    "curriculum_penalizaciones": 0.0,
}


FASES_RECOMPENSA: dict[int, dict[str, Any]] = {
    # ------------------------------------------------------------------
    # FASE 0: configuración base sin currículo. Útil para depuración y
    # para compatibilidad con scripts que esperan fase 0.
    # ------------------------------------------------------------------
    0: {
        "nombre": "base_actual",
        "descripcion": "La configuracion actual del entorno, sin curriculo.",
        "ajustes": {},
    },

    # ------------------------------------------------------------------
    # FASE 1: aprender a MANTENER la pose ideal del XML.
    #
    # El robot empieza muy cerca de qpos0. El objetivo es quedarse ahi.
    # La recompensa de postura es la señal dominante. Contacto y soporte tienen
    # peso bajo. Penalizaciones de esfuerzo casi apagadas para no
    # incentivar inmovilismo.
    # ------------------------------------------------------------------
    1: {
        "nombre": "fase_1_mantener_pose_xml",
        "descripcion": (
            "El robot empieza cerca de qpos0 del XML ideal y aprende a "
            "mantener exactamente esa pose. Reward principal: imitacion de "
            "pose XML. Contacto y soporte no dominan. Penalizaciones de "
            "esfuerzo casi apagadas."
        ),
        "ajustes": {
            **PESOS_RECOMPENSA_APAGADOS,
            "interacciones_minimas_curriculum_recompensa": (
                INTERACCIONES_MINIMAS_PARA_SUBIR_DE_FASE
            ),

            # XML de pose ideal: garantiza que la referencia es correcta.
            "xml_path": _XML_POSE_IDEAL,
            "usar_pose_referencia_xml": True,
            "usar_altura_referencia_xml": True,
            "include_pose_error": True,
            "usar_filtros_contacto_recompensa": True,

            # Reinicio: muy cerca de qpos0 del XML.
            "reset_pose_mode": "default",
            "reset_randomize_state": True,
            "reset_use_joint_base_pose": False,
            "reset_project_feet_to_floor": False,
            "reset_body_z_base": -1.0,
            "reset_body_xy_noise": 0.005,
            "reset_body_z_noise": 0.003,
            "reset_roll_pitch_deg": 2.0,
            "reset_yaw_deg": 5.0,
            "reset_q1_noise_deg": 4.0,
            "reset_q2_noise_deg": 4.0,
            "reset_q3_noise_deg": 4.0,
            "reset_noise_scale": 0.0,
            "reset_step_count_max": 0,
            "reset_episode_length_jitter_steps": 0,
            "reset_grace_step_jitter_steps": 0,

            # Terminación.
            "healthy_z_min": 0.03,
            "healthy_z_max": 0.35,
            "critical_z_min": 0.015,
            "critical_z_max": 0.50,
            "initial_grace_steps": 50,
            "terminate_on_persistent_invalid_ground_contact": True,
            "invalid_ground_contact_done_steps": 80,
            "body_or_chassis_ground_contact_done_steps": 8,
            "terminal_failure_penalty": 2.0,
            "clipear_reward_minimo": True,

            # Currículo de penalizaciones: arranca bajo para no bloquear
            # exploracion inicial.
            "curriculum_penalizaciones": 0.20,

            # === RECOMPENSAS ===
            # Pose: senal dominante.
            "pose_imitacion_reward_weight": 4.7,
            "pose_imitacion_sigma": 0.45,
            "pose_xml_final_reward_weight": 2.0,
            "pose_xml_final_tilt_sigma_deg": 7.0,
            "pose_xml_final_height_sigma": 0.020,

            # Altura del XML.
            "altura_reward_weight": 0.50,
            "target_height_sigma": 0.020,
            "altura_exceso_penalty_weight": 0.30,
            "altura_exceso_tolerancia": 0.010,
            "altura_exceso_sigma": 0.030,
            "altura_baja_penalty_weight": 0.30,
            "altura_baja_tolerancia": 0.006,
            "altura_baja_sigma": 0.022,

            # Orientacion.
            "vector_gravedad_reward_weight": 0.50,
            "vector_gravedad_sigma": 0.02,
            "vector_gravedad_estabilidad_reward_weight": 0.55,
            "cuerpo_paralelo_reward_weight": 0.60,
            "cuerpo_paralelo_sigma": 0.064,
            "q1_centrados_reward_weight": 0.60,
            "q1_center_sigma_deg": 12.0,

            # Geometria XML: evita que q4/q3 se cierren hacia el centro.
            "apertura_efectores_xml_reward_weight": 0.75,
            "apertura_efectores_xml_sigma": 0.050,
            "q3_distancia_xml_reward_weight": 0.40,
            "q3_distancia_xml_sigma": 0.050,

            # Quietud: peso moderado para que aprenda a mantener sin temblores,
            # pero no bloquea la correccion activa.
            "velocidad_lineal_cero_reward_weight": 0.35,
            "velocidad_lineal_cero_sigma": 0.08,
            "velocidad_angular_cero_reward_weight": 0.35,
            "velocidad_angular_cero_sigma": 0.60,
            "velocidad_articular_cero_reward_weight": 0.18,
            "velocidad_articular_cero_sigma": 2.0,

            # Contacto: muy bajo; la compuerta de postura ya lo filtra.
            "contactos_suelo_reward_weight": 0.15,
            "soporte_estatico_reward_weight": 0.0,
            "min_foot_contacts_for_support": 4,

            # Supervivencia.
            "supervivencia_reward_weight": 0.10,

            # === PENALIZACIONES (escaladas por k_c=0.20) ===
            "contacto_invalido_penalty_weight": 1.2,
            "contacto_invalido_persistente_penalty_weight": 0.8,
            "rodilla_suelo_penalty_weight": 3.5,
            "cuerpo_chasis_suelo_penalty_weight": 5.0,
            "inclinacion_suave_penalty_weight": 0.36,
            "inclinacion_suave_tolerancia_deg": 1.6,
            "inclinacion_suave_sigma_deg": 4.0,
            "efector_encima_q3_penalty_weight": 0.35,
            "velocidad_vertical_cuerpo_penalty_weight": 0.10,
            "velocidad_angular_cuerpo_penalty_weight": 0.10,
            "control_penalty_weight": 0.000,
            "cambio_accion_penalty_weight": 0.002,
            "limite_articular_penalty_weight": 0.01,
        },
    },

    # ------------------------------------------------------------------
    # FASE 2: LLEGAR a la pose ideal desde el suelo.
    #
    # El robot empieza lejos de la pose de referencia. Debe incorporarse
    # y llegar a qpos0. pose_imitacion sigue dominando. Se activan altura,
    # orientacion, contacto y soporte (filtrados por support_gate).
    # Penalizaciones moderadas.
    # ------------------------------------------------------------------
    2: {
        "nombre": "fase_2_llegar_a_pose_xml_desde_suelo",
        "descripcion": (
            "El robot empieza mas lejos de la pose de referencia (suelo "
            "variado). Debe llegar a qpos0 del XML ideal. pose_imitacion "
            "sigue siendo dominante; se anaden altura, orientacion, contacto "
            "y soporte. Penalizaciones moderadas."
        ),
        "ajustes": {
            **PESOS_RECOMPENSA_APAGADOS,
            "interacciones_minimas_curriculum_recompensa": (
                INTERACCIONES_MINIMAS_PARA_SUBIR_DE_FASE
            ),

            "xml_path": _XML_POSE_IDEAL,
            "usar_pose_referencia_xml": True,
            "usar_altura_referencia_xml": True,
            "include_pose_error": True,
            "usar_filtros_contacto_recompensa": True,

            # Reinicio: empieza en el suelo, variado.
            # reset_use_joint_base_pose=True solo para crear estados iniciales
            # tumbados; el OBJETIVO de la recompensa sigue siendo qpos0 del XML.
            "reset_pose_mode": "suelo_variado",
            "reset_randomize_state": True,
            "reset_use_joint_base_pose": True,
            "reset_q1_base_deg": 0.0,
            "reset_q2_base_deg": -45.0,
            "reset_q3_base_deg": -15.0,
            "reset_project_feet_to_floor": True,
            "reset_foot_ground_margin": 0.003,
            "reset_body_z_base": 0.07,
            "reset_body_xy_noise": 0.030,
            "reset_body_z_noise": 0.020,
            "reset_roll_pitch_deg": 25.0,
            "reset_yaw_deg": 180.0,
            "reset_q1_noise_deg": 25.0,
            "reset_q2_noise_deg": 35.0,
            "reset_q3_noise_deg": 35.0,
            "reset_noise_scale": 0.0,
            "reset_step_count_max": 100,
            "reset_episode_length_jitter_steps": 100,
            "reset_grace_step_jitter_steps": 30,

            # Terminación.
            "healthy_z_min": 0.015,
            "healthy_z_max": 0.45,
            "critical_z_min": 0.005,
            "critical_z_max": 0.65,
            "initial_grace_steps": 120,
            "terminate_on_persistent_invalid_ground_contact": True,
            "invalid_ground_contact_done_steps": 120,
            "body_or_chassis_ground_contact_done_steps": 25,
            "terminal_failure_penalty": 3.0,
            "clipear_reward_minimo": True,

            "curriculum_penalizaciones": 0.45,

            # === RECOMPENSAS ===
            "pose_imitacion_reward_weight": 4.7,
            "pose_imitacion_sigma": 0.30,
            "pose_xml_final_reward_weight": 2.4,
            "pose_xml_final_tilt_sigma_deg": 6.5,
            "pose_xml_final_height_sigma": 0.020,

            "altura_reward_weight": 0.50,
            "target_height_sigma": 0.020,
            "altura_exceso_penalty_weight": 0.30,
            "altura_exceso_tolerancia": 0.012,
            "altura_exceso_sigma": 0.035,
            "altura_baja_penalty_weight": 0.30,
            "altura_baja_tolerancia": 0.008,
            "altura_baja_sigma": 0.026,

            "vector_gravedad_reward_weight": 0.50,
            "vector_gravedad_sigma": 0.02,
            "vector_gravedad_estabilidad_reward_weight": 0.65,
            "cuerpo_paralelo_reward_weight": 0.60,
            "cuerpo_paralelo_sigma": 0.064,
            "q1_centrados_reward_weight": 0.60,
            "q1_center_sigma_deg": 12.0,

            "apertura_efectores_xml_reward_weight": 0.85,
            "apertura_efectores_xml_sigma": 0.055,
            "q3_distancia_xml_reward_weight": 0.50,
            "q3_distancia_xml_sigma": 0.055,

            "velocidad_lineal_cero_reward_weight": 0.15,
            "velocidad_lineal_cero_sigma": 0.12,
            "velocidad_angular_cero_reward_weight": 0.20,
            "velocidad_angular_cero_sigma": 0.90,
            "velocidad_articular_cero_reward_weight": 0.16,
            "velocidad_articular_cero_sigma": 2.4,

            # Contacto y soporte activos (filtrados por support_gate).
            "contactos_suelo_reward_weight": 0.10,
            "soporte_estatico_reward_weight": 0.60,
            "min_foot_contacts_for_support": 3,

            "supervivencia_reward_weight": 0.05,

            # === PENALIZACIONES (escaladas por k_c=0.45) ===
            "contacto_invalido_penalty_weight": 2.2,
            "contacto_invalido_persistente_penalty_weight": 1.5,
            "rodilla_suelo_penalty_weight": 3.8,
            "cuerpo_chasis_suelo_penalty_weight": 5.0,
            "inclinacion_suave_penalty_weight": 0.36,
            "inclinacion_suave_tolerancia_deg": 1.6,
            "inclinacion_suave_sigma_deg": 4.0,
            "efector_encima_q3_penalty_weight": 0.50,
            "velocidad_vertical_cuerpo_penalty_weight": 0.25,
            "velocidad_angular_cuerpo_penalty_weight": 0.35,
            "control_penalty_weight": 0.002,
            "cambio_accion_penalty_weight": 0.004,
            "limite_articular_penalty_weight": 0.025,
        },
    },

    # ------------------------------------------------------------------
    # FASE 3: RECUPERAR la pose ideal desde caidas/perturbaciones.
    #
    # El robot empieza desde posiciones extremas. Debe volver a qpos0 y
    # quedar estable. Sigma de pose mas estricto. Penalizaciones mas
    # fuertes. Contacto y soporte siguen filtrados por support_gate.
    # ------------------------------------------------------------------
    3: {
        "nombre": "fase_3_recuperar_pose_xml_desde_caida",
        "descripcion": (
            "El robot empieza desde posiciones extremas (caida variada). "
            "Debe volver a la pose del XML y quedar estable. Sigma de pose "
            "mas estricto. Penalizaciones mas fuertes. Contacto y soporte "
            "filtrados por support_gate (pose+altura+orientacion)."
        ),
        "ajustes": {
            **PESOS_RECOMPENSA_APAGADOS,
            "interacciones_minimas_curriculum_recompensa": (
                INTERACCIONES_MINIMAS_PARA_SUBIR_DE_FASE
            ),

            "xml_path": _XML_POSE_IDEAL,
            "usar_pose_referencia_xml": True,
            "usar_altura_referencia_xml": True,
            "include_pose_error": True,
            "usar_filtros_contacto_recompensa": True,

            # Reinicio: caída variada desde posiciones extremas.
            "reset_pose_mode": "caida_variada",
            "reset_randomize_state": True,
            "reset_use_joint_base_pose": False,
            "reset_project_feet_to_floor": False,
            "reset_body_z_base": 0.22,
            "reset_body_xy_noise": 0.060,
            "reset_body_z_noise": 0.060,
            "reset_roll_pitch_deg": 70.0,
            "reset_yaw_deg": 180.0,
            "reset_q1_noise_deg": 45.0,
            "reset_q2_noise_deg": 50.0,
            "reset_q3_noise_deg": 50.0,
            "reset_noise_scale": 0.0,
            "reset_step_count_max": 200,
            "reset_episode_length_jitter_steps": 200,
            "reset_grace_step_jitter_steps": 50,

            # Terminación.
            "healthy_z_min": 0.010,
            "healthy_z_max": 0.60,
            "critical_z_min": 0.003,
            "critical_z_max": 0.80,
            "initial_grace_steps": 180,
            "terminate_on_persistent_invalid_ground_contact": True,
            "invalid_ground_contact_done_steps": 160,
            "body_or_chassis_ground_contact_done_steps": 40,
            "terminal_failure_penalty": 3.0,
            "clipear_reward_minimo": True,

            "curriculum_penalizaciones": 0.80,

            # === RECOMPENSAS ===
            # Sigma mas estricto: exige parecido real a la pose XML.
            "pose_imitacion_reward_weight": 5.3,
            "pose_imitacion_sigma": 0.18,
            "pose_xml_final_reward_weight": 2.8,
            "pose_xml_final_tilt_sigma_deg": 6.0,
            "pose_xml_final_height_sigma": 0.020,

            "altura_reward_weight": 0.50,
            "target_height_sigma": 0.020,
            "altura_exceso_penalty_weight": 0.30,
            "altura_exceso_tolerancia": 0.012,
            "altura_exceso_sigma": 0.032,
            "altura_baja_penalty_weight": 0.30,
            "altura_baja_tolerancia": 0.008,
            "altura_baja_sigma": 0.026,

            "vector_gravedad_reward_weight": 0.50,
            "vector_gravedad_sigma": 0.02,
            "vector_gravedad_estabilidad_reward_weight": 0.75,
            "cuerpo_paralelo_reward_weight": 0.60,
            "cuerpo_paralelo_sigma": 0.064,
            "q1_centrados_reward_weight": 0.60,
            "q1_center_sigma_deg": 12.0,

            "apertura_efectores_xml_reward_weight": 0.95,
            "apertura_efectores_xml_sigma": 0.050,
            "q3_distancia_xml_reward_weight": 0.65,
            "q3_distancia_xml_sigma": 0.050,

            "velocidad_lineal_cero_reward_weight": 0.25,
            "velocidad_lineal_cero_sigma": 0.10,
            "velocidad_angular_cero_reward_weight": 0.35,
            "velocidad_angular_cero_sigma": 0.70,
            "velocidad_articular_cero_reward_weight": 0.22,
            "velocidad_articular_cero_sigma": 2.2,

            "contactos_suelo_reward_weight": 0.10,
            "soporte_estatico_reward_weight": 0.85,
            "min_foot_contacts_for_support": 3,

            "supervivencia_reward_weight": 0.05,

            # === PENALIZACIONES (escaladas por k_c=0.80) ===
            "contacto_invalido_penalty_weight": 3.0,
            "contacto_invalido_persistente_penalty_weight": 2.1,
            "rodilla_suelo_penalty_weight": 4.2,
            "cuerpo_chasis_suelo_penalty_weight": 5.5,
            "inclinacion_suave_penalty_weight": 0.36,
            "inclinacion_suave_tolerancia_deg": 1.6,
            "inclinacion_suave_sigma_deg": 4.0,
            "efector_encima_q3_penalty_weight": 0.70,
            "velocidad_vertical_cuerpo_penalty_weight": 0.40,
            "velocidad_angular_cuerpo_penalty_weight": 0.55,
            "control_penalty_weight": 0.004,
            "cambio_accion_penalty_weight": 0.008,
            "limite_articular_penalty_weight": 0.040,
        },
    },
}


def aplicar_fase_curriculo_recompensa(
    configuracion_entorno: config_dict.ConfigDict,
    fase: int,
) -> dict[str, Any]:
    """Mete en el entorno los pesos y resets de la fase elegida."""

    if fase not in FASES_RECOMPENSA:
        raise ValueError(f"Fase de recompensa no valida: {fase}. Usa 0, 1, 2 o 3.")

    fase_recompensa = FASES_RECOMPENSA[fase]
    for nombre_parametro, valor_parametro in fase_recompensa["ajustes"].items():
        configuracion_entorno[nombre_parametro] = valor_parametro

    configuracion_entorno.fase_curriculum_recompensa = fase
    configuracion_entorno.nombre_curriculum_recompensa = fase_recompensa["nombre"]
    return preparar_resumen_fase_recompensa(fase)


def preparar_resumen_fase_recompensa(fase: int) -> dict[str, Any]:
    """Prepara el JSON que se guarda dentro de la carpeta del entrenamiento."""

    if fase not in FASES_RECOMPENSA:
        raise ValueError(f"Fase de recompensa no valida: {fase}. Usa 0, 1, 2 o 3.")

    fase_recompensa = FASES_RECOMPENSA[fase]
    return {
        "fase": fase,
        "nombre": fase_recompensa["nombre"],
        "descripcion": fase_recompensa["descripcion"],
        "interacciones_minimas_para_subir": (
            0 if fase == 0 else INTERACCIONES_MINIMAS_PARA_SUBIR_DE_FASE
        ),
        "regla_para_subir_de_fase": (
            "Subir manualmente a la siguiente fase solo cuando haya mas de "
            "5 millones de interacciones y la grafica tenga un suelo estable "
            "que ya no se rompa hacia abajo."
        ),
        "ajustes": dict(fase_recompensa["ajustes"]),
    }
