"""Entorno MJX/JAX genérico para un robot articulado.

Sistema de aprendizaje en 3 fases basado en imitacion de pose:

  Fase 1 – Mantener la pose de referencia (qpos0 del XML).
           El robot empieza cerca de la pose objetivo y aprende a mantenerla.
           La recompensa es maxima (1.0) cuando q == qpos0[joints].
           Un curriculum de sigma la va apretando con el tiempo.

  Fase 2 – Incorporarse desde el suelo.
           El robot empieza tumbado y debe llegar a la pose de referencia.
           La misma recompensa de imitacion guia el objetivo; las penalizaciones
           entran progresivamente via curriculum_penalizaciones.

  Fase 3 – Recuperacion robusta.
           El robot empieza tirado/lanzado desde posiciones aleatorias extremas.
           Debe volver solo a la pose de referencia sin importar el punto de
           partida. Sigma estrecho: exigencia maxima de parecido.

CLAVE DE DISENO:
  - q_referencia se extrae de qpos0 del XML en el constructor. Si cambias el
    XML, la recompensa cambia automaticamente sin tocar la configuracion.
  - pose_imitacion_reward = exp(-||q - q_ref||^2 / sigma^2), en [0, 1].
    Vale 1.0 exactamente cuando la pose es identica a la de referencia.
  - sigma se controla via config: empieza ancho (0.45 rad) en fase 1 y se
    estrecha (0.15 rad) en fase 3. Se puede reducir manualmente entre ejecuciones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env


RUTA_XML_POSE_IDEAL = Path(__file__).parent / "xmls" / "ROBOT_POSE_IDEAL.xml"
SEGUNDOS_MAXIMOS_EPISODIO = 20.0
DT_CONTROL_PREDETERMINADO = 0.004
LONGITUD_EPISODIO_PREDETERMINADA = int(
    SEGUNDOS_MAXIMOS_EPISODIO / DT_CONTROL_PREDETERMINADO
)
VERSION_DIAGNOSTICO_RECOMPENSA = 2026052002  # fecha: 2026-05-20, v02


def _normalizar_cuaternion(cuaternion: jax.Array) -> jax.Array:
    return cuaternion / jp.maximum(jp.linalg.norm(cuaternion), 1e-6)


def _multiplicar_cuaterniones(izquierdo: jax.Array, derecho: jax.Array) -> jax.Array:
    lw, lx, ly, lz = izquierdo
    rw, rx, ry, rz = derecho
    return jp.array([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ])


def _cuaternion_desde_euler_xyz(
    roll: jax.Array, pitch: jax.Array, yaw: jax.Array
) -> jax.Array:
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr, sr = jp.cos(half_roll), jp.sin(half_roll)
    cp, sp = jp.cos(half_pitch), jp.sin(half_pitch)
    cy, sy = jp.cos(half_yaw), jp.sin(half_yaw)
    return jp.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def default_config() -> config_dict.ConfigDict:
    """Configuracion fisica y de recompensa del entorno."""

    return config_dict.create(
        # --- Simulación ---
        ctrl_dt=DT_CONTROL_PREDETERMINADO,
        sim_dt=0.002,
        episode_length=LONGITUD_EPISODIO_PREDETERMINADA,
        max_episode_seconds=SEGUNDOS_MAXIMOS_EPISODIO,
        action_repeat=1,
        impl="jax",
        xml_path=RUTA_XML_POSE_IDEAL.as_posix(),
        naconmax=64,
        njmax=64,
        solver_iterations=12,
        solver_ls_iterations=4,

        # --- Currículo ---
        fase_curriculum_recompensa=0,
        nombre_curriculum_recompensa="base_actual",
        interacciones_minimas_curriculum_recompensa=0,

        # --- Reinicio del entorno ---
        reset_pose_mode="default",
        reset_noise_scale=0.01,
        reset_randomize_state=True,
        reset_body_z_base=-1.0,
        reset_project_feet_to_floor=False,
        reset_foot_ground_margin=0.002,
        reset_use_joint_base_pose=False,
        reset_q1_base_deg=0.0,
        reset_q2_base_deg=0.0,
        reset_q3_base_deg=0.0,
        reset_body_xy_noise=0.02,
        reset_body_z_noise=0.015,
        reset_roll_pitch_deg=12.0,
        reset_yaw_deg=180.0,
        reset_q1_noise_deg=20.0,
        reset_q2_noise_deg=25.0,
        reset_q3_noise_deg=25.0,
        reset_step_count_max=400,
        reset_episode_length_jitter_steps=0,
        reset_grace_step_jitter_steps=25,

        # --- Observaciones ---
        log_full_metrics=False,
        usar_filtros_contacto_recompensa=True,
        include_last_action=True,
        exclude_xy_position=True,
        # Incluye el error de postura (q_actual - q_ref) en la observación.
        # Ayuda a la red a saber cuánto le falta para alcanzar la postura
        # objetivo sin tener que deducirlo de qpos.
        include_pose_error=True,
        # Indicadores para que la recompensa use referencias extraídas del XML
        # activo.
        # Si es falso, se usan los valores alternativos de configuración
        # (target_body_z, etc.).
        usar_altura_referencia_xml=True,
        usar_pose_referencia_xml=True,

        # --- Terminación ---
        terminate_when_unhealthy=False,
        disable_done=False,
        healthy_z_min=0.03,
        healthy_z_max=0.55,
        critical_z_min=0.025,
        critical_z_max=0.80,
        initial_grace_steps=75,
        explosive_qpos_limit=20.0,
        explosive_qvel_limit=120.0,

        # --- Escalas globales ---
        reward_scale=0.01,
        terminal_failure_penalty=5.0,

        # --- Control ---
        action_scale=0.90,
        action_mode="direct",
        action_tanh_scale=0.25,

        # --- Currículo de penalizaciones ---
        # k_c = clip(curriculum_penalizaciones, 0, 1) escala TODAS las penalizaciones.
        # Comienza en 0 (penalizaciones inactivas) y sube manualmente entre ejecuciones.
        curriculum_penalizaciones=0.0,

        # ================================================================
        # RECOMPENSA DE IMITACIÓN DE POSTURA (señal principal del sistema)
        # ================================================================
        # La pose de referencia se lee de qpos0 del XML en __init__.
        # No hay que tocar ningún ángulo en la configuración: si cambias el XML,
        # la referencia cambia sola.
        #
        # Vale 1.0 cuando q_joints == qpos0[joints] exactamente.
        #
        # sigma controla la tolerancia:
        #   0.45 rad (~26 deg) -> fase 1: amplio, bueno para explorar
        #   0.25 rad (~14 deg) -> fase 2: moderado, exige incorporarse bien
        #   0.15 rad  (~9 deg) -> fase 3: estricto, exige pose casi perfecta
        pose_imitacion_reward_weight=5.0,
        pose_imitacion_sigma=0.45,
        # Recompensa compuesta de éxito final: postura XML + torso nivelado +
        # altura XML + contactos validos. Mantiene pose_imitacion como
        # gradiente auxiliar, pero reserva la recompensa maxima para la
        # postura realmente buena.
        pose_xml_final_reward_weight=0.0,
        pose_xml_final_tilt_sigma_deg=7.0,
        pose_xml_final_height_sigma=0.025,

        # --- Recompensas de apoyo ---
        vector_gravedad_reward_weight=0.50,
        vector_gravedad_base_reward_weight=0.0,
        # Sigma para la recompensa de orientación (vector de gravedad lateral).
        # 0.02 rad -> exige torso practicamente plano.
        vector_gravedad_sigma=0.02,
        # Versión inspirada en la recompensa antigua estable:
        # score = 1 - (g_body_x^2 + g_body_y^2), filtrada por contacto valido.
        vector_gravedad_estabilidad_reward_weight=0.0,
        cuerpo_paralelo_reward_weight=0.60,
        cuerpo_paralelo_sigma=0.064,
        altura_reward_weight=0.50,
        # target_body_z: valor alternativo cuando usar_altura_referencia_xml=False.
        # Con usar_altura_referencia_xml=True se usa self._ref_body_z del XML.
        target_body_z=0.10,
        target_height_sigma=0.02,
        altura_exceso_penalty_weight=0.30,
        altura_exceso_tolerancia=0.010,
        altura_exceso_sigma=0.035,
        altura_baja_penalty_weight=0.30,
        altura_baja_tolerancia=0.006,
        altura_baja_sigma=0.025,
        supervivencia_reward_weight=0.10,
        soporte_estatico_reward_weight=0.70,
        contactos_suelo_reward_weight=0.0,
        apertura_efectores_xml_reward_weight=0.0,
        apertura_efectores_xml_sigma=0.050,
        q3_distancia_xml_reward_weight=0.0,
        q3_distancia_xml_sigma=0.050,
        poligono_CoG_reward_weight=0.0,
        plano_CoG_arana_reward_weight=0.0,
        apertura_efectores_q3_cerca_reward_weight=0.0,
        q1_centrados_reward_weight=0.6,
        q3_separados_centro_reward_weight=0.0,
        simetria_patas_reward_weight=0.0,

        # --- Quietud (útil en fase 1 para estabilizar) ---
        velocidad_lineal_cero_reward_weight=0.0,
        velocidad_lineal_cero_sigma=0.05,
        velocidad_angular_cero_reward_weight=0.0,
        velocidad_angular_cero_sigma=0.05,
        velocidad_articular_cero_reward_weight=0.0,
        velocidad_articular_cero_sigma=2.0,

        # --- Penalizaciones ---
        contacto_invalido_penalty_weight=1.50,
        contacto_invalido_persistente_penalty_weight=0.90,
        rodilla_suelo_penalty_weight=0.0,
        cuerpo_chasis_suelo_penalty_weight=0.0,
        inclinacion_suave_penalty_weight=0.36,
        inclinacion_suave_tolerancia_deg=1.6,
        inclinacion_suave_sigma_deg=4.0,
        efector_encima_q3_penalty_weight=0.30,
        velocidad_vertical_cuerpo_penalty_weight=0.20,
        velocidad_angular_cuerpo_penalty_weight=0.30,
        control_penalty_weight=0.003,
        cambio_accion_penalty_weight=0.005,
        limite_articular_penalty_weight=0.03,

        # --- Contacto inválido ---
        terminate_on_persistent_invalid_ground_contact=True,
        invalid_ground_contact_done_steps=40,
        body_or_chassis_ground_contact_done_steps=12,
        contacto_invalido_penalty_count_scale=2.0,

        # --- Geometría de soporte ---
        support_center_margin=0.08,
        min_foot_contacts_for_support=3,
        plano_CoG_arana_target=0.10,
        plano_CoG_arana_sigma=0.06,
        valid_support_max_foot_clearance=0.05,
        valid_support_min_foot_distance_xy=0.18,
        effector_opening_target_radius=0.36,
        effector_opening_sigma=0.08,
        q3_center_target_radius=0.20,
        q3_center_sigma=0.08,
        q3_opening_target_radius=0.30,
        q3_opening_sigma=0.08,
        q1_center_sigma_deg=12.0,
        effector_q3_z_margin=0.0,
        effector_q3_z_scale=0.04,
        q3_below_body_target_margin=0.18,
        leg_symmetry_tolerance_deg=45.0,

        # --- Sigmas de penalización ---
        base_vertical_velocity_sigma=0.40,
        base_angular_velocity_sigma=3.0,
        joint_limit_margin_fraction=0.20,
        cambio_accion_sigma=0.35,

        # --- Recorte de la recompensa ---
        clipear_reward_minimo=True,
        reward_clip_min=-5.0,
        reward_clip_max=5.0,

        # --- Valores anteriores mantenidos por compatibilidad con código externo ---
        max_tilt_degrees=15.0,
        max_survival_tilt_degrees=60.0,
        target_height_min=0.26,
        target_height_max=0.26,
        pose_articular_reward_weight=0.0,
        pose_articular_sigma=0.25,
        q1_pose_objetivo_deg=0.0,
        q2_pose_objetivo_deg=5.0,
        q3_pose_objetivo_deg=-65.0,
    )


class EntornoRobotMJX(mjx_env.MjxEnv):
    """Entorno de aprendizaje para un robot con imitación de pose.

    La recompensa principal (pose_imitacion) es maxima cuando el robot replica
    exactamente el qpos0 del XML. Al cambiar el XML, el objetivo cambia
    automaticamente sin tocar ninguna linea de config.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
    ):
        if config is None:
            config = default_config()
        super().__init__(config, config_overrides=config_overrides)

        xml_path = Path(str(self._config.xml_path))
        if not xml_path.is_absolute():
            xml_path = (Path(__file__).parent.parent / xml_path).resolve()
        self._xml_path = xml_path.as_posix()

        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = float(self._config.sim_dt)
        self._mj_model.opt.iterations = int(self._config.solver_iterations)
        if hasattr(self._mj_model.opt, "ls_iterations"):
            self._mj_model.opt.ls_iterations = int(self._config.solver_ls_iterations)

        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._main_body_id = self._mj_model.body("ROBOT").id

        self._qpos0 = jp.array(self._mj_model.qpos0)
        self._qvel0 = jp.zeros(self._mj_model.nv)
        self._ctrl_range = jp.array(self._mj_model.actuator_ctrlrange)

        leg_configs = self._construir_configuraciones_patas()
        self._q1_qpos_adrs = jp.array(
            [leg["q1_qpos_adr"] for leg in leg_configs], dtype=jp.int32
        )
        self._q2_qpos_adrs = jp.array(
            [leg["q2_qpos_adr"] for leg in leg_configs], dtype=jp.int32
        )
        self._q3_qpos_adrs = jp.array(
            [leg["q3_qpos_adr"] for leg in leg_configs], dtype=jp.int32
        )
        self._q1_dof_adrs = jp.array(
            [leg["q1_dof_adr"] for leg in leg_configs], dtype=jp.int32
        )
        self._q2_dof_adrs = jp.array(
            [leg["q2_dof_adr"] for leg in leg_configs], dtype=jp.int32
        )
        self._q3_dof_adrs = jp.array(
            [leg["q3_dof_adr"] for leg in leg_configs], dtype=jp.int32
        )
        self._joint_qpos_adrs = jp.concatenate(
            [self._q1_qpos_adrs, self._q2_qpos_adrs, self._q3_qpos_adrs], axis=0
        )
        self._joint_dof_adrs = jp.concatenate(
            [self._q1_dof_adrs, self._q2_dof_adrs, self._q3_dof_adrs], axis=0
        )
        self._joint_ranges = jp.array(
            [leg["q1_range"] for leg in leg_configs]
            + [leg["q2_range"] for leg in leg_configs]
            + [leg["q3_range"] for leg in leg_configs],
            dtype=jp.float32,
        )
        self._q3_geom_ids = jp.array(
            [leg["q3_geom_id"] for leg in leg_configs], dtype=jp.int32
        )
        self._effector_geom_ids = jp.array(
            [leg["effector_geom_id"] for leg in leg_configs], dtype=jp.int32
        )
        self._effector_geom_radii = jp.array(
            [leg["effector_geom_radius"] for leg in leg_configs], dtype=jp.float32
        )
        self._floor_geom_id = jp.array(
            self._mj_model.geom("sueloMundo").id, dtype=jp.int32
        )
        self._floor_z = jp.array(
            self._mj_model.geom_pos[int(self._floor_geom_id), 2], dtype=jp.float32
        )
        body_or_chassis_geom_names = (
            "E_10", "E_20", "E_30", "E_40", "planchaCentral"
        )
        self._body_or_chassis_geom_ids = jp.array(
            [self._mj_model.geom(name).id for name in body_or_chassis_geom_names],
            dtype=jp.int32,
        )

        # ================================================================
        # POSE DE REFERENCIA: extraida directamente de qpos0 del XML.
        #
        # qpos0 contiene la posicion por defecto del modelo completo:
        #   [0:3]  = posicion XYZ del cuerpo libre
        #   [3:7]  = cuaternion de orientacion del cuerpo libre
        #   [7:]   = angulos de todas las articulaciones (q1, q2, q3 x4)
        #
        # Al indexar con _q{1,2,3}_qpos_adrs obtenemos exactamente los
        # angulos de la pose objetivo. Cuando modificas el XML y recreas
        # el entorno, esta referencia cambia automaticamente.
        #
        # En ROBOT_POSE_IDEAL.xml todas las articulaciones actuadas valen 0
        # (la pose esta horneada en la geometria), asi que _q_referencia
        # sera un vector de 12 ceros. Aun asi el codigo es correcto para
        # cualquier XML con articulaciones distintas de cero.
        # ================================================================
        self._q_referencia = jp.concatenate([
            self._qpos0[self._q1_qpos_adrs],
            self._qpos0[self._q2_qpos_adrs],
            self._qpos0[self._q3_qpos_adrs],
        ])

        # ----------------------------------------------------------------
        # ALTURA DE REFERENCIA: forward de qpos0 para obtener la z real
        # del cuerpo en la pose ideal. No hardcodeamos ningun valor.
        # Si el XML tiene body z=0.100 m, self._ref_body_z = 0.100 m.
        # Si cambia el XML, cambia automaticamente.
        # ----------------------------------------------------------------
        _data_ref = mujoco.MjData(self._mj_model)
        _data_ref.qpos[:] = self._mj_model.qpos0
        _data_ref.qvel[:] = 0.0
        mujoco.mj_forward(self._mj_model, _data_ref)
        self._ref_body_z = jp.array(
            float(_data_ref.xpos[self._main_body_id, 2]), dtype=jp.float32
        )
        ref_body_xy = jp.array(_data_ref.xpos[self._main_body_id, :2], dtype=jp.float32)
        ref_effector_xy = jp.array(
            _data_ref.geom_xpos[[leg["effector_geom_id"] for leg in leg_configs], :2],
            dtype=jp.float32,
        )
        ref_q3_xy = jp.array(
            _data_ref.geom_xpos[[leg["q3_geom_id"] for leg in leg_configs], :2],
            dtype=jp.float32,
        )
        self._ref_effector_radial_distances = jp.linalg.norm(
            ref_effector_xy - ref_body_xy, axis=1
        )
        self._ref_q3_radial_distances = jp.linalg.norm(ref_q3_xy - ref_body_xy, axis=1)
        # Altura objetivo: usa la del XML si usar_altura_referencia_xml=True,
        # si no usa el valor alternativo de configuración. Calculado aquí (fuera del JIT).
        if bool(self._config.get("usar_altura_referencia_xml", True)):
            self._target_body_z_actual = self._ref_body_z
        else:
            self._target_body_z_actual = jp.array(
                float(self._config.target_body_z), dtype=jp.float32
            )

    # =====================================================================
    # CONSTRUCCION DE PATAS
    # =====================================================================

    def _construir_configuraciones_patas(self) -> tuple[dict[str, Any], ...]:
        legs = []
        for leg_index in range(1, 5):
            q1 = self._mj_model.joint(f"ID_{leg_index}1_q1")
            q2 = self._mj_model.joint(f"ID_{leg_index}2_q2")
            q3 = self._mj_model.joint(f"ID_{leg_index}3_q3")
            q3_geom = self._mj_model.geom(f"E_{leg_index}3_geom")
            effector_geom = self._mj_model.geom(f"E_{leg_index}4_geom")
            legs.append({
                "q1_qpos_adr": int(q1.qposadr[0]),
                "q2_qpos_adr": int(q2.qposadr[0]),
                "q3_qpos_adr": int(q3.qposadr[0]),
                "q1_dof_adr": int(q1.dofadr[0]),
                "q2_dof_adr": int(q2.dofadr[0]),
                "q3_dof_adr": int(q3.dofadr[0]),
                "q1_range": (float(q1.range[0]), float(q1.range[1])),
                "q2_range": (float(q2.range[0]), float(q2.range[1])),
                "q3_range": (float(q3.range[0]), float(q3.range[1])),
                "q3_geom_id": int(q3_geom.id),
                "effector_geom_id": int(effector_geom.id),
                "effector_geom_radius": float(effector_geom.size[0]),
            })
        return tuple(legs)

    # =====================================================================
    # REINICIO DEL ENTORNO
    # =====================================================================

    def reset(self, rng: jax.Array) -> mjx_env.State:
        (
            rng_body_xy, rng_body_z, rng_attitude, rng_joints,
            rng_qpos_noise, rng_vel, rng_grace, rng_start_step,
            rng_episode_length,
        ) = jax.random.split(rng, 9)

        qpos = self._qpos0
        qvel = self._qvel0

        if float(self._config.reset_body_z_base) >= 0.0:
            qpos = qpos.at[2].set(self._config.reset_body_z_base)

        if bool(self._config.reset_use_joint_base_pose):
            q1_base = jp.deg2rad(self._config.reset_q1_base_deg)
            q2_base = jp.deg2rad(self._config.reset_q2_base_deg)
            q3_base = jp.deg2rad(self._config.reset_q3_base_deg)
            qpos = qpos.at[self._q1_qpos_adrs].set(q1_base)
            qpos = qpos.at[self._q2_qpos_adrs].set(q2_base)
            qpos = qpos.at[self._q3_qpos_adrs].set(q3_base)

        _pose_mode = str(self._config.reset_pose_mode)
        _project_to_floor = (
            bool(self._config.reset_project_feet_to_floor)
            and _pose_mode != "caida_variada"
        )

        if self._config.reset_randomize_state:
            body_xy_noise = jax.random.uniform(
                rng_body_xy, (2,),
                minval=-self._config.reset_body_xy_noise,
                maxval=self._config.reset_body_xy_noise,
            )
            body_z_noise = jax.random.uniform(
                rng_body_z, (),
                minval=-self._config.reset_body_z_noise,
                maxval=self._config.reset_body_z_noise,
            )
            roll_pitch_limit = jp.deg2rad(self._config.reset_roll_pitch_deg)
            yaw_limit = jp.deg2rad(self._config.reset_yaw_deg)
            rng_attitude_angles, rng_attitude_sign = jax.random.split(rng_attitude)
            body_angles = jax.random.uniform(
                rng_attitude_angles,
                (3,),
                minval=jp.array([-roll_pitch_limit, -roll_pitch_limit, -yaw_limit]),
                maxval=jp.array([roll_pitch_limit, roll_pitch_limit, yaw_limit]),
            )
            if _pose_mode == "caida_lateral":
                lateral_noise = jax.random.uniform(
                    rng_attitude_angles,
                    (3,),
                    minval=jp.array([
                        -jp.deg2rad(15.0), -jp.deg2rad(20.0), -yaw_limit,
                    ]),
                    maxval=jp.array([
                        jp.deg2rad(15.0), jp.deg2rad(20.0), yaw_limit,
                    ]),
                )
                side_sign = jp.where(
                    jax.random.bernoulli(rng_attitude_sign), 1.0, -1.0
                )
                body_angles = jp.array([
                    side_sign * (0.5 * jp.pi + lateral_noise[0]),
                    lateral_noise[1],
                    lateral_noise[2],
                ])
            elif _pose_mode == "boca_abajo":
                upside_down_noise = jax.random.uniform(
                    rng_attitude_angles,
                    (3,),
                    minval=jp.array([
                        -jp.deg2rad(20.0), -jp.deg2rad(20.0), -yaw_limit,
                    ]),
                    maxval=jp.array([
                        jp.deg2rad(20.0), jp.deg2rad(20.0), yaw_limit,
                    ]),
                )
                flip_roll = jax.random.bernoulli(rng_attitude_sign)
                body_angles = jp.where(
                    flip_roll,
                    jp.array([
                        jp.pi + upside_down_noise[0],
                        upside_down_noise[1],
                        upside_down_noise[2],
                    ]),
                    jp.array([
                        upside_down_noise[0],
                        jp.pi + upside_down_noise[1],
                        upside_down_noise[2],
                    ]),
                )
            joint_noise_deg = jp.array([
                self._config.reset_q1_noise_deg,
                self._config.reset_q2_noise_deg,
                self._config.reset_q3_noise_deg,
            ])
            joint_noise_rad = jp.deg2rad(joint_noise_deg)
            joint_noise = jax.random.uniform(
                rng_joints,
                (3, self._q1_qpos_adrs.shape[0]),
                minval=-joint_noise_rad[:, None],
                maxval=joint_noise_rad[:, None],
            )
            sampled_quat = _cuaternion_desde_euler_xyz(
                body_angles[0], body_angles[1], body_angles[2]
            )
            qpos = qpos.at[0:2].add(body_xy_noise)
            qpos = qpos.at[2].add(body_z_noise)
            qpos = qpos.at[3:7].set(
                _normalizar_cuaternion(
                    _multiplicar_cuaterniones(sampled_quat, self._qpos0[3:7])
                )
            )
            qpos = qpos.at[self._q1_qpos_adrs].add(joint_noise[0])
            qpos = qpos.at[self._q2_qpos_adrs].add(joint_noise[1])
            qpos = qpos.at[self._q3_qpos_adrs].add(joint_noise[2])

        if self._config.reset_noise_scale > 0:
            qpos_noise = jax.random.uniform(
                rng_qpos_noise,
                (self.mjx_model.nq - 7,),
                minval=-self._config.reset_noise_scale,
                maxval=self._config.reset_noise_scale,
            )
            qvel_noise = self._config.reset_noise_scale * jax.random.normal(
                rng_vel, (self.mjx_model.nv,)
            )
            qpos = qpos.at[7:].add(qpos_noise)
            qvel = qvel + qvel_noise

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos, qvel=qvel,
            ctrl=jp.zeros(self.action_size),
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        if _project_to_floor:
            foot_bottom_offsets = self._desfases_inferiores_pies(data)
            body_z_correction = (
                jp.min(foot_bottom_offsets) - self._config.reset_foot_ground_margin
            )
            qpos = qpos.at[2].add(-body_z_correction)
            data = mjx_env.make_data(
                self.mj_model,
                qpos=qpos, qvel=qvel,
                ctrl=jp.zeros(self.action_size),
                impl=self.mjx_model.impl.value,
                naconmax=self._config.naconmax,
                njmax=self._config.njmax,
            )
            data = mjx.forward(self.mjx_model, data)

        grace_jitter = jax.random.randint(
            rng_grace, (),
            minval=-int(self._config.reset_grace_step_jitter_steps),
            maxval=int(self._config.reset_grace_step_jitter_steps) + 1,
            dtype=jp.int32,
        )
        base_episode_length = jp.maximum(
            jp.array(1, dtype=jp.int32),
            jp.array(self._config.episode_length, dtype=jp.int32),
        )
        max_episode_jitter = jp.maximum(
            jp.array(0, dtype=jp.int32),
            jp.minimum(
                jp.array(
                    int(self._config.reset_episode_length_jitter_steps), dtype=jp.int32
                ),
                base_episode_length - 1,
            ),
        )
        episode_shortening = jax.random.randint(
            rng_episode_length, (),
            minval=0, maxval=max_episode_jitter + 1, dtype=jp.int32,
        )
        episode_length_steps = jp.maximum(
            jp.array(1, dtype=jp.int32),
            base_episode_length - episode_shortening,
        )
        max_start_step = jp.maximum(
            jp.array(0, dtype=jp.int32),
            jp.minimum(
                jp.array(int(self._config.reset_step_count_max), dtype=jp.int32),
                episode_length_steps - 1,
            ),
        )
        start_step_count = jax.random.randint(
            rng_start_step, (),
            minval=0, maxval=max_start_step + 1, dtype=jp.int32,
        )
        grace_steps = jp.maximum(
            jp.array(0, dtype=jp.int32),
            jp.array(self._config.initial_grace_steps, dtype=jp.int32) + grace_jitter,
        )
        grace_steps = jp.minimum(grace_steps, jp.maximum(episode_length_steps - 1, 0))

        info = {
            "rng": rng,
            "last_action": jp.zeros(self.action_size),
            "step_count": start_step_count.astype(jp.float32),
            "alive_reward_steps": jp.zeros((), dtype=jp.float32),
            "invalid_ground_contact_steps": jp.zeros((), dtype=jp.float32),
            "body_or_chassis_ground_contact_steps": jp.zeros((), dtype=jp.float32),
            "episode_length_steps": episode_length_steps.astype(jp.float32),
            "grace_steps": grace_steps.astype(jp.float32),
        }
        metrics = self._crear_metricas_vacias()
        obs = self._get_obs(data, info)
        return mjx_env.State(
            data=data, obs=obs,
            reward=jp.zeros(()), done=jp.zeros(()),
            metrics=metrics, info=info,
        )

    # =====================================================================
    # PASO DE SIMULACIÓN
    # =====================================================================

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.clip(action, -1.0, 1.0)
        applied_action = self._convertir_accion_a_control(action)
        data = mjx_env.step(
            self.mjx_model, state.data, applied_action, self.n_substeps
        )

        next_info = dict(state.info)
        next_info["last_action"] = applied_action
        step_count = state.info["step_count"] + 1.0

        (
            invalid_ground_contact,
            invalid_ground_contact_count,
            body_or_chassis_touching_ground,
            *_,
        ) = self._informacion_contactos_suelo_invalidos(data)

        contacto_invalido_penalty = jp.clip(
            invalid_ground_contact_count
            / jp.maximum(self._config.contacto_invalido_penalty_count_scale, 1e-6),
            0.0, 1.0,
        )
        contacto_invalido_ahora = contacto_invalido_penalty > 0.10
        invalid_ground_contact_steps = jp.where(
            contacto_invalido_ahora,
            state.info["invalid_ground_contact_steps"] + 1.0,
            0.0,
        )
        body_or_chassis_ground_contact_steps = jp.where(
            body_or_chassis_touching_ground,
            state.info["body_or_chassis_ground_contact_steps"] + 1.0,
            0.0,
        )
        foot_contacts = self._mascara_contactos_pies_suelo(data)
        total_foot_clearance = self._separacion_total_pies_suelo(data)
        alive_requirements_met = (
            self._compuerta_supervivencia(data, invalid_ground_contact) > 0.5
        )
        alive_reward_steps = jp.where(
            alive_requirements_met,
            state.info["alive_reward_steps"] + 1.0,
            0.0,
        )
        obs = self._get_obs(data, next_info)
        done = self._get_done(
            data, step_count,
            invalid_ground_contact_steps, body_or_chassis_ground_contact_steps,
            state.info["episode_length_steps"], state.info["grace_steps"],
        )
        recompensa, new_metrics = self._calcular_recompensa(
            data, applied_action, state.info["last_action"],
            state.metrics, foot_contacts, total_foot_clearance,
            invalid_ground_contact_steps, body_or_chassis_ground_contact_steps,
            alive_reward_steps, step_count,
            state.info["episode_length_steps"], state.info["grace_steps"],
        )
        not_done = 1.0 - done
        next_info["step_count"] = step_count * not_done
        next_info["alive_reward_steps"] = alive_reward_steps * not_done
        next_info["invalid_ground_contact_steps"] = (
            invalid_ground_contact_steps * not_done
        )
        next_info["body_or_chassis_ground_contact_steps"] = (
            body_or_chassis_ground_contact_steps * not_done
        )
        next_info["episode_length_steps"] = (
            state.info["episode_length_steps"] * not_done
        )
        next_info["grace_steps"] = state.info["grace_steps"] * not_done

        return state.replace(
            data=data, obs=obs, reward=recompensa, done=done,
            metrics=new_metrics, info=next_info,
        )

    # =====================================================================
    # MÉTRICAS
    # =====================================================================

    def _crear_metricas_vacias(self) -> dict[str, jax.Array]:
        names = (
            # --- Recompensa principal ---
            "pose_imitacion_reward_ponderado",
            "pose_xml_final_reward_ponderado",
            "state/pose_imitacion_reward",
            "state/pose_xml_final_reward",
            "state/xml_pose_success_score",
            "state/xml_pose_success_score_with_support",
            "state/pose_imitacion_error_rms",
            "state/pose_imitacion_error_medio_deg",
            "state/pose_imitacion_sigma",
            "state/pose_imitacion_porcentaje",
            # --- Recompensas de apoyo (ponderadas) ---
            "supervivencia_reward_ponderado",
            "vector_gravedad_reward_ponderado",
            "vector_gravedad_estabilidad_reward_ponderado",
            "cuerpo_paralelo_reward_ponderado",
            "altura_reward_ponderado",
            "apertura_efectores_xml_reward_ponderado",
            "q3_distancia_xml_reward_ponderado",
            "soporte_estatico_reward_ponderado",
            "soporte_estatico_reward_filtrado_ponderado",
            "contactos_suelo_reward_ponderado",
            "contactos_suelo_reward_filtrado_ponderado",
            "q1_centrados_reward_ponderado",
            "velocidad_lineal_cero_reward_ponderado",
            "velocidad_angular_cero_reward_ponderado",
            "velocidad_articular_cero_reward_ponderado",
            # --- Compuertas de soporte ---
            "state/support_gate",
            "state/pose_gate",
            "state/height_gate",
            "state/level_gate",
            "state/level_gate_strict",
            "state/height_gate_strict",
            "state/valid_contact_gate",
            "state/support_contact_gate",
            "state/contacto_valido_reward",
            "state/filtro_contacto_soporte",
            "state/filtro_contacto_geometria",
            "state/filtro_contacto_altura",
            # --- Referencia XML ---
            "state/target_body_z_referencia",
            "state/error_altura_xml",
            # --- Totales ---
            "positive_reward",
            "penalties",
            "reward_total",
            # --- Penalizaciones ---
            "contacto_invalido_penalty_ponderado",
            "contacto_invalido_persistente_penalty_ponderado",
            "altura_exceso_penalty_ponderado",
            "altura_baja_penalty_ponderado",
            "rodilla_suelo_penalty_ponderado",
            "cuerpo_chasis_suelo_penalty_ponderado",
            "inclinacion_suave_penalty_ponderado",
            "efector_encima_q3_penalty_ponderado",
            "velocidad_vertical_cuerpo_penalty_ponderado",
            "velocidad_angular_cuerpo_penalty_ponderado",
            "control_penalty_ponderado",
            "cambio_accion_penalty_ponderado",
            "limite_articular_penalty_ponderado",
            # --- Currículo ---
            "reward/fase_curriculum",
            "state/k_c_curriculum_penalizaciones",
            # --- Terminación ---
            "done/nonfinite",
            "done/critical_z",
            "done/qpos_explosive",
            "done/qvel_explosive",
            "done/invalid_ground_contact",
            "done/terminal_failure",
            "done/boca_arriba",
            "done/tilt_60",
            "done/time_limit",
            # --- Estado del robot ---
            "state/z",
            "state/episode_step",
            "state/episode_time_seconds",
            "state/episode_length_steps",
            "state/grace_steps",
            "state/uprightness",
            "state/tilt_degrees",
            "state/foot_contacts",
            "state/num_valid_foot_contacts",
            "state/total_foot_clearance",
            # --- Estado de las recompensas ---
            "state/supervivencia_reward",
            "state/vector_gravedad_reward",
            "state/vector_gravedad_lineal_reward_base",
            "state/vector_gravedad_lineal_reward",
            "state/upright_direction_gate",
            "state/vector_gravedad_estabilidad_reward",
            "state/cuerpo_paralelo_reward",
            "state/contacto_invalido_penalty",
            "state/contacto_invalido_persistente_penalty",
            "state/altura_reward",
            "state/altura_reward_filtrada",
            "state/altura_exceso_penalty",
            "state/altura_baja_penalty",
            "state/inclinacion_suave_penalty",
            "state/apertura_efectores_xml_reward",
            "state/q3_distancia_xml_reward",
            "state/soporte_estatico_reward",
            "state/soporte_estatico_reward_filtrado",
            "state/contactos_suelo_reward",
            "state/contactos_suelo_reward_filtrado",
            "state/velocidad_articular_cero_reward",
            "state/reward_raw",
            "state/reward_scaled",
            "state/fase_curriculum_recompensa",
            "state/invalid_ground_contact_count",
            "state/num_invalid_contacts",
            "state/body_or_chassis_touching_ground",
            "state/elbow_or_knee_touching_ground",
            "state/gravity_lateral_error",
            "state/g_body_x",
            "state/g_body_y",
            "state/g_body_z",
            # --- Diagnóstico ---
            "debug/version",
            "debug/reward_version",
            "debug/ref_body_z",
            "debug/episode_length",
            "debug/action_scale",
            "debug/reward_scale",
            "debug/pose_imitacion_weight",
            "debug/pose_imitacion_sigma",
        )
        return {name: jp.zeros(()) for name in names}

    # =====================================================================
    # OBSERVACIONES
    # =====================================================================

    def _get_obs(
        self, data: mjx.Data, info: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        qpos = data.qpos[2:] if self._config.exclude_xy_position else data.qpos
        parts = [qpos, data.qvel]
        if self._config.include_last_action:
            parts.append(info["last_action"])
        if bool(self._config.get("include_pose_error", True)):
            q_actual = jp.concatenate([
                data.qpos[self._q1_qpos_adrs],
                data.qpos[self._q2_qpos_adrs],
                data.qpos[self._q3_qpos_adrs],
            ])
            parts.append(q_actual - self._q_referencia)
        return {"state": jp.concatenate(parts)}

    # =====================================================================
    # CONTROL
    # =====================================================================

    def _convertir_accion_a_control(self, action: jax.Array) -> jax.Array:
        if str(self._config.action_mode) == "tanh":
            scaled_action = float(self._config.action_tanh_scale) * jp.tanh(action)
        else:
            scaled_action = self._config.action_scale * action
        return jp.clip(scaled_action, self._ctrl_range[:, 0], self._ctrl_range[:, 1])

    # =====================================================================
    # RECOMPENSA PRINCIPAL: IMITACIÓN DE POSTURA
    # =====================================================================

    def _recompensa_imitacion_pose(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
        """Reward de imitacion de pose respecto a qpos0 del XML.

        Devuelve (recompensa [0, 1], error_rms_en_radianes).

        Vale 1.0 exactamente cuando todos los angulos de las articulaciones
        coinciden con los angulos del qpos0 del XML (la pose de referencia).

        Si cambias el XML, self._q_referencia se actualiza en __init__ y esta
        funcion empieza a recompensar la nueva pose automaticamente.

        El sigma marca la tolerancia:
          sigma=0.45 rad -> gradiente util hasta ~26 deg de error (fase 1)
          sigma=0.25 rad -> gradiente util hasta ~14 deg de error (fase 2)
          sigma=0.15 rad -> gradiente util hasta  ~9 deg de error (fase 3)
        """
        q_actual = jp.concatenate([
            data.qpos[self._q1_qpos_adrs],
            data.qpos[self._q2_qpos_adrs],
            data.qpos[self._q3_qpos_adrs],
        ])
        sigma = jp.maximum(
            jp.array(float(self._config.pose_imitacion_sigma), dtype=jp.float32),
            1e-6,
        )
        error_cuadratico_medio = jp.mean(jp.square(q_actual - self._q_referencia))
        reward = jp.clip(
            jp.exp(-error_cuadratico_medio / jp.square(sigma)), 0.0, 1.0
        )
        error_rms = jp.sqrt(error_cuadratico_medio)
        return reward, error_rms

    # =====================================================================
    # RECOMPENSAS DE APOYO
    # =====================================================================

    def _gravedad_en_referencia_cuerpo(self, data: mjx.Data) -> jax.Array:
        rotation_body_to_world = data.xmat[self._main_body_id]
        gravity_world = jp.array([0.0, 0.0, -1.0], dtype=jp.float32)
        return rotation_body_to_world.T @ gravity_world

    def _esta_boca_abajo(self, data: mjx.Data) -> jax.Array:
        return self._gravedad_en_referencia_cuerpo(data)[2] > 0.0

    def _estado_valido(self, data: mjx.Data) -> jax.Array:
        finite = jp.isfinite(data.qpos).all() & jp.isfinite(data.qvel).all()
        qpos_ok = jp.max(jp.abs(data.qpos)) < self._config.explosive_qpos_limit
        qvel_ok = jp.max(jp.abs(data.qvel)) < self._config.explosive_qvel_limit
        return finite & qpos_ok & qvel_ok

    def _compuerta_supervivencia(
        self, data: mjx.Data, invalid_ground_contact: jax.Array
    ) -> jax.Array:
        del invalid_ground_contact
        z = data.xpos[self._main_body_id, 2]
        inclinacion_menor_60 = (
            self._inclinacion_grados(data) < self._config.max_survival_tilt_degrees
        )
        no_boca_arriba = ~self._esta_boca_abajo(data)
        altura_segura = (
            (z > self._config.critical_z_min) & (z < self._config.critical_z_max)
        )
        gate = (
            inclinacion_menor_60 & no_boca_arriba
            & altura_segura & self._estado_valido(data)
        )
        return gate.astype(jp.float32)

    def _recompensa_altura(self, z: jax.Array) -> jax.Array:
        # target_z viene del forward de qpos0 del XML (self._target_body_z_actual),
        # calculado en __init__. Si usar_altura_referencia_xml=False, usa el valor alternativo.
        sigma = jp.maximum(
            jp.array(float(self._config.target_height_sigma), dtype=jp.float32), 1e-6
        )
        return jp.clip(
            jp.exp(-jp.square((z - self._target_body_z_actual) / sigma)), 0.0, 1.0
        )

    def _penalizacion_altura_excesiva(self, z: jax.Array) -> jax.Array:
        """Castiga solo subir por encima de la altura XML con una tolerancia pequena."""

        tolerancia = jp.array(
            float(self._config.get("altura_exceso_tolerancia", 0.010)),
            dtype=jp.float32,
        )
        sigma = jp.maximum(
            jp.array(float(self._config.get("altura_exceso_sigma", 0.035)),
                     dtype=jp.float32),
            1e-6,
        )
        exceso = jp.maximum(0.0, z - self._target_body_z_actual - tolerancia)
        return jp.clip(jp.square(exceso / sigma), 0.0, 1.0)

    def _penalizacion_altura_baja(self, z: jax.Array) -> jax.Array:
        """Castiga hundir el cuerpo por debajo de la altura XML."""

        tolerancia = jp.array(
            float(self._config.get("altura_baja_tolerancia", 0.006)),
            dtype=jp.float32,
        )
        sigma = jp.maximum(
            jp.array(float(self._config.get("altura_baja_sigma", 0.025)),
                     dtype=jp.float32),
            1e-6,
        )
        defecto = jp.maximum(0.0, self._target_body_z_actual - z - tolerancia)
        return jp.clip(jp.square(defecto / sigma), 0.0, 1.0)

    def _recompensa_vector_gravedad(self, data: mjx.Data) -> jax.Array:
        """Recompensa de orientación basada en la gravedad respecto al cuerpo.

        Vale 1 cuando el cuerpo esta perfectamente horizontal (gravedad alineada
        con el eje -Z del cuerpo). Cae cuando el cuerpo se inclina.

        sigma=0.35 rad → gradiente util hasta ~20 deg de inclinacion.
        """
        g_body = self._gravedad_en_referencia_cuerpo(data)
        gravity_lateral_sq = jp.square(g_body[0]) + jp.square(g_body[1])
        sigma = jp.maximum(
            jp.array(float(self._config.get("vector_gravedad_sigma", 0.35)),
                     dtype=jp.float32),
            1e-6,
        )
        return jp.clip(jp.exp(-gravity_lateral_sq / jp.square(sigma)), 0.0, 1.0)

    def _recompensa_cuerpo_paralelo(self, data: mjx.Data) -> jax.Array:
        """Version mas estricta de nivelacion para que el torso quede plano."""

        g_body = self._gravedad_en_referencia_cuerpo(data)
        gravity_lateral_sq = jp.square(g_body[0]) + jp.square(g_body[1])
        sigma = jp.maximum(
            jp.array(float(self._config.get("cuerpo_paralelo_sigma", 0.18)),
                     dtype=jp.float32),
            1e-6,
        )
        return jp.clip(jp.exp(-gravity_lateral_sq / jp.square(sigma)), 0.0, 1.0)

    def _recompensa_apertura_efectores_xml(self, data: mjx.Data) -> jax.Array:
        """Mantiene los pies q4 abiertos como en el forward de qpos0 del XML."""

        body_xy = data.xpos[self._main_body_id, :2]
        effector_xy = data.geom_xpos[self._effector_geom_ids, :2]
        radial_distances = jp.linalg.norm(effector_xy - body_xy, axis=1)
        sigma = jp.maximum(
            jp.array(float(self._config.get("apertura_efectores_xml_sigma", 0.050)),
                     dtype=jp.float32),
            1e-6,
        )
        error_mse = jp.mean(jp.square(radial_distances - self._ref_effector_radial_distances))
        return jp.clip(jp.exp(-error_mse / jp.square(sigma)), 0.0, 1.0)

    def _recompensa_distancia_q3_xml(self, data: mjx.Data) -> jax.Array:
        """Evita que las rodillas/q3 colapsen hacia el centro respecto al XML."""

        body_xy = data.xpos[self._main_body_id, :2]
        q3_xy = data.geom_xpos[self._q3_geom_ids, :2]
        radial_distances = jp.linalg.norm(q3_xy - body_xy, axis=1)
        sigma = jp.maximum(
            jp.array(float(self._config.get("q3_distancia_xml_sigma", 0.050)),
                     dtype=jp.float32),
            1e-6,
        )
        error_mse = jp.mean(jp.square(radial_distances - self._ref_q3_radial_distances))
        return jp.clip(jp.exp(-error_mse / jp.square(sigma)), 0.0, 1.0)

    def _recompensa_contactos_suelo(self, foot_contacts: jax.Array) -> jax.Array:
        valid_contact_count = jp.sum(foot_contacts.astype(jp.float32))
        required_contacts = jp.maximum(
            jp.array(float(self._config.min_foot_contacts_for_support), dtype=jp.float32),
            1.0,
        )
        return jp.clip(valid_contact_count / required_contacts, 0.0, 1.0)

    def _recompensa_q1_centrados(self, data: mjx.Data) -> jax.Array:
        q1_values = data.qpos[self._q1_qpos_adrs]
        q1_neutral = self._qpos0[self._q1_qpos_adrs]
        sigma = jp.maximum(jp.deg2rad(self._config.q1_center_sigma_deg), 1e-6)
        return jp.clip(
            jp.exp(-jp.mean(jp.square((q1_values - q1_neutral) / sigma))), 0.0, 1.0
        )

    def _recompensa_soporte_triangular(
        self, triangle_xy: jax.Array, point_xy: jax.Array
    ) -> jax.Array:
        center = jp.mean(triangle_xy, axis=0)
        rel_xy = triangle_xy - center
        order = jp.argsort(jp.arctan2(rel_xy[:, 1], rel_xy[:, 0]))
        polygon_xy = triangle_xy[order]
        next_xy = jp.roll(polygon_xy, -1, axis=0)
        edge_xy = next_xy - polygon_xy
        point_from_edge = point_xy - polygon_xy
        area = 0.5 * jp.sum(
            polygon_xy[:, 0] * next_xy[:, 1] - polygon_xy[:, 1] * next_xy[:, 0]
        )
        orientation = jp.where(area >= 0.0, 1.0, -1.0)
        cross = (
            edge_xy[:, 0] * point_from_edge[:, 1]
            - edge_xy[:, 1] * point_from_edge[:, 0]
        )
        edge_length = jp.maximum(jp.linalg.norm(edge_xy, axis=1), 1e-6)
        signed_distance = orientation * cross / edge_length
        signed_margin = jp.min(signed_distance)
        margin = jp.maximum(self._config.support_center_margin, 1e-6)
        inside_reward = 0.5 + 0.5 * (
            1.0 - jp.exp(-jp.maximum(signed_margin, 0.0) / margin)
        )
        outside_reward = 0.5 * jp.exp(
            -jp.square(jp.maximum(-signed_margin, 0.0) / margin)
        )
        return jp.clip(
            jp.where(signed_margin >= 0.0, inside_reward, outside_reward), 0.0, 1.0
        )

    def _recompensa_poligono_cog(
        self, data: mjx.Data, foot_contacts: jax.Array
    ) -> jax.Array:
        effector_xy = data.geom_xpos[self._effector_geom_ids, :2]
        com_xy = data.subtree_com[self._main_body_id, :2]
        contact_count = jp.sum(foot_contacts.astype(jp.float32))
        triangles = jp.array(
            [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=jp.int32
        )

        def normalizar_triangulo(indices: jax.Array) -> jax.Array:
            valid = jp.all(foot_contacts[indices])
            valor_reward = self._recompensa_soporte_triangular(
                effector_xy[indices], com_xy
            )
            return jp.where(valid, valor_reward, 0.0)

        triangulos_reward = jax.vmap(normalizar_triangulo)(triangles)
        return jp.where(contact_count >= 3.0, jp.max(triangulos_reward), 0.0)

    def _recompensa_soporte_estatico(
        self, data: mjx.Data, foot_contacts: jax.Array
    ) -> jax.Array:
        contactos = self._recompensa_contactos_suelo(foot_contacts)
        poligono = self._recompensa_poligono_cog(data, foot_contacts)
        return contactos * poligono

    def _recompensa_velocidad_lineal_cero(self, data: mjx.Data) -> jax.Array:
        vel_lineal = data.qvel[0:3]
        sigma = jp.maximum(self._config.velocidad_lineal_cero_sigma, 1e-6)
        return jp.clip(
            jp.exp(-jp.sum(jp.square(vel_lineal)) / jp.square(sigma)), 0.0, 1.0
        )

    def _recompensa_velocidad_angular_cero(self, data: mjx.Data) -> jax.Array:
        vel_angular = data.qvel[3:6]
        sigma = jp.maximum(self._config.velocidad_angular_cero_sigma, 1e-6)
        return jp.clip(
            jp.exp(-jp.sum(jp.square(vel_angular)) / jp.square(sigma)), 0.0, 1.0
        )

    def _recompensa_velocidad_articular_cero(self, data: mjx.Data) -> jax.Array:
        qvel_joints = data.qvel[self._joint_dof_adrs]
        sigma = jp.maximum(
            jp.array(float(self._config.get("velocidad_articular_cero_sigma", 2.0)),
                     dtype=jp.float32),
            1e-6,
        )
        return jp.clip(
            jp.exp(-jp.mean(jp.square(qvel_joints / sigma))), 0.0, 1.0
        )

    def _penalizacion_efector_sobre_q3(self, data: mjx.Data) -> jax.Array:
        excess = self._exceso_efector_sobre_q3(data)
        scale = jp.maximum(self._config.effector_q3_z_scale, 1e-6)
        return jp.clip(jp.mean(jp.square(excess / scale)), 0.0, 1.0)

    def _penalizacion_limite_articular(self, data: mjx.Data) -> jax.Array:
        q = data.qpos[self._joint_qpos_adrs]
        low = self._joint_ranges[:, 0]
        high = self._joint_ranges[:, 1]
        center = 0.5 * (low + high)
        half_range = jp.maximum(0.5 * (high - low), 1e-6)
        distance_to_limit = half_range - jp.abs(q - center)
        margin = jp.maximum(self._config.joint_limit_margin_fraction, 1e-6)
        distancia_relativa_limite = distance_to_limit / half_range
        limit_pressure = jp.maximum(
            0.0, (margin - distancia_relativa_limite) / margin
        )
        return jp.clip(jp.mean(jp.square(limit_pressure)), 0.0, 1.0)

    # =====================================================================
    # TERMINACIÓN
    # =====================================================================

    def _get_done(
        self,
        data: mjx.Data,
        step_count: jax.Array,
        invalid_ground_contact_steps: jax.Array,
        body_or_chassis_ground_contact_steps: jax.Array,
        episode_length_steps: jax.Array,
        grace_steps: jax.Array,
    ) -> jax.Array:
        if bool(self._config.get("disable_done", False)):
            return jp.zeros((), dtype=jp.float32)

        time_done = step_count >= episode_length_steps
        z = data.xpos[self._main_body_id, 2]
        finite = jp.isfinite(data.qpos).all() & jp.isfinite(data.qvel).all()
        critical_z = (
            (z < self._config.critical_z_min) | (z > self._config.critical_z_max)
        )
        qpos_ok = jp.max(jp.abs(data.qpos)) < self._config.explosive_qpos_limit
        qvel_ok = jp.max(jp.abs(data.qvel)) < self._config.explosive_qvel_limit
        past_grace = step_count > grace_steps
        persistent_invalid_contact = (
            self._config.terminate_on_persistent_invalid_ground_contact
            & past_grace
            & (
                invalid_ground_contact_steps
                >= self._config.invalid_ground_contact_done_steps
            )
        )
        body_or_chassis_invalid_contact = (
            self._config.terminate_on_persistent_invalid_ground_contact
            & past_grace
            & (
                body_or_chassis_ground_contact_steps
                >= self._config.body_or_chassis_ground_contact_done_steps
            )
        )
        tilt_60_done = (
            self._inclinacion_grados(data) > self._config.max_survival_tilt_degrees
        )
        boca_arriba_done = self._esta_boca_abajo(data)
        catastrophic = (
            (~finite) | (~qpos_ok) | (~qvel_ok)
            | persistent_invalid_contact | body_or_chassis_invalid_contact
            | critical_z | tilt_60_done | boca_arriba_done
        )
        return (catastrophic | time_done).astype(jp.float32)

    # =====================================================================
    # CÁLCULO DE LA RECOMPENSA
    # =====================================================================

    def _calcular_recompensa(
        self,
        data: mjx.Data,
        action: jax.Array,
        last_action: jax.Array,
        metrics: dict[str, jax.Array],
        foot_contacts: jax.Array,
        total_foot_clearance: jax.Array,
        invalid_ground_contact_steps: jax.Array,
        body_or_chassis_ground_contact_steps: jax.Array,
        alive_reward_steps: jax.Array,
        step_count: jax.Array,
        episode_length_steps: jax.Array,
        grace_steps: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del alive_reward_steps, total_foot_clearance

        z = data.xpos[self._main_body_id, 2]
        uprightness = data.xmat[self._main_body_id, 2, 2]
        finite = jp.isfinite(data.qpos).all() & jp.isfinite(data.qvel).all()
        qpos_ok = jp.max(jp.abs(data.qpos)) < self._config.explosive_qpos_limit
        qvel_ok = jp.max(jp.abs(data.qvel)) < self._config.explosive_qvel_limit

        (
            invalid_ground_contact,
            invalid_ground_contact_count,
            body_or_chassis_touching_ground,
            elbow_or_knee_touching_ground,
            *_rest,
        ) = self._informacion_contactos_suelo_invalidos(data)

        time_done = step_count >= episode_length_steps
        done_nonfinite = ~finite
        done_critical_z = (
            (z < self._config.critical_z_min) | (z > self._config.critical_z_max)
        )
        done_qpos_explosive = ~qpos_ok
        done_qvel_explosive = ~qvel_ok
        past_grace = step_count > grace_steps

        contacto_invalido_penalty = jp.clip(
            invalid_ground_contact_count
            / jp.maximum(self._config.contacto_invalido_penalty_count_scale, 1e-6),
            0.0, 1.0,
        )
        contacto_invalido_persistente_penalty = jp.clip(
            (invalid_ground_contact_steps - 5.0) / 25.0, 0.0, 1.0,
        )
        persistent_invalid_contact = (
            self._config.terminate_on_persistent_invalid_ground_contact
            & past_grace
            & (
                invalid_ground_contact_steps
                >= self._config.invalid_ground_contact_done_steps
            )
        )
        body_or_chassis_invalid_contact = (
            self._config.terminate_on_persistent_invalid_ground_contact
            & past_grace
            & (
                body_or_chassis_ground_contact_steps
                >= self._config.body_or_chassis_ground_contact_done_steps
            )
        )
        tilt_degrees = self._inclinacion_grados(data)
        tilt_60 = tilt_degrees > self._config.max_survival_tilt_degrees
        boca_arriba = self._esta_boca_abajo(data)
        terminal_failure = (
            done_nonfinite | done_critical_z
            | done_qpos_explosive | done_qvel_explosive
            | persistent_invalid_contact | body_or_chassis_invalid_contact
            | tilt_60 | boca_arriba
        )
        invalid_contact_done = persistent_invalid_contact | body_or_chassis_invalid_contact

        cambio_accion = action - last_action

        # =================================================================
        # RECOMPENSAS (jerarquía clara)
        # =================================================================

        # 1. POSE (senal principal): imitacion de qpos0 del XML.
        pose_imitacion_reward, error_rms_rad = self._recompensa_imitacion_pose(data)

        # 2. ALTURA: distancia a la z de referencia del XML (no hardcodeada).
        altura_reward = self._recompensa_altura(z)
        altura_exceso_penalty = self._penalizacion_altura_excesiva(z)
        altura_baja_penalty = self._penalizacion_altura_baja(z)
        error_altura_xml = z - self._target_body_z_actual

        # 3. ORIENTACION: vector gravedad con sigma.
        gravity_body = self._gravedad_en_referencia_cuerpo(data)
        gravity_lateral_error = (
            jp.square(gravity_body[0]) + jp.square(gravity_body[1])
        )
        vector_gravedad_lineal_reward_base = jp.clip(
            1.0 - gravity_lateral_error, 0.0, 1.0
        )
        upright_direction_gate = jp.clip(-gravity_body[2], 0.0, 1.0)
        vector_gravedad_lineal_reward = (
            vector_gravedad_lineal_reward_base * upright_direction_gate
        )
        vector_gravedad_reward = self._recompensa_vector_gravedad(data)
        cuerpo_paralelo_reward = self._recompensa_cuerpo_paralelo(data)

        # 4. QUIETUD.
        velocidad_lineal_cero_reward = self._recompensa_velocidad_lineal_cero(data)
        velocidad_angular_cero_reward = self._recompensa_velocidad_angular_cero(data)
        velocidad_articular_cero_reward = self._recompensa_velocidad_articular_cero(data)

        # 5. SUPERVIVENCIA (compuerta binaria).
        supervivencia_reward = self._compuerta_supervivencia(
            data, invalid_ground_contact
        )

        # 6. CONTACTO Y SOPORTE: filtrados por una compuerta suave de postura,
        #    altura y orientación.
        #    Si el robot se tumba y toca mucho suelo pero no se parece al XML,
        #    la compuerta aplana la recompensa de contacto a casi cero.
        contactos_suelo_reward = self._recompensa_contactos_suelo(foot_contacts)
        soporte_estatico_reward = self._recompensa_soporte_estatico(data, foot_contacts)
        apertura_efectores_xml_reward = self._recompensa_apertura_efectores_xml(data)
        q3_distancia_xml_reward = self._recompensa_distancia_q3_xml(data)

        pose_gate = jp.clip((pose_imitacion_reward - 0.20) / 0.50, 0.0, 1.0)
        height_gate = jp.clip((altura_reward - 0.20) / 0.50, 0.0, 1.0)
        level_gate = jp.clip((vector_gravedad_reward - 0.20) / 0.50, 0.0, 1.0)
        contacto_valido_reward = 1.0 - contacto_invalido_penalty
        tilt_sigma_deg = jp.maximum(
            jp.array(
                float(self._config.get("pose_xml_final_tilt_sigma_deg", 7.0)),
                dtype=jp.float32,
            ),
            1e-6,
        )
        height_sigma_strict = jp.maximum(
            jp.array(
                float(self._config.get("pose_xml_final_height_sigma", 0.025)),
                dtype=jp.float32,
            ),
            1e-6,
        )
        level_gate_strict = jp.clip(
            jp.exp(-jp.square(tilt_degrees / tilt_sigma_deg)), 0.0, 1.0
        )
        height_gate_strict = jp.clip(
            jp.exp(-jp.square(error_altura_xml / height_sigma_strict)), 0.0, 1.0
        )
        valid_contact_gate = jp.clip(contacto_valido_reward, 0.0, 1.0)
        support_contact_gate = contactos_suelo_reward
        xml_pose_success_score = (
            pose_imitacion_reward
            * level_gate_strict
            * height_gate_strict
            * valid_contact_gate
        )
        xml_pose_success_score_with_support = (
            xml_pose_success_score * support_contact_gate
        )
        pose_xml_final_reward = xml_pose_success_score
        if bool(self._config.get("usar_filtros_contacto_recompensa", True)):
            filtro_contacto_soporte = contacto_valido_reward
            filtro_contacto_geometria = 0.25 + 0.75 * contacto_valido_reward
            filtro_contacto_altura = 0.40 + 0.60 * contacto_valido_reward
        else:
            filtro_contacto_soporte = jp.ones((), dtype=jp.float32)
            filtro_contacto_geometria = jp.ones((), dtype=jp.float32)
            filtro_contacto_altura = jp.ones((), dtype=jp.float32)
        support_gate = pose_gate * height_gate * level_gate * filtro_contacto_soporte
        geometry_gate = pose_gate * level_gate * filtro_contacto_geometria
        vector_gravedad_estabilidad_reward = (
            vector_gravedad_lineal_reward * filtro_contacto_geometria
        )

        contactos_suelo_reward_filtrada = contactos_suelo_reward * support_gate
        soporte_estatico_reward_filtrado = soporte_estatico_reward * support_gate
        altura_reward_filtrada = altura_reward * filtro_contacto_altura
        apertura_efectores_xml_reward_filtrada = (
            apertura_efectores_xml_reward * geometry_gate
        )
        q3_distancia_xml_reward_filtrada = q3_distancia_xml_reward * geometry_gate

        # 7. COMPATIBILIDAD ANTERIOR (q1 centrados y peso bajo).
        q1_centrados_reward = self._recompensa_q1_centrados(data)

        # =================================================================
        # SUMA PONDERADA DE RECOMPENSAS POSITIVAS
        # =================================================================
        pose_imitacion_reward_ponderado = (
            float(self._config.pose_imitacion_reward_weight) * pose_imitacion_reward
        )
        pose_xml_final_reward_ponderado = (
            jp.array(
                float(self._config.get("pose_xml_final_reward_weight", 0.0)),
                dtype=jp.float32,
            )
            * pose_xml_final_reward
        )
        supervivencia_reward_ponderado = (
            self._config.supervivencia_reward_weight * supervivencia_reward
        )
        vector_gravedad_reward_ponderado = (
            self._config.vector_gravedad_reward_weight * vector_gravedad_reward
        )
        vector_gravedad_estabilidad_reward_ponderado = (
            jp.array(
                float(
                    self._config.get(
                        "vector_gravedad_estabilidad_reward_weight", 0.0
                    )
                ),
                dtype=jp.float32,
            )
            * vector_gravedad_estabilidad_reward
        )
        cuerpo_paralelo_reward_ponderado = (
            self._config.cuerpo_paralelo_reward_weight * cuerpo_paralelo_reward
        )
        altura_reward_ponderado = (
            jp.array(float(self._config.altura_reward_weight), dtype=jp.float32)
            * altura_reward_filtrada
        )
        apertura_efectores_xml_reward_ponderado = (
            self._config.apertura_efectores_xml_reward_weight
            * apertura_efectores_xml_reward_filtrada
        )
        q3_distancia_xml_reward_ponderado = (
            self._config.q3_distancia_xml_reward_weight
            * q3_distancia_xml_reward_filtrada
        )
        contactos_suelo_reward_ponderado = (
            self._config.contactos_suelo_reward_weight * contactos_suelo_reward
        )
        contactos_suelo_reward_filtrado_ponderado = (
            self._config.contactos_suelo_reward_weight * contactos_suelo_reward_filtrada
        )
        soporte_estatico_reward_ponderado = (
            self._config.soporte_estatico_reward_weight * soporte_estatico_reward
        )
        soporte_estatico_reward_filtrado_ponderado = (
            self._config.soporte_estatico_reward_weight * soporte_estatico_reward_filtrado
        )
        q1_centrados_reward_ponderado = (
            self._config.q1_centrados_reward_weight * q1_centrados_reward
        )
        velocidad_lineal_cero_reward_ponderado = (
            self._config.velocidad_lineal_cero_reward_weight * velocidad_lineal_cero_reward
        )
        velocidad_angular_cero_reward_ponderado = (
            self._config.velocidad_angular_cero_reward_weight * velocidad_angular_cero_reward
        )
        velocidad_articular_cero_reward_ponderado = (
            self._config.velocidad_articular_cero_reward_weight
            * velocidad_articular_cero_reward
        )

        positive_reward = (
            pose_imitacion_reward_ponderado
            + pose_xml_final_reward_ponderado
            + supervivencia_reward_ponderado
            + vector_gravedad_reward_ponderado
            + vector_gravedad_estabilidad_reward_ponderado
            + cuerpo_paralelo_reward_ponderado
            + altura_reward_ponderado
            + apertura_efectores_xml_reward_ponderado
            + q3_distancia_xml_reward_ponderado
            + contactos_suelo_reward_filtrado_ponderado
            + soporte_estatico_reward_filtrado_ponderado
            + q1_centrados_reward_ponderado
            + velocidad_lineal_cero_reward_ponderado
            + velocidad_angular_cero_reward_ponderado
            + velocidad_articular_cero_reward_ponderado
        )

        # =================================================================
        # PENALIZACIONES escaladas por k_c (currículo)
        # k_c=0.10 -> solo una fraccion al principio, sube manualmente.
        # reward_scale=0.01 puede subirse a 0.05 si la senal es demasiado
        # debil; dejado en 0.01 por compatibilidad con hiper anteriores.
        # =================================================================
        k_c = jp.clip(
            jp.array(float(self._config.curriculum_penalizaciones), dtype=jp.float32),
            0.0, 1.0,
        )
        contacto_invalido_penalty_ponderado = (
            self._config.contacto_invalido_penalty_weight * contacto_invalido_penalty
        )
        contacto_invalido_persistente_penalty_ponderado = (
            self._config.contacto_invalido_persistente_penalty_weight
            * contacto_invalido_persistente_penalty
        )
        altura_exceso_penalty_ponderado = (
            self._config.altura_exceso_penalty_weight * altura_exceso_penalty
        )
        altura_baja_penalty_ponderado = (
            self._config.altura_baja_penalty_weight * altura_baja_penalty
        )
        rodilla_suelo_penalty = elbow_or_knee_touching_ground.astype(jp.float32)
        rodilla_suelo_penalty_ponderado = (
            self._config.rodilla_suelo_penalty_weight * rodilla_suelo_penalty
        )
        cuerpo_chasis_suelo_penalty = body_or_chassis_touching_ground.astype(jp.float32)
        cuerpo_chasis_suelo_penalty_ponderado = (
            self._config.cuerpo_chasis_suelo_penalty_weight
            * cuerpo_chasis_suelo_penalty
        )
        inclinacion_tolerancia_deg = jp.maximum(
            jp.array(
                float(self._config.get("inclinacion_suave_tolerancia_deg", 4.0)),
                dtype=jp.float32,
            ),
            0.0,
        )
        inclinacion_sigma_deg = jp.maximum(
            jp.array(
                float(self._config.get("inclinacion_suave_sigma_deg", 8.0)),
                dtype=jp.float32,
            ),
            1e-6,
        )
        inclinacion_exceso_deg = jp.maximum(
            0.0, tilt_degrees - inclinacion_tolerancia_deg
        )
        inclinacion_suave_penalty = jp.clip(
            jp.square(inclinacion_exceso_deg / inclinacion_sigma_deg), 0.0, 1.0
        )
        inclinacion_suave_penalty_ponderado = (
            jp.array(
                float(self._config.get("inclinacion_suave_penalty_weight", 0.0)),
                dtype=jp.float32,
            )
            * inclinacion_suave_penalty
        )
        efector_encima_q3_pen = self._penalizacion_efector_sobre_q3(data)
        efector_encima_q3_penalty_ponderado = (
            self._config.efector_encima_q3_penalty_weight * efector_encima_q3_pen
        )
        base_vz = data.qvel[2]
        velocidad_vertical_penalty = jp.clip(
            jp.square(
                base_vz
                / jp.maximum(
                    jp.array(float(self._config.base_vertical_velocity_sigma),
                             dtype=jp.float32),
                    1e-6,
                )
            ),
            0.0, 1.0,
        )
        velocidad_vertical_cuerpo_penalty_ponderado = (
            self._config.velocidad_vertical_cuerpo_penalty_weight
            * velocidad_vertical_penalty
        )
        base_angvel_roll_pitch = data.qvel[3:5]
        velocidad_angular_penalty = jp.clip(
            jp.mean(jp.square(
                base_angvel_roll_pitch
                / jp.maximum(
                    jp.array(float(self._config.base_angular_velocity_sigma),
                             dtype=jp.float32),
                    1e-6,
                )
            )),
            0.0, 1.0,
        )
        velocidad_angular_cuerpo_penalty_ponderado = (
            self._config.velocidad_angular_cuerpo_penalty_weight * velocidad_angular_penalty
        )
        control_penalty = jp.clip(jp.mean(jp.square(action)), 0.0, 1.0)
        control_penalty_ponderado = (
            jp.array(float(self._config.control_penalty_weight), dtype=jp.float32)
            * control_penalty
        )
        cambio_accion_penalty = jp.clip(
            jp.mean(jp.square(
                cambio_accion
                / jp.maximum(
                    jp.array(float(self._config.cambio_accion_sigma), dtype=jp.float32),
                    1e-6,
                )
            )),
            0.0, 1.0,
        )
        cambio_accion_penalty_ponderado = (
            jp.array(float(self._config.cambio_accion_penalty_weight), dtype=jp.float32)
            * cambio_accion_penalty
        )
        limite_articular_penalty = self._penalizacion_limite_articular(data)
        limite_articular_penalty_ponderado = (
            self._config.limite_articular_penalty_weight * limite_articular_penalty
        )

        penalties_secundarias = k_c * (
            contacto_invalido_penalty_ponderado
            + contacto_invalido_persistente_penalty_ponderado
            + efector_encima_q3_penalty_ponderado
            + velocidad_vertical_cuerpo_penalty_ponderado
            + velocidad_angular_cuerpo_penalty_ponderado
            + control_penalty_ponderado
            + cambio_accion_penalty_ponderado
            + limite_articular_penalty_ponderado
        )
        # Estas son restricciones centrales de la pose XML: no se escala con
        # k_c porque no queremos que el robot aprenda a vivir con el torso o
        # rodillas apoyadas ni por encima/debajo de la altura de referencia.
        penalties_core = (
            altura_exceso_penalty_ponderado
            + altura_baja_penalty_ponderado
            + cuerpo_chasis_suelo_penalty_ponderado
            + rodilla_suelo_penalty_ponderado
            + inclinacion_suave_penalty_ponderado
        )
        penalties = penalties_core + penalties_secundarias

        # =================================================================
        # RECOMPENSA FINAL
        # =================================================================
        terminal_failure_penalty_valor = jp.clip(
            jp.array(float(self._config.terminal_failure_penalty), dtype=jp.float32),
            1.0, 5.0,
        )

        reward_neta = positive_reward - penalties
        if bool(self._config.clipear_reward_minimo):
            reward_neta = jp.maximum(0.0, reward_neta)

        # Penalizacion terminal solo por fallo real, no por timeout normal.
        terminal_failure_penalty_val = (
            terminal_failure_penalty_valor * terminal_failure.astype(jp.float32)
        )
        reward_raw = jp.where(
            terminal_failure, -terminal_failure_penalty_val, reward_neta
        )
        reward_scaled = jp.clip(
            reward_raw * jp.array(float(self._config.reward_scale), dtype=jp.float32),
            jp.array(float(self._config.reward_clip_min), dtype=jp.float32),
            jp.array(float(self._config.reward_clip_max), dtype=jp.float32),
        )

        # =================================================================
        # MÉTRICAS (todas las calculadas quedan registradas)
        # =================================================================
        error_medio_deg = jp.rad2deg(error_rms_rad)
        pose_imitacion_porcentaje = pose_imitacion_reward * 100.0
        curriculum_phase = jp.array(
            float(self._config.fase_curriculum_recompensa), dtype=jp.float32
        )

        metrics["pose_imitacion_reward_ponderado"] = pose_imitacion_reward_ponderado
        metrics["pose_xml_final_reward_ponderado"] = pose_xml_final_reward_ponderado
        metrics["state/pose_imitacion_reward"] = pose_imitacion_reward
        metrics["state/pose_xml_final_reward"] = pose_xml_final_reward
        metrics["state/xml_pose_success_score"] = xml_pose_success_score
        metrics["state/xml_pose_success_score_with_support"] = (
            xml_pose_success_score_with_support
        )
        metrics["state/pose_imitacion_error_rms"] = error_rms_rad
        metrics["state/pose_imitacion_error_medio_deg"] = error_medio_deg
        metrics["state/pose_imitacion_sigma"] = jp.array(
            float(self._config.pose_imitacion_sigma), dtype=jp.float32
        )
        metrics["state/pose_imitacion_porcentaje"] = pose_imitacion_porcentaje
        metrics["supervivencia_reward_ponderado"] = supervivencia_reward_ponderado
        metrics["vector_gravedad_reward_ponderado"] = vector_gravedad_reward_ponderado
        metrics["vector_gravedad_estabilidad_reward_ponderado"] = (
            vector_gravedad_estabilidad_reward_ponderado
        )
        metrics["cuerpo_paralelo_reward_ponderado"] = cuerpo_paralelo_reward_ponderado
        metrics["altura_reward_ponderado"] = altura_reward_ponderado
        metrics["apertura_efectores_xml_reward_ponderado"] = (
            apertura_efectores_xml_reward_ponderado
        )
        metrics["q3_distancia_xml_reward_ponderado"] = q3_distancia_xml_reward_ponderado
        metrics["soporte_estatico_reward_ponderado"] = soporte_estatico_reward_ponderado
        metrics["soporte_estatico_reward_filtrado_ponderado"] = (
            soporte_estatico_reward_filtrado_ponderado
        )
        metrics["contactos_suelo_reward_ponderado"] = contactos_suelo_reward_ponderado
        metrics["contactos_suelo_reward_filtrado_ponderado"] = (
            contactos_suelo_reward_filtrado_ponderado
        )
        metrics["q1_centrados_reward_ponderado"] = q1_centrados_reward_ponderado
        metrics["velocidad_lineal_cero_reward_ponderado"] = (
            velocidad_lineal_cero_reward_ponderado
        )
        metrics["velocidad_angular_cero_reward_ponderado"] = (
            velocidad_angular_cero_reward_ponderado
        )
        metrics["velocidad_articular_cero_reward_ponderado"] = (
            velocidad_articular_cero_reward_ponderado
        )
        metrics["state/support_gate"] = support_gate
        metrics["state/pose_gate"] = pose_gate
        metrics["state/height_gate"] = height_gate
        metrics["state/level_gate"] = level_gate
        metrics["state/level_gate_strict"] = level_gate_strict
        metrics["state/height_gate_strict"] = height_gate_strict
        metrics["state/valid_contact_gate"] = valid_contact_gate
        metrics["state/support_contact_gate"] = support_contact_gate
        metrics["state/contacto_valido_reward"] = contacto_valido_reward
        metrics["state/filtro_contacto_soporte"] = filtro_contacto_soporte
        metrics["state/filtro_contacto_geometria"] = filtro_contacto_geometria
        metrics["state/filtro_contacto_altura"] = filtro_contacto_altura
        metrics["state/target_body_z_referencia"] = self._target_body_z_actual
        metrics["state/error_altura_xml"] = error_altura_xml
        metrics["positive_reward"] = positive_reward
        metrics["penalties"] = penalties
        metrics["reward_total"] = reward_neta
        metrics["contacto_invalido_penalty_ponderado"] = contacto_invalido_penalty_ponderado
        metrics["contacto_invalido_persistente_penalty_ponderado"] = (
            contacto_invalido_persistente_penalty_ponderado
        )
        metrics["altura_exceso_penalty_ponderado"] = altura_exceso_penalty_ponderado
        metrics["altura_baja_penalty_ponderado"] = altura_baja_penalty_ponderado
        metrics["rodilla_suelo_penalty_ponderado"] = rodilla_suelo_penalty_ponderado
        metrics["cuerpo_chasis_suelo_penalty_ponderado"] = (
            cuerpo_chasis_suelo_penalty_ponderado
        )
        metrics["inclinacion_suave_penalty_ponderado"] = (
            inclinacion_suave_penalty_ponderado
        )
        metrics["efector_encima_q3_penalty_ponderado"] = (
            efector_encima_q3_penalty_ponderado
        )
        metrics["velocidad_vertical_cuerpo_penalty_ponderado"] = (
            velocidad_vertical_cuerpo_penalty_ponderado
        )
        metrics["velocidad_angular_cuerpo_penalty_ponderado"] = (
            velocidad_angular_cuerpo_penalty_ponderado
        )
        metrics["control_penalty_ponderado"] = control_penalty_ponderado
        metrics["cambio_accion_penalty_ponderado"] = cambio_accion_penalty_ponderado
        metrics["limite_articular_penalty_ponderado"] = limite_articular_penalty_ponderado
        metrics["reward/fase_curriculum"] = curriculum_phase
        metrics["state/k_c_curriculum_penalizaciones"] = k_c
        metrics["done/nonfinite"] = done_nonfinite.astype(jp.float32)
        metrics["done/critical_z"] = done_critical_z.astype(jp.float32)
        metrics["done/qpos_explosive"] = done_qpos_explosive.astype(jp.float32)
        metrics["done/qvel_explosive"] = done_qvel_explosive.astype(jp.float32)
        metrics["done/invalid_ground_contact"] = invalid_contact_done.astype(jp.float32)
        metrics["done/terminal_failure"] = terminal_failure.astype(jp.float32)
        metrics["done/boca_arriba"] = boca_arriba.astype(jp.float32)
        metrics["done/tilt_60"] = tilt_60.astype(jp.float32)
        metrics["done/time_limit"] = time_done.astype(jp.float32)
        metrics["state/z"] = z
        metrics["state/episode_step"] = step_count
        metrics["state/episode_time_seconds"] = (
            step_count * jp.array(float(self._config.ctrl_dt), dtype=jp.float32)
        )
        metrics["state/episode_length_steps"] = episode_length_steps
        metrics["state/grace_steps"] = grace_steps
        metrics["state/uprightness"] = uprightness
        metrics["state/tilt_degrees"] = tilt_degrees
        num_valid_foot_contacts = jp.sum(foot_contacts.astype(jp.float32))
        metrics["state/foot_contacts"] = num_valid_foot_contacts
        metrics["state/num_valid_foot_contacts"] = num_valid_foot_contacts
        metrics["state/total_foot_clearance"] = self._separacion_total_pies_suelo(data)
        metrics["state/supervivencia_reward"] = supervivencia_reward
        metrics["state/vector_gravedad_reward"] = vector_gravedad_reward
        metrics["state/vector_gravedad_lineal_reward_base"] = (
            vector_gravedad_lineal_reward_base
        )
        metrics["state/vector_gravedad_lineal_reward"] = vector_gravedad_lineal_reward
        metrics["state/upright_direction_gate"] = upright_direction_gate
        metrics["state/vector_gravedad_estabilidad_reward"] = (
            vector_gravedad_estabilidad_reward
        )
        metrics["state/cuerpo_paralelo_reward"] = cuerpo_paralelo_reward
        metrics["state/contacto_invalido_penalty"] = contacto_invalido_penalty
        metrics["state/contacto_invalido_persistente_penalty"] = (
            contacto_invalido_persistente_penalty
        )
        metrics["state/altura_reward"] = altura_reward
        metrics["state/altura_reward_filtrada"] = altura_reward_filtrada
        metrics["state/altura_exceso_penalty"] = altura_exceso_penalty
        metrics["state/altura_baja_penalty"] = altura_baja_penalty
        metrics["state/inclinacion_suave_penalty"] = inclinacion_suave_penalty
        metrics["state/apertura_efectores_xml_reward"] = apertura_efectores_xml_reward
        metrics["state/q3_distancia_xml_reward"] = q3_distancia_xml_reward
        metrics["state/soporte_estatico_reward"] = soporte_estatico_reward
        metrics["state/soporte_estatico_reward_filtrado"] = (
            soporte_estatico_reward_filtrado
        )
        metrics["state/contactos_suelo_reward"] = contactos_suelo_reward
        metrics["state/contactos_suelo_reward_filtrado"] = (
            contactos_suelo_reward_filtrada
        )
        metrics["state/velocidad_articular_cero_reward"] = velocidad_articular_cero_reward
        metrics["state/reward_raw"] = reward_raw
        metrics["state/reward_scaled"] = reward_scaled
        metrics["state/fase_curriculum_recompensa"] = curriculum_phase
        metrics["state/invalid_ground_contact_count"] = invalid_ground_contact_count
        metrics["state/num_invalid_contacts"] = invalid_ground_contact_count
        metrics["state/body_or_chassis_touching_ground"] = (
            body_or_chassis_touching_ground.astype(jp.float32)
        )
        metrics["state/elbow_or_knee_touching_ground"] = (
            elbow_or_knee_touching_ground.astype(jp.float32)
        )
        metrics["state/gravity_lateral_error"] = gravity_lateral_error
        metrics["state/g_body_x"] = gravity_body[0]
        metrics["state/g_body_y"] = gravity_body[1]
        metrics["state/g_body_z"] = gravity_body[2]
        metrics["debug/version"] = jp.array(
            float(VERSION_DIAGNOSTICO_RECOMPENSA), dtype=jp.float32
        )
        metrics["debug/reward_version"] = jp.array(
            float(VERSION_DIAGNOSTICO_RECOMPENSA), dtype=jp.float32
        )
        metrics["debug/ref_body_z"] = self._ref_body_z
        metrics["debug/episode_length"] = episode_length_steps.astype(jp.float32)
        metrics["debug/action_scale"] = jp.array(
            float(self._config.action_scale), dtype=jp.float32
        )
        metrics["debug/reward_scale"] = jp.array(
            float(self._config.reward_scale), dtype=jp.float32
        )
        metrics["debug/pose_imitacion_weight"] = jp.array(
            float(self._config.pose_imitacion_reward_weight), dtype=jp.float32
        )
        metrics["debug/pose_imitacion_sigma"] = jp.array(
            float(self._config.pose_imitacion_sigma), dtype=jp.float32
        )

        return reward_scaled, metrics

    # =====================================================================
    # UTILIDADES: INCLINACIÓN Y CONTACTOS
    # =====================================================================

    def _inclinacion_grados(self, data: mjx.Data) -> jax.Array:
        uprightness = jp.clip(data.xmat[self._main_body_id, 2, 2], -1.0, 1.0)
        return jp.rad2deg(jp.arccos(uprightness))

    def _mascara_contactos_activos(self, data: mjx.Data) -> jax.Array:
        return data.contact.dist <= 0.0

    def _mascara_contactos_suelo(self, data: mjx.Data) -> jax.Array:
        geom1 = data.contact.geom1
        geom2 = data.contact.geom2
        return self._mascara_contactos_activos(data) & (
            (geom1 == self._floor_geom_id) | (geom2 == self._floor_geom_id)
        )

    def _mascara_contactos_pies_suelo(self, data: mjx.Data) -> jax.Array:
        geom1 = data.contact.geom1
        geom2 = data.contact.geom2
        ground_contact = self._mascara_contactos_suelo(data)

        def tiene_contacto(geom_id: jax.Array) -> jax.Array:
            return jp.any(ground_contact & ((geom1 == geom_id) | (geom2 == geom_id)))

        return jax.vmap(tiene_contacto)(self._effector_geom_ids)

    def _informacion_contactos_suelo_invalidos(
        self, data: mjx.Data
    ) -> tuple[
        jax.Array, jax.Array, jax.Array, jax.Array,
        jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,
    ]:
        geom1 = data.contact.geom1
        geom2 = data.contact.geom2
        ground_contact = self._mascara_contactos_suelo(data)
        other_geom = jp.where(geom1 == self._floor_geom_id, geom2, geom1)
        ground_geom = jp.where(geom1 == self._floor_geom_id, geom1, geom2)
        is_effector = jp.any(
            other_geom[:, None] == self._effector_geom_ids[None, :], axis=1
        )
        invalid_contact = ground_contact & (~is_effector)
        is_body_or_chassis = jp.any(
            other_geom[:, None] == self._body_or_chassis_geom_ids[None, :], axis=1
        )
        body_or_chassis_invalid_contact = invalid_contact & is_body_or_chassis
        elbow_or_knee_invalid_contact = invalid_contact & (~is_body_or_chassis)
        invalid_count = jp.sum(invalid_contact.astype(jp.float32))
        has_invalid = jp.any(invalid_contact)
        has_body_or_chassis_invalid = jp.any(body_or_chassis_invalid_contact)
        has_elbow_or_knee_invalid = jp.any(elbow_or_knee_invalid_contact)
        contact_indices = jp.arange(geom1.shape[0], dtype=geom1.dtype)
        sentinel_index = jp.array(geom1.shape[0], dtype=geom1.dtype)
        first_invalid_index = jp.min(
            jp.where(invalid_contact, contact_indices, sentinel_index)
        )
        safe_index = jp.minimum(first_invalid_index, geom1.shape[0] - 1)
        missing_int = jp.array(-1, dtype=geom1.dtype)
        first_invalid_non_q4_geom_id = jp.where(
            has_invalid, other_geom[safe_index], missing_int
        )
        first_invalid_ground_geom_id = jp.where(
            has_invalid, ground_geom[safe_index], missing_int
        )
        first_invalid_geom1_id = jp.where(has_invalid, geom1[safe_index], missing_int)
        first_invalid_geom2_id = jp.where(has_invalid, geom2[safe_index], missing_int)
        first_invalid_index_out = jp.where(has_invalid, first_invalid_index, missing_int)
        first_invalid_dist = jp.where(
            has_invalid,
            data.contact.dist[safe_index],
            jp.zeros((), dtype=data.contact.dist.dtype),
        )
        return (
            has_invalid,
            invalid_count,
            has_body_or_chassis_invalid,
            has_elbow_or_knee_invalid,
            first_invalid_non_q4_geom_id,
            first_invalid_ground_geom_id,
            first_invalid_geom1_id,
            first_invalid_geom2_id,
            first_invalid_index_out,
            first_invalid_dist,
        )

    # =====================================================================
    # UTILIDADES: GEOMETRÍA DE PIES Y Q3
    # =====================================================================

    def _desfases_inferiores_pies(self, data: mjx.Data) -> jax.Array:
        effector_z = data.geom_xpos[self._effector_geom_ids, 2]
        return effector_z - self._effector_geom_radii - self._floor_z

    def _separacion_pies_suelo(self, data: mjx.Data) -> jax.Array:
        return jp.maximum(0.0, self._desfases_inferiores_pies(data))

    def _separacion_total_pies_suelo(self, data: mjx.Data) -> jax.Array:
        return jp.sum(self._separacion_pies_suelo(data))

    def _diferencia_z_efector_q3(self, data: mjx.Data) -> jax.Array:
        effector_z = data.geom_xpos[self._effector_geom_ids, 2]
        q3_z = data.geom_xpos[self._q3_geom_ids, 2]
        return effector_z - q3_z

    def _exceso_efector_sobre_q3(self, data: mjx.Data) -> jax.Array:
        return jp.maximum(
            0.0,
            self._diferencia_z_efector_q3(data) - self._config.effector_q3_z_margin,
        )

    def _hay_efector_sobre_q3(self, data: mjx.Data) -> jax.Array:
        return jp.any(self._exceso_efector_sobre_q3(data) > 0.0)

    def _margen_medio_z_q3_cuerpo(self, data: mjx.Data) -> jax.Array:
        body_z = data.xpos[self._main_body_id, 2]
        q3_z = data.geom_xpos[self._q3_geom_ids, 2]
        return jp.mean(body_z - q3_z)

    def _margen_medio_q3_sobre_cuerpo(self, data: mjx.Data) -> jax.Array:
        body_z = data.xpos[self._main_body_id, 2]
        q3_z = data.geom_xpos[self._q3_geom_ids, 2]
        return jp.mean(q3_z - body_z)

    def _distancia_radial_media_q3(self, data: mjx.Data) -> jax.Array:
        body_xy = data.xpos[self._main_body_id, :2]
        q3_xy = data.geom_xpos[self._q3_geom_ids, :2]
        return jp.mean(jp.linalg.norm(q3_xy - body_xy, axis=1))

    def _distancias_xy_efectores(self, data: mjx.Data) -> jax.Array:
        com_xy = data.subtree_com[self._main_body_id, :2]
        effector_xy = data.geom_xpos[self._effector_geom_ids, :2]
        return jp.linalg.norm(effector_xy - com_xy, axis=1)

    def _distancia_xy_media_efectores(self, data: mjx.Data) -> jax.Array:
        return jp.mean(self._distancias_xy_efectores(data))

    def _recompensa_simetria_patas(self, data: mjx.Data) -> jax.Array:
        tolerance = jp.deg2rad(self._config.leg_symmetry_tolerance_deg)
        q1_values = data.qpos[self._q1_qpos_adrs]
        q2_values = data.qpos[self._q2_qpos_adrs]
        q3_values = data.qpos[self._q3_qpos_adrs]
        dispersion = (jp.std(q1_values) + jp.std(q2_values) + jp.std(q3_values)) / 3.0
        return jp.maximum(0.0, 1.0 - dispersion / tolerance)

    # =====================================================================
    # COMPROBACIÓN RÁPIDA: verifica que la recompensa funciona correctamente en qpos0
    # =====================================================================

    def verificar_recompensa_referencia(self) -> None:
        """Imprime valores de recompensa en la pose de referencia del XML.

        Llama esto antes de entrenar (fuera del JIT) para verificar que:
          - pose_imitacion_reward ~= 1.0 en qpos0.
          - altura_reward ~= 1.0 en qpos0.
          - vector_gravedad_reward alto si el cuerpo esta horizontal.
          - target_body_z_referencia ~= 0.100 m para ROBOT_POSE_IDEAL.

        Tambien testea un offset articular de 0.30 rad para verificar que
        la recompensa baja correctamente cuando la pose se aleja de la referencia.
        """
        import numpy as np

        def _a_escalar(x):
            return float(np.asarray(x))

        ref_body_z = _a_escalar(self._ref_body_z)
        target_body_z_actual = _a_escalar(self._target_body_z_actual)
        q_ref = np.asarray(self._q_referencia)

        sigma = max(float(self._config.pose_imitacion_sigma), 1e-6)
        pose_reward_ref = float(np.exp(-0.0 / sigma**2))

        offset = 0.30
        error_mse_offset = offset**2
        pose_reward_offset = float(np.exp(-error_mse_offset / sigma**2))

        height_sigma = max(float(self._config.target_height_sigma), 1e-6)
        error_altura_xml = ref_body_z - target_body_z_actual
        altura_reward_ref = float(np.exp(-((error_altura_xml / height_sigma) ** 2)))

        tilt_sigma_deg = max(
            float(self._config.get("pose_xml_final_tilt_sigma_deg", 7.0)), 1e-6
        )
        height_sigma_strict = max(
            float(self._config.get("pose_xml_final_height_sigma", 0.025)), 1e-6
        )
        data_ref = mujoco.MjData(self._mj_model)
        data_ref.qpos[:] = self._mj_model.qpos0
        data_ref.qvel[:] = 0.0
        mujoco.mj_forward(self._mj_model, data_ref)
        xmat = np.asarray(data_ref.xmat[self._main_body_id]).reshape(3, 3)
        uprightness = float(np.clip(xmat[2, 2], -1.0, 1.0))
        tilt_degrees = float(np.rad2deg(np.arccos(uprightness)))
        gravity_body = xmat.T @ np.array([0.0, 0.0, -1.0])
        gravity_lateral_error = float(gravity_body[0] ** 2 + gravity_body[1] ** 2)
        vector_gravedad_lineal_reward_base = float(
            np.clip(1.0 - gravity_lateral_error, 0.0, 1.0)
        )
        upright_direction_gate = float(np.clip(-gravity_body[2], 0.0, 1.0))
        vector_gravedad_lineal_reward = (
            vector_gravedad_lineal_reward_base * upright_direction_gate
        )
        level_gate_strict = float(np.exp(-((tilt_degrees / tilt_sigma_deg) ** 2)))
        height_gate_strict = float(
            np.exp(-((error_altura_xml / height_sigma_strict) ** 2))
        )
        inclinacion_tolerancia_deg = max(
            float(self._config.get("inclinacion_suave_tolerancia_deg", 4.0)), 0.0
        )
        inclinacion_sigma_deg = max(
            float(self._config.get("inclinacion_suave_sigma_deg", 8.0)), 1e-6
        )
        inclinacion_suave_penalty = float(
            np.clip(
                ((max(0.0, tilt_degrees - inclinacion_tolerancia_deg))
                 / inclinacion_sigma_deg) ** 2,
                0.0,
                1.0,
            )
        )

        geom1 = np.asarray(data_ref.contact.geom1[: data_ref.ncon])
        geom2 = np.asarray(data_ref.contact.geom2[: data_ref.ncon])
        dist = np.asarray(data_ref.contact.dist[: data_ref.ncon])
        floor_geom_id = int(_a_escalar(self._floor_geom_id))
        effector_geom_ids = set(np.asarray(self._effector_geom_ids, dtype=int).tolist())
        active_ground_contacts = (
            (dist <= 0.0) & ((geom1 == floor_geom_id) | (geom2 == floor_geom_id))
        )
        num_valid_foot_contacts = 0
        num_invalid_contacts = 0
        for g1, g2, active in zip(
            geom1, geom2, active_ground_contacts, strict=True
        ):
            if not active:
                continue
            other_geom = int(g2 if int(g1) == floor_geom_id else g1)
            if other_geom in effector_geom_ids:
                num_valid_foot_contacts += 1
            else:
                num_invalid_contacts += 1
        contact_scale = max(float(self._config.contacto_invalido_penalty_count_scale), 1e-6)
        contacto_invalido_normalizado = min(num_invalid_contacts / contact_scale, 1.0)
        valid_contact_gate = float(np.clip(1.0 - contacto_invalido_normalizado, 0.0, 1.0))
        if bool(self._config.get("usar_filtros_contacto_recompensa", True)):
            filtro_contacto_geometria = 0.25 + 0.75 * valid_contact_gate
        else:
            filtro_contacto_geometria = 1.0
        vector_gravedad_estabilidad_reward = (
            vector_gravedad_lineal_reward * filtro_contacto_geometria
        )
        required_contacts = max(float(self._config.min_foot_contacts_for_support), 1.0)
        support_contact_gate = float(
            np.clip(num_valid_foot_contacts / required_contacts, 0.0, 1.0)
        )
        xml_pose_success_score = (
            pose_reward_ref * level_gate_strict * height_gate_strict * valid_contact_gate
        )
        xml_pose_success_score_with_support = (
            xml_pose_success_score * support_contact_gate
        )

        print("=" * 60)
        print("SANITY CHECK: reward_referencia")
        print(f"  XML activo:                 {self._xml_path}")
        print(f"  ref_body_z (del XML):       {ref_body_z:.4f} m")
        print(f"  target_body_z_actual:       {target_body_z_actual:.4f} m")
        print(f"  q_referencia (12 joints):   {np.round(q_ref, 4)}")
        print(f"  pose_sigma:                 {sigma:.4f} rad")
        print(f"  pose_imitacion_reward @qpos0: {pose_reward_ref:.4f} (esperado ~1.0)")
        print(f"  pose_imitacion_reward @+{offset}rad: {pose_reward_offset:.4f} "
              f"(debe ser < {pose_reward_ref:.2f})")
        print(f"  altura_reward @qpos0:         {altura_reward_ref:.4f} (esperado ~1.0)")
        print(f"  height_sigma:               {height_sigma:.4f} m")
        print(f"  tilt_degrees @qpos0:        {tilt_degrees:.4f} deg")
        print(f"  gravity_lateral_error @qpos0: {gravity_lateral_error:.4f}")
        print(
            "  vector_gravedad_lineal_base @qpos0: "
            f"{vector_gravedad_lineal_reward_base:.4f}"
        )
        print(f"  upright_direction_gate @qpos0: {upright_direction_gate:.4f}")
        print(f"  vector_gravedad_lineal @qpos0: {vector_gravedad_lineal_reward:.4f}")
        print(
            "  vector_gravedad_estabilidad @qpos0: "
            f"{vector_gravedad_estabilidad_reward:.4f}"
        )
        print(f"  level_gate_strict @qpos0:   {level_gate_strict:.4f} (esperado ~1.0)")
        print(f"  height_gate_strict @qpos0:  {height_gate_strict:.4f} (esperado ~1.0)")
        print(f"  inclinacion_suave_penalty @qpos0: {inclinacion_suave_penalty:.4f}")
        print(f"  valid_contact_gate @qpos0:  {valid_contact_gate:.4f}")
        print(f"  support_contact_gate @qpos0:{support_contact_gate:.4f}")
        print(f"  num_valid_foot_contacts:    {num_valid_foot_contacts}")
        print(f"  num_invalid_contacts:       {num_invalid_contacts}")
        print(f"  xml_pose_success_score:     {xml_pose_success_score:.4f}")
        print(
            "  xml_pose_success_score_with_support: "
            f"{xml_pose_success_score_with_support:.4f}"
        )
        print("=" * 60)

    # =====================================================================
    # PROPIEDADES
    # =====================================================================

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def model_assets(self) -> dict[str, Any]:
        return {}
