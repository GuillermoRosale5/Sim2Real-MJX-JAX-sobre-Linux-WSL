"""Sistema genérico de entrenamiento Sim2Real con MJX/JAX/PPO."""

from sim2real_mjx.entorno_robot_mjx import EntornoRobotMJX
from sim2real_mjx.entorno_robot_mjx import default_config

__all__ = ["EntornoRobotMJX", "default_config"]
