from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.elixis_revolute_stand.elixis_revolute_stand_config import ElixisRevoluteStandCfg, ElixisRevoluteStandCfgPPO
from legged_gym.envs.elixis_revolute_stand.elixis_revolute_stand_env import ElixisRevoluteStand

from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "elixis_revolute_stand", ElixisRevoluteStand, ElixisRevoluteStandCfg(), ElixisRevoluteStandCfgPPO())
