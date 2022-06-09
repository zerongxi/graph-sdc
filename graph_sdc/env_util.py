from functools import partial
import logging
from typing import Dict, Optional
import gym
import highway_env

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv

from graph_sdc.env import GraphEnvWrapper, HighwayEnvWrapper


def make_highway_env(
    env_id: str,
    env_config: Dict,
    graph_config: Optional[Dict] = None
) -> gym.Env:
    env = gym.make(env_id)
    if env_config["observation"]["type"] != "Kinematics":
        env.configure(env_config)
        env.reset()
        return env
    if env_id.startswith("highway"):
        env = HighwayEnvWrapper(env, env_config)
    if graph_config is not None:
        env = GraphEnvWrapper(env, graph_config)
    return env


def make_venv(
    env_id: str,
    n_envs: int = 1,
    env_config: Dict = {},
    graph_config: Optional[Dict] = None,
    enable_subprocess: bool = False,
) -> VecEnv:
    if n_envs < 1:
        raise ValueError("n_envs must be greater than 1")
    elif n_envs == 1:
        enable_subprocess = False
    
    _make_highway_env = partial(
        make_highway_env,
        env_id=env_id,
        env_config=env_config,
        graph_config=graph_config,
    )
    
    vec_env_cls = SubprocVecEnv if enable_subprocess else DummyVecEnv
    venv = make_vec_env(
        _make_highway_env,
        n_envs=n_envs,
        vec_env_cls=vec_env_cls,
    )
    return venv
