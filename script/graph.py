from pathlib import Path
from stable_baselines3 import PPO

import yaml

root_path = Path(__file__).parents[1]

#! relative path import
import sys
sys.path.append(str(root_path.resolve()))
import graph_sdc

#! ignore warnings
import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


if __name__ == '__main__':
    with open(root_path.joinpath("config/graph.yaml"), "r") as fp:
        config = yaml.safe_load(fp)
    rl_cls_name = "PPO"
    rl_cls = PPO
    graph_cls_name = "GATFeaturesExtractor"
    graph_cls = graph_sdc.graph_feature.GATFeaturesExtractor
    env_id = config["env_id"]
    
    train_config = config[rl_cls_name]["train"]
    model_config = config[rl_cls_name]["model"]
    model_config["policy_kwargs"].update(dict(
        features_extractor_class=graph_cls,
        features_extractor_kwargs=dict(config=config["GAT"])
    ))
    model_config["tensorboard_log"] = root_path.joinpath(model_config["tensorboard_log"]).resolve()
    
    train_venv = graph_sdc.env_util.make_venv(
        env_id=env_id,
        n_envs=train_config["n_train_envs"],
        env_config=config["env"],
        graph_config=config["graph"],
        enable_subprocess=config["enable_venv_subprocess"]
    )
    eval_venv = graph_sdc.env_util.make_venv(
        env_id=env_id,
        n_envs=train_config["n_eval_envs"],
        env_config=config["env"],
        graph_config=config["graph"],
        enable_subprocess=config["enable_venv_subprocess"]
    )
    
    model = rl_cls(**model_config, env=train_venv)
    callback = graph_sdc.callback.EvalCallback(
        eval_timesteps=train_config["eval_timesteps"],
        eval_env=eval_venv,
        n_eval_episodes=train_config["n_eval_episodes"],
    )
    model.learn(
        total_timesteps=train_config["total_timesteps"],
        callback=callback
    )
