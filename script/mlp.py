from pathlib import Path
from pprint import pprint
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

import argparse


if __name__ == '__main__':
    with open(root_path.joinpath("config/mlp.yaml"), "r") as fp:
        config = yaml.safe_load(fp)
    parser = argparse.ArgumentParser()
    parser.add_argument("--absolute", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    config["env"]["observation"]["absolute"] = args.absolute
    
    rl_cls_name = "PPO"
    rl_cls = PPO
    env_id = config["env_id"]
    
    train_config = config[rl_cls_name]["train"]
    model_config = config[rl_cls_name]["model"]
    model_config["tensorboard_log"] = "tensorboard/MLP_{}".format(
        "absolute" if args.absolute else "relative")
    model_config["tensorboard_log"] = root_path.joinpath(model_config["tensorboard_log"]).resolve()
    pprint(config)
    
    train_venv = graph_sdc.env_util.make_venv(
        env_id=env_id,
        n_envs=train_config["n_train_envs"],
        env_config=config["env"],
        enable_subprocess=config["enable_venv_subprocess"],
    )
    eval_venv = graph_sdc.env_util.make_venv(
        env_id=env_id,
        n_envs=train_config["n_eval_envs"],
        env_config=config["env"],
        enable_subprocess=config["enable_venv_subprocess"],
    )
    
    model = rl_cls(**model_config, env=train_venv)
    callback = graph_sdc.callback.EvalCallback(
        eval_timesteps=train_config["eval_timesteps"],
        eval_env=eval_venv,
        n_eval_episodes=train_config["n_eval_episodes"],
    )
    model.learn(
        total_timesteps=train_config["total_timesteps"],
        callback=callback,
    )
