from pathlib import Path
from pprint import pprint
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.utils import get_linear_fn

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
    parser.add_argument("--absolute", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visible", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()
    if args.absolute is not None:
        config["env"]["observation"]["absolute"] = args.absolute
    if args.visible is not None:
        config["env"]["observation"]["vehicles_count"] = args.visible + 1 # add self
    rl_cls_name = config["rl_cls"]
    if args.lr is not None:
        config[rl_cls_name]["model"]["learning_rate"] = args.lr

    rl_cls = globals()[rl_cls_name]
    env_id = config["env_id"]
    
    model_name = ["mlp"]
    if args.absolute is not None:
        model_name.append("absolute" if args.absolute else "ego-centric")
    if args.visible is not None:
        model_name.append("visible={}".format(args.visible))
    if args.lr is not None:
        model_name.append("lr={:.1e}".format(args.lr))
    model_name = "_".join(model_name)
    
    root_path.joinpath("model/").mkdir(parents=True, exist_ok=True)
    with open(root_path.joinpath("model/{}.yaml".format(model_name)), "w") as fp:
        yaml.safe_dump(config, fp)
    train_config = config[rl_cls_name]["train"]
    model_config = config[rl_cls_name]["model"]
    model_config["tensorboard_log"] = "tensorboard/{}".format(model_name)
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
    learning_rate = get_linear_fn(
        train_config["lr_init"],
        train_config["lr_final"],
        train_config["lr_frac"])
    
    model = rl_cls(**model_config, env=train_venv, learning_rate=learning_rate)
    callback = graph_sdc.callback.EvalCallback(
        eval_timesteps=train_config["eval_timesteps"],
        eval_env=eval_venv,
        n_eval_episodes=train_config["n_eval_episodes"],
    )
    model.learn(
        total_timesteps=train_config["total_timesteps"],
        callback=callback,
    )
    model.save(root_path.joinpath("model/{}.zip".format(model_name)))