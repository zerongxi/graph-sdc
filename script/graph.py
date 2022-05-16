from pathlib import Path
from pprint import pprint
from stable_baselines3 import PPO
import torch as th

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

import argparse


if __name__ == '__main__':
    with open(root_path.joinpath("config/graph.yaml"), "r") as fp:
        config = yaml.safe_load(fp)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_cls", type=str)
    parser.add_argument("--absolute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n_neighbors", type=int)
    args = parser.parse_args()
    if args.graph_cls is not None:
        config["graph_cls"] = args.graph_cls
    config["env"]["observation"]["absolute"] = args.absolute
    if args.n_neighbors != None:
        config["graph"]["n_neighbors"] = args.n_neighbors
    
    rl_cls_name = "PPO"
    rl_cls = PPO
    env_id = config["env_id"]
    
    model_name = "{}_{}_knn_{}".format(
        args.graph_cls,
        "absolute" if args.absolute else "relative",
        config["graph"]["n_neighbors"])
    train_config = config[rl_cls_name]["train"]
    model_config = config[rl_cls_name]["model"]
    model_config["tensorboard_log"] = "tensorboard/{}".format(model_name)
    model_config["policy_kwargs"].update(dict(
        features_extractor_class=graph_sdc.graph_feature.GraphFeaturesExtractor,
        features_extractor_kwargs=dict(
            config=config[config["graph_cls"]],
            graph_cls_name=config["graph_cls"],)
    ))
    model_config["tensorboard_log"] = root_path.joinpath(model_config["tensorboard_log"]).resolve()
    pprint(config)
    
    train_venv = graph_sdc.env_util.make_venv(
        env_id=env_id,
        n_envs=train_config["n_train_envs"],
        env_config=config["env"],
        graph_config=config["graph"],
        enable_subprocess=config["enable_venv_subprocess"],
    )
    eval_venv = graph_sdc.env_util.make_venv(
        env_id=env_id,
        n_envs=train_config["n_eval_envs"],
        env_config=config["env"],
        graph_config=config["graph"],
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
    model.save(root_path.joinpath("model/{}.pkl".format(model_name)))