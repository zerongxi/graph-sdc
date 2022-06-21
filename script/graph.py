from pathlib import Path
from pprint import pprint
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.utils import get_linear_fn
import yaml
import graph_sdc
import argparse

#! ignore warnings
import warnings
warnings.filterwarnings("ignore")

import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def main():
    root_path = Path(__file__).parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--absolute", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visible", type=int, default=None)
    parser.add_argument("--density", type=float, default=None)
    parser.add_argument("--n_neighbors", type=int, default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--knn_metric", type=str, default=None)
    parser.add_argument("--knn_seconds", type=float, default=None)
    parser.add_argument("--traj_discount_factor", type=float, default=None)
    args = parser.parse_args()
    
    config_path = "config/graph.yaml"
    if args.local:
        config_path = "config/local_test.yaml"
    
    with open(root_path.joinpath(config_path), "r") as fp:
        config = yaml.safe_load(fp)
    if args.absolute is not None:
        config["env"]["observation"]["absolute"] = args.absolute
    if args.visible is not None:
        config["env"]["observation"]["vehicles_count"] = args.visible + 1 # add self
    if args.density is not None:
        config["env"]["vehicles_count"] = int(config["env"]["vehicles_count"]\
            * args.density / config["env"]["vehicles_density"])
        config["env"]["vehicles_density"] = args.density
    if args.n_neighbors is not None:
        config["graph"]["n_neighbors"] = args.n_neighbors
    if args.radius is not None:
        config["graph"]["radius"] = args.radius
    if args.knn_metric is not None:
        config["graph"]["metric"] = args.knn_metric
    if args.knn_seconds is not None:
        config["graph"][config["graph"]["metric"]]["seconds"] =\
            args.knn_seconds
    if args.traj_discount_factor is not None:
        config["graph"]["trajectory"]["discount_factor"] =\
            args.traj_discount_factor
        
    rl_cls_name = config["rl_cls"]
    learning_rate = get_linear_fn(
        config[rl_cls_name]["train"]["lr_init"],
        config[rl_cls_name]["train"]["lr_final"],
        config[rl_cls_name]["train"]["lr_frac"])

    rl_cls = globals()[rl_cls_name]
    env_id = config["env_id"]
    
    model_name = ["grad"]
    if args.absolute is not None:
        model_name.append("global" if args.absolute else "ego-centric")
    if args.visible is not None:
        model_name.append("visible={}".format(args.visible))
    if args.density is not None:
        model_name.append("density={}".format(args.density))
    model_name.append("filter={}".format("radius" if config["graph"]["radius"] is not None else "knn"))
    if args.n_neighbors is not None:
        model_name.append("knn={}".format(args.n_neighbors))
    if args.radius is not None:
        model_name.append("radius={}".format(args.radius))
    if args.knn_metric is not None:
        model_name.append("metric={}".format(args.knn_metric))
    if args.knn_seconds is not None:
        model_name.append("seconds={:.1f}".format(args.knn_seconds))
    if args.traj_discount_factor is not None:
        model_name.append("discount={:.1f}".format(args.traj_discount_factor))
    model_name = "_".join(model_name)
    
    root_path.joinpath("model/").mkdir(parents=True, exist_ok=True)
    with open(root_path.joinpath("model/{}.yaml".format(model_name)), "w") as fp:
        yaml.safe_dump(config, fp)
    
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
    train_venv.close()
    eval_venv.close()


if __name__ == '__main__':
    main()
