import logging
from pathlib import Path
import pandas as pd
from typing import Optional, Sequence, Union
import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.logger import KVWriter


class EvalCallback(BaseCallback):
    def __init__(
        self,
        eval_timesteps: Union[int, Sequence[int]],
        eval_env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 5,
        csv_path: Optional[Union[str, Path]] = None,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.csv_path = None if csv_path is None else Path(csv_path)
        self.eval_timesteps = eval_timesteps
        self.next_eval = 0

    def _init_callback(self):
        if self.csv_path is not None:
            self.csv_path.unlink(missing_ok=True)
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.num_timesteps >= self.next_eval:
            self._evaluate_policy()
            self.next_eval += self.eval_timesteps
        return super()._init_callback()
    
    def _evaluate_policy(self) -> None:
        logging.info("Eval model at timestep {}".format(self.num_timesteps))
        rewards, lengths = evaluate_policy(
            model=self.model,
            env=self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            return_episode_rewards=True,
        )
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)
        length_mean = np.mean(lengths)
        step_rewards = np.mean(np.sum(rewards) / np.sum(lengths))
        logging.info("len: {:.0f}, mean: {:.2f}, std: {:.2f}, step_rew: {:.2f}".format(
            length_mean, reward_mean, reward_std, step_rewards))
        
        log_data = {
            "eval/ep_length_mean": length_mean,
            "eval/ep_reward_mean": reward_mean,
            "eval/step_reward_mean": step_rewards,}
        to_exclude = {k: None for k in log_data.keys()}
        for _format in self.model.logger.output_formats:
            if isinstance(_format, KVWriter):
                _format.write(log_data, to_exclude, self.num_timesteps)
        if self.csv_path is not None:
            df = pd.DataFrame(
                [[self.next_eval, reward_mean, reward_std]],
                columns=["step", "reward_mean", "reward_std"]
            )
            if self.csv_path.exists():
                df = pd.concat([df, pd.read_csv(self.csv_path)])
            df.to_csv(self.csv_path, index=False)
    
    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self._evaluate_policy()
            self.next_eval += self.eval_timesteps
        return super()._on_step()
