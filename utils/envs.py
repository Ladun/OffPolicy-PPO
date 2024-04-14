import gymnasium as gym

import numpy as np

def create_mujoco_env(env_name, video_path=None):
    env = gym.make(env_name, render_mode='rgb_array')
    if video_path:
        env = gym.wrappers.RecordVideo(env, video_path)
    return env
    