
import os
import random
import logging
import time
from datetime import datetime
import numpy as np

import omegaconf
from omegaconf import OmegaConf

import torch


class Timer:
    def __init__(self):
        self.start = 0
        self.end = 0
        self.duration = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.duration = self.end - self.start

    def get(self):
        return round(self.duration, 5)


class TimerManager:
    def __init__(self):
        self.timers = {}

    def get_timer(self, name):
        if name in self.timers:
            return self.timers[name]
        else:
            self.timers[name] = Timer()
            return self.timers[name]


def get_cur_time_code():
    return datetime.now().replace(microsecond=0).strftime('%Y%m%d%H%M%S')


def make_directory_and_get_path(dir_path, file_name=None, uniqueness=False):
    if not os.path.exists(dir_path):
        print("Make directory:", dir_path)
    os.makedirs(dir_path, exist_ok=True)

    if file_name:
        if uniqueness:
            n, e = os.path.splitext(file_name)
            cur_time = get_cur_time_code()
            file_name = f'{n}_{cur_time}{e}'
        
        path = os.path.join(dir_path, file_name)
    else: 
        path = dir_path

    print(f"Path is '{path}'")

    return path


def get_config(yaml_file=None, yaml_string=None, **kwargs):
    assert yaml_file is not None or yaml_string is not None, 'Must enter yaml_file or string'

    if yaml_string is not None:
        conf = OmegaConf.create(yaml_string)
    else:
        conf = OmegaConf.load(yaml_file)

    # Update additional options
    if kwargs:
        conf.update(kwargs)

    if 'checkpoint_path' not in conf:
        conf.checkpoint_path = '.'

    return conf

def get_device(device=None):
    # --- Define torch device from config 'device'
    device = torch.device(
        'cpu' 
        if not torch.cuda.is_available() or device is None
        else device
    )
    torch.cuda.empty_cache()
    return device


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng_state():
    rng_state_dict = {
        'cpu_rng_state': torch.get_rng_state(),
        'gpu_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        'numpy_rng_state': np.random.get_state(),
        'py_rng_state': random.getstate()
    }
    return rng_state_dict

def set_rng_state(rng_state_dict):
    torch.set_rng_state(rng_state_dict['cpu_rng_state'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(rng_state_dict['gpu_rng_state'])
    np.random.set_state(rng_state_dict['numpy_rng_state'])
    random.setstate(rng_state_dict['py_rng_state'])


def pretty_config(config, indent_level=0, logger=None):

    print_func = logger.info if logger else print

    tab = '\t' * indent_level

    for k, v in config.items():
        if isinstance(v, omegaconf.dictconfig.DictConfig):
            print_func(f"{tab}{k}:")
            pretty_config(v, indent_level + 1, logger)
        else:
            print_func(f"{tab}{k}: {v}")