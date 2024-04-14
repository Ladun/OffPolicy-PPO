
import random
from collections import defaultdict, deque

import numpy as np
import torch

class PPOMemory:
    def __init__(self, gamma, tau, advantage_type, device,
                 off_policy_buffer_size):
        
        self.gamma  = gamma
        self.tau    = tau
        self.device = device
        self.advantage_type = advantage_type

        self.trajectories = deque(maxlen=off_policy_buffer_size)
        self.temp_memory = {
            "state": [],
            "action": [],
            "reward": [],
            "done": [],
            "value": [],
            "logprob": []
        }

    def store(self, **kwargs):

        for k, v in kwargs.items():
            if k not in self.temp_memory:
                print("[Warning] wrong data insertion")
            else:
                self.temp_memory[k].append(v)

    def finish(self, ns, v, d):
        """
        Parameters:
            ===========  ==========================  ==================
            Symbol       Shape                       Type
            ===========  ==========================  ==================
            ns           (num_envs,)                 numpy.ndarray      // next state
            v            (num_envs,)                 torch.Tensor       // value in the next state
            d            (num_envs,)                 numpy.ndarray      // done in the next state

        Returns:
            original ppo data dictionary

        Description:
            Information about the value in storage
            ===========  ==================================  ==================
            Symbol       Shape                               Type
            ===========  ==================================  ==================
            state        list of (num_envs, (obs_space))     numpy.ndarray
            reward       list of (num_envs,)                 numpy.ndarray
            done         list of (num_envs,)                 numpy.ndarray
            action       list of (num_envs,)                 torch.Tensor
            logprob      list of (num_envs,)                 torch.Tensor
            value        list of (num_envs,)                 torch.Tensor
            ===========  ==================================  ==================
        """
        trajectory = {k: torch.stack(v)
                        if isinstance(v[0], torch.Tensor)
                        else torch.from_numpy(np.stack(v)).to(self.device)
                      for k, v in self.temp_memory.items()}

        steps, num_envs = trajectory['reward'].size()

        # TODO: code refactoring (remove dictionary, Now the code is harmful to read)
        trajectory['state'] = torch.cat([trajectory['state'], torch.from_numpy(ns).to(self.device).unsqueeze(0)], dim=0)
        trajectory['value'] = torch.cat([trajectory['value'], v.unsqueeze(0)], dim=0)
        trajectory['done']  = torch.cat([trajectory['done'], torch.from_numpy(d).to(self.device).unsqueeze(0)], dim=0).float()

        for i in range(num_envs):
            self.trajectories.append({k: v[:, i] for k, v in trajectory.items()})

        self.reset_temp_memory()

    def get_data(self, num_of_trajectories, network):

        total_size = len(self.trajectories)
        indices = np.arange(total_size)
        np.random.shuffle(indices)
        
        batch = defaultdict(list)
        
        for i in indices[:num_of_trajectories]:
            for k, v in self.trajectories[i].items():
                batch[k].append(v.unsqueeze(1))

        batch = {k: torch.cat(batch[k], dim=1)
                 for k in ["state", "action", "reward", "done", "value", "logprob"]}
        if self.advantage_type == 'gae':
            batch = self.calculate_gae(network, **batch)
        elif self.advantage_type == 'vtrace':
            batch = self.calculate_vtrace(network, **batch)
        elif self.advantage_type == 'vtrace_gae':
            batch = self.calculate_vtrace_gae(network, **batch)

        return batch

    def calculate_gae(self, network,
                      state, action, reward, done, value, logprob):

        steps, num_envs = reward.size()

        gae_t       = torch.zeros(num_envs).to(self.device)
        advantage   = torch.zeros((steps, num_envs)).to(self.device)

        # Each episode is calculated separately by done.
        for t in reversed(range(steps)):
            # delta(t)   = reward(t) + γ * value(t+1) - value(t)
            delta_t      = reward[t] + self.gamma * value[t+1] * (1 - done[t + 1]) - value[t]

            # gae(t)     = delta(t) + γ * τ * gae(t + 1)
            gae_t        = delta_t + self.gamma * self.tau * gae_t * (1 - done[t + 1])
            advantage[t] = gae_t

        # Remove value in the next state
        v_target = advantage + value[:steps]

        trajectory = {
            "state"     : state,
            "reward"    : reward,
            "action"    : action,
            "logprob"   : logprob,
            "done"      : done,
            "value"     : value,
            "advant"    : advantage,
            "v_target"  : v_target
        }
        # The first two values ​​refer to the trajectory length and number of envs.
        return {k: v[:steps].reshape(-1, *v.size()[2:]) for k, v in trajectory.items()}

    def calculate_vtrace(self, network,
                         state, action, reward, done, value, logprob):

        steps, num_envs = reward.size()

        with torch.no_grad():
            values = network.get_value(state.reshape((steps + 1) * num_envs, *state.size()[2:]).float())
            _, pi_logp, _, _ = network(state[:steps].reshape(steps * num_envs, *state.size()[2:]).float(),
                                       action=action.reshape(steps * num_envs, *action.size()[2:]))

            values  = values.reshape(steps + 1, num_envs)
            pi_logp = pi_logp.reshape(steps, num_envs)
            ratio   = torch.exp(pi_logp - logprob)

        # c
        c = torch.min(torch.ones_like(ratio), ratio)
        # rho
        rho = torch.min(torch.ones_like(ratio), ratio)

        vtrace = torch.zeros((steps + 1, num_envs)).to(self.device)
        for t in reversed(range(steps)):
            # delta(s)  = rho * (reward(s) + γ * Value(s+1) - Value(s))
            delta       = rho[t] * (reward[t] + self.gamma * value[t + 1] * (1 - done[t + 1]) - value[t])

            # vtrace(s) = delta(s) + γ * c_s * vtrace(s+1)
            vtrace[t]   = delta + self.gamma * c[t] * vtrace[t + 1] * (1 - done[t + 1])

        # vtrace(s) = vtrace(s) + value(s)
        vtrace      = vtrace + value
        v_target    = vtrace
        advantage   = rho * (reward + self.gamma * vtrace[1:] - value[:steps])

        trajectory = {
            "state"     : state,
            "reward"    : reward,
            "action"    : action,
            "logprob"   : logprob,
            "done"      : done,
            "value"     : value,
            "advant"    : advantage,
            "v_target"  : v_target
        }

        # The first two values ​​refer to the trajectory length and number of envs.
        return {k: v[:steps].reshape(-1, *v.size()[2:]) for k, v in trajectory.items()}

    def calculate_vtrace_gae(self, network,
                             state, action, reward, done, value, logprob):

        steps, num_envs = reward.size()

        with torch.no_grad():
            values = network.get_value(state.reshape((steps + 1) * num_envs, *state.size()[2:]).float())
            _, pi_logp, _, _ = network(state[:steps].reshape(steps * num_envs, *state.size()[2:]).float(),
                                       action=action.reshape(steps * num_envs, *action.size()[2:]))

            values  = values.reshape(steps + 1, num_envs)
            pi_logp = pi_logp.reshape(steps, num_envs)

            ratio   = torch.exp(pi_logp - logprob)
            ratio   = torch.min(torch.ones_like(ratio), ratio)            

        gae_t       = torch.zeros(num_envs).to(self.device)
        advantage   = torch.zeros((steps, num_envs)).to(self.device)

        # Each episode is calculated separately by done.
        for t in reversed(range(steps)):
            # delta(t)   = rho(t) * (reward(t) + γ * value(t+1) - value(t))
            delta_t      = reward[t] + self.gamma * value[t+1] * (1 - done[t+1]) - value[t]
            delta_t      = ratio[t] * delta_t

            # gae(t)     = delta(t) + rho(t) * γ * τ * gae(t + 1)
            gae_t        = delta_t + ratio[t] * self.gamma * self.tau * gae_t * (1 - done[t + 1])
            advantage[t] = gae_t

        # Remove value in the next state
        v_target = advantage + value[:steps]

        trajectory = {
            "state"     : state,
            "reward"    : reward,
            "action"    : action,
            "logprob"   : logprob,
            "done"      : done,
            "value"     : value,
            "advant"    : advantage,
            "v_target"  : v_target
        }

        # The first two values ​​refer to the trajectory length and number of envs.
        return {k: v[:steps].reshape(-1, *v.size()[2:]) for k, v in trajectory.items()}


    def reset_temp_memory(self):
        self.temp_memory = {k: [] for k, v in self.temp_memory.items()}

