
import random
from collections import defaultdict, deque
import torch.nn.functional as F


import numpy as np
import torch

class SumTree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.node = []# [[0, 0]] * (2 * capacity - 1)
        for _ in range(2 * capacity - 1):
            self.node.append([0, 0])
        self.data = [None] * capacity
        self.data_idx = 0

        self.size = 0
        
    def total(self, inverse=0):
        return self.node[0][inverse]

    def update(self, data_idx, value, inverse=0):
        idx = data_idx + self.capacity - 1
        change = value - self.node[idx][inverse]

        self.node[idx][inverse] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.node[parent][inverse] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        self.data[self.data_idx] = data
        
        # TODO: combining into one function and using value variable
        self.update(self.data_idx, 1)
        self.update(self.data_idx, 1, inverse=1)

        self.data_idx = (self.data_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self, s, inverse=0):

        idx = 0
        while 2 * idx + 1 < len(self.node):
            left, right = 2 * idx + 1, 2 * idx + 2

            if s <= self.node[left][inverse]:
                idx = left
            else:
                idx = right
                s = s - self.node[left][inverse]

        data_idx = idx - (self.capacity - 1)

        return data_idx, self.node[idx][inverse], self.data[data_idx]


class PPOMemory:
    def __init__(self, gamma, tau, advantage_type, device,
                 off_policy_buffer_size):
        
        self.gamma  = gamma
        self.tau    = tau
        self.device = device
        self.advantage_type = advantage_type

        self.tree = SumTree(off_policy_buffer_size)
        
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
    
        priority = torch.ones((num_envs))
        for i in range(num_envs):
            self.tree.add(value=priority[i].item(),
                          data={k: v[:, i] for k, v in trajectory.items()})

        self.reset_temp_memory()

    def uniform_sample(self, num_of_trajectories, network):
        inds = np.arange(self.tree.size)
        np.random.shuffle(inds)
        inds = inds[:num_of_trajectories]
        
        batch = defaultdict(list)        
        for i in inds:
            for k, v in self.tree.data[i].items():
                batch[k].append(v.unsqueeze(1))
                
        return self.calculate(network, batch), inds
        
    def priority_sample(self, num_of_trajectories, network, inverse=0):

        segment = self.tree.total(inverse) / num_of_trajectories
        batch = defaultdict(list)
        inds = []
        for i in range(num_of_trajectories):
            try:
                a = segment * i
                b = segment * (i + 1)
                s = random.uniform(a, b)
                data_idx, _, data = self.tree.get(s, inverse)
                data = {k: t.clone().detach() for k, t in data.items()}
                for k, v in data.items():
                    batch[k].append(v.unsqueeze(1))
                inds.append(data_idx)
            except:
                print(f"total: {self.tree.total(inverse)} | s: {s} | segment: {segment} | [a, b]: {a}, {b} | {data_idx}, {data}")

        return self.calculate(network, batch), inds
    
    def update_priority(self, network, inds):
        
        batch = defaultdict(list)
        for i in inds:
            for k, v in self.tree.data[i].items():
                batch[k].append(v.unsqueeze(1))
        
        batch = self.calculate(network, batch)
        batch = {k: v.reshape(-1, len(inds), *v.size()[1:]) for k, v in batch.items()}
        for i in range(batch['advant'].size(1)):
            p = torch.mean(batch['advant'][:, i], dim=0)
            
            # TODO: combined into one function
            self.tree.update(inds[i], F.softplus(p).item())
            self.tree.update(inds[i], F.softplus(-p).item(), inverse=1)

    def calculate(self, network, batch):
        
        batch = {k: torch.cat(batch[k], dim=1)
                 for k in ["state", "action", "reward", "done", "value", "logprob"]}
        if self.advantage_type == 'gae':
            batch = self.calc_gae(**batch)
        elif self.advantage_type == 'vtrace':
            batch = self.calc_vtrace(network, **batch)
        elif self.advantage_type == 'vtrace_gae':
            batch = self.calc_vtrace_gae(network, **batch)
        
        return batch

    def calc_gae(self, state, action, reward, done, value, logprob):

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

    def calc_vtrace(self, network, 
                    state, action, reward, done, value, logprob):

        actor, critic = network
        steps, num_envs = reward.size()

        with torch.no_grad():
            values = critic(state.reshape((steps + 1) * num_envs, *state.size()[2:]).float())
            _, pi_logp, _ = actor(state[:steps].reshape(steps * num_envs, *state.size()[2:]).float(),
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

    def calc_vtrace_gae(self, network, 
                        state, action, reward, done, value, logprob):

        actor, critic = network
        steps, num_envs = reward.size()

        with torch.no_grad():
            values = critic(state.reshape((steps + 1) * num_envs, *state.size()[2:]).float())
            _, pi_logp, _ = actor(state[:steps].reshape(steps * num_envs, *state.size()[2:]).float(),
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

