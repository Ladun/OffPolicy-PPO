
import os
import glob
import numpy as np
import logging
import shutil
import math
from datetime import datetime
from PIL import Image
from collections import deque

from omegaconf import OmegaConf
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import PPOMemory
from model import Actor, Critic
from scheduler import WarmupLinearSchedule
from utils.general import (
    set_seed, get_rng_state, set_rng_state,
    pretty_config, get_cur_time_code,
    TimerManager, get_config, get_device
)
from utils.stuff import RewardScaler, ObservationNormalizer


logger = logging.getLogger(__name__)


            
def data_iterator(batch_size, given_data, t=False):
    # Simple mini-batch spliter

    ob, ac, oldpas, adv, tdlamret, old_v = given_data
    total_size = len(ob)
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    n_batches = total_size // batch_size
    for nb in range(n_batches):
        ind = indices[batch_size * nb : batch_size * (nb + 1)]
        yield ob[ind], ac[ind], oldpas[ind], adv[ind], tdlamret[ind], old_v[ind]      


class PPOAgent:
    def __init__(self, config):

        self.config = config
        self.device = get_device(config.device)
        
        set_seed(self.config.seed)
        rng_state, _ = gym.utils.seeding.np_random(self.config.seed)
        self.env_rng_state = rng_state
        
        # -------- Define models --------
        self.policy     = Actor(config, self.device).to(self.device)
        self.old_policy = Actor(config, self.device).to(self.device)
        
        self.critic     = Critic(config, self.device).to(self.device)
        
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            **self.config.network.optimizer
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            **self.config.network.optimizer
        )

        if self.config.train.scheduler:
            self.scheduler1 = WarmupLinearSchedule(optimizer=self.policy_optimizer,
                                                   warmup_steps=0,
                                                   max_steps=self.config.train.total_timesteps // (self.config.train.max_episode_len * self.config.env.num_envs))
            self.scheduler2 = WarmupLinearSchedule(optimizer=self.critic_optimizer,
                                                   warmup_steps=0,
                                                   max_steps=self.config.train.total_timesteps // (self.config.train.max_episode_len * self.config.env.num_envs))
            
        # [EXPERIMENT] - reward scaler: r / rs.std()
        if self.config.train.reward_scaler:
            self.reward_scaler = RewardScaler(self.config.env.num_envs, gamma=self.config.train.gamma)

        # [EXPERIMENT] - observation scaler: (ob - ob.mean()) / (ob.std())
        if self.config.train.observation_normalizer:
            sp = (config.env.state_dim, ) if isinstance(config.env.state_dim, int) else list(config.env.state_dim)
            self.obs_normalizer = ObservationNormalizer(self.config.env.num_envs, sp)

        self.timer_manager  = TimerManager()
        self.writer         = None
        self.memory         = None
        self.timesteps      = 0
        self.trained_epoch  = 0

        logger.info("----------- Config -----------")
        pretty_config(config, logger=logger)
        logger.info(f"Device: {self.device}")
    

    def save(self, postfix, envs=None):
        '''
        ckpt_root
            exp_name
                config.yaml
                checkpoints
                    1
                    2
                    ...
                
        '''

        ckpt_path = os.path.join(self.config.experiment_path, "checkpoints")
        if os.path.exists(ckpt_path):
            # In order to save only the maximum number of checkpoints as max_save_store,
            # checkpoints exceeding that number are deleted. (exclude 'best')
            current_ckpt = [f for f in os.listdir(ckpt_path) if f.startswith('timesteps')]
            current_ckpt.sort(key=lambda x: int(x[9:]))
            # Delete exceeded checkpoints
            if self.config.train.max_ckpt_count > 0 and self.config.train.max_ckpt_count <= len(current_ckpt):
                for ckpt in current_ckpt[:len(current_ckpt) - self.config.train.max_ckpt_count - 1]:
                    shutil.rmtree(os.path.join(self.config.experiment_path, "checkpoints", ckpt), ignore_errors=True)


        # Save configuration file
        os.makedirs(self.config.experiment_path, exist_ok=True)
        with open(os.path.join(self.config.experiment_path, "config.yaml"), 'w') as fp:
            OmegaConf.save(config=self.config, f=fp)

        # postfix is ​​a variable for storing each episode or the best model
        ckpt_path = os.path.join(ckpt_path, postfix)
        os.makedirs(ckpt_path, exist_ok=True)
        
        # save model and optimizers
        torch.save(self.policy.state_dict(), os.path.join(ckpt_path, "policy.pt"))
        torch.save(self.critic.state_dict(), os.path.join(ckpt_path, "critic.pt"))
        torch.save(self.policy_optimizer.state_dict(), os.path.join(ckpt_path, "policy_optimizer.pt"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(ckpt_path, "critic_optimizer.pt"))
        if self.config.train.scheduler:
            torch.save(self.scheduler1.state_dict(), os.path.join(ckpt_path, "scheduler1.pt"))
            torch.save(self.scheduler2.state_dict(), os.path.join(ckpt_path, "scheduler2.pt"))

        # save random state
        torch.save(get_rng_state(), os.path.join(ckpt_path, 'rng_state.ckpt'))
        if envs:
            torch.save(envs.np_random, os.path.join(ckpt_path, 'env_rng_state.ckpt'))

        with open(os.path.join(ckpt_path, "appendix"), "w") as f:
            f.write(f"{self.timesteps}\n")
            f.write(f"{self.trained_epoch}\n")

    @classmethod
    def load(cls, experiment_path, postfix, resume=True):

        config = get_config(os.path.join(experiment_path, "config.yaml"))
        config.network.action_std_init = config.network.min_action_std
        ppo_algo = PPOAgent(config, get_device(config.device))
        
        # Create a variable to indicate which path the model will be read from
        ckpt_path = os.path.join(experiment_path, "checkpoints", postfix)
        print(f"Load pretrained model from {ckpt_path}")

        ppo_algo.policy.load_state_dict(torch.load(os.path.join(ckpt_path, "policy.pt")))
        ppo_algo.critic.load_state_dict(torch.load(os.path.join(ckpt_path, "critic.pt")))
        ppo_algo.policy_optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, "policy_optimizer.pt")))
        ppo_algo.critic_optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, "critic_optimizer.pt")))
        if ppo_algo.config.train.scheduler:
            ppo_algo.scheduler1.load_state_dict(torch.load(os.path.join(ckpt_path, "scheduler1.pt")))
            ppo_algo.scheduler2.load_state_dict(torch.load(os.path.join(ckpt_path, "scheduler2.pt")))

        # load random state
        set_rng_state(torch.load(os.path.join(ckpt_path, 'rng_state.ckpt'), map_location='cpu'))

        with open(os.path.join(ckpt_path, "appendix"), "r") as f:
            lines = f.readlines()

        if resume:
            ppo_algo.timesteps = int(lines[0])
            ppo_algo.trained_epoch = int(lines[1])
            if os.path.exists(os.path.join(ckpt_path, 'env_rng_state.ckpt')):
                ppo_algo.env_rng_state = torch.load(os.path.join(ckpt_path, 'env_rng_state.ckpt'), map_location='cpu')

        return ppo_algo

    def train(self, envs, exp_name=None):

        # Set random state for reproducibility
        envs.np_random = self.env_rng_state        

        # -------- Initialize --------

        start_time = datetime.now().replace(microsecond=0)
        
        # Create an experiment directory to record training data
        self.config.experiment_name = f"exp{get_cur_time_code()}" if exp_name is None else exp_name
        self.config.experiment_path = os.path.join(self.config.checkpoint_path, self.config.experiment_name)

        # If an existing experiment has the same name, add a number to the end of the path.
        while os.path.exists(self.config.experiment_path):
            exp_name  = self.config.experiment_path[len(self.config.checkpoint_path) + 1:]
            exp_split = exp_name.split("_")

            try:
                exp_num  = int(exp_split[-1]) + 1
                exp_name = f"{'_'.join(exp_split[:max(1, len(exp_split) - 1)])}_{str(exp_num)}"
            except:
                exp_name = f"{exp_name}_0"

            self.config.experiment_name = exp_name
            self.config.experiment_path = os.path.join(self.config.checkpoint_path, self.config.experiment_name)
        os.makedirs(self.config.experiment_path, exist_ok=True)
        logger.addHandler( logging.FileHandler(os.path.join(self.config.experiment_path, f"running_train_log.log")))

        # For logging training state
        writer_path     = os.path.join( self.config.experiment_path, 'runs')
        self.writer     = SummaryWriter(writer_path)
        # Queue to record learning data,
        # [0] is a value to prevent errors caused by missing data.
        reward_queue    = deque([0], maxlen=self.config.train.average_interval)
        duration_queue  = deque([0], maxlen=self.config.train.average_interval)

        episodic_reward = np.zeros(self.config.env.num_envs)
        duration        = np.zeros(self.config.env.num_envs)
        best_score      = -1e9

        # make rollout buffer
        self.memory = PPOMemory(
            gamma=self.config.train.gamma,
            tau=self.config.train.tau,
            advantage_type=self.config.train.advantage_type,
            device=self.device,
            off_policy_buffer_size=self.config.train.off_policy_buffer_size if self.config.train.off_policy_buffer_size > 0 else self.config.env.num_envs,
        )        
          
        # for continuous action space
        next_action_std_decay_step = self.config.network.action_std_decay_freq

        '''
        Environment symbol's information
        ===========  ==========================  ==================
        Symbol       Shape                       Type
        ===========  ==========================  ==================
        state        (num_envs, (obs_space))     numpy.ndarray
        reward       (num_envs,)                 numpy.ndarray
        term         (num_envs,)                 numpy.ndarray
        done         (num_envs,)                 numpy.ndarray
        ===========  ==========================  ==================
        '''
        state, _  = envs.reset()
        done = np.zeros(self.config.env.num_envs)

        # -------- Training Loop --------
        
        print(f"================ Start training ================")
        print(f"========= Exp name: {self.config.experiment_name} ==========")
        self.policy.eval()
        while self.timesteps < self.config.train.total_timesteps:

            with self.timer_manager.get_timer("Total"):
                with self.timer_manager.get_timer("Collect Trajectory"):
                    for t in range(0, self.config.train.max_episode_len ):

                        # ------------- Collect Trajectories -------------

                        '''
                        Actor-Critic symbol's information
                        ===========  ==========================  ==================
                        Symbol       Shape                       Type
                        ===========  ==========================  ==================
                        action       (num_envs,)                 torch.Tensor
                        logprobs     (num_envs,)                 torch.Tensor
                        ent          (num_envs,)                 torch.Tensor
                        values       (num_envs, 1)               torch.Tensor
                        ===========  ==========================  ==================
                        '''

                            
                        with torch.no_grad():
                            if self.config.train.observation_normalizer:
                                state = self.obs_normalizer(state)
                            _state = torch.from_numpy(state).to(self.device, dtype=torch.float)
                            action, logprobs, _ = self.policy(_state)
                            values = self.critic(_state)
                            values = values.flatten() # reshape shape of the value to (num_envs,)
                            
                        next_state, reward, terminated, truncated, _ = envs.step(np.clip(action.cpu().numpy(), envs.action_space.low, envs.action_space.high))
                        self.timesteps += self.config.env.num_envs

                        # update episodic_reward
                        episodic_reward += reward
                        duration += 1
                        
                        if self.config.train.reward_scaler:
                            reward = self.reward_scaler(reward, terminated + truncated)
                            
                        # add experience to the memory                    
                        self.memory.store(
                            state=state,
                            action=action,
                            reward=reward,
                            done=done,
                            value=values,
                            logprob=logprobs
                        )
                        done = terminated + truncated

                        for idx, d in enumerate(done):
                            if d:
                                reward_queue.append(episodic_reward[idx])
                                duration_queue.append(duration[idx])                       
                                
                                episodic_reward[idx] = 0
                                duration[idx] = 0          
                        
                        # update state
                        state = next_state                        
                    
                # ------------- Calculate gae for optimizing-------------

                # Estimate next state value for gae
                with torch.no_grad():
                    if self.config.train.observation_normalizer:
                        next_state = self.obs_normalizer(next_state)
                    next_value = self.critic(torch.Tensor(next_state).to(self.device))                    
                    next_value = next_value.flatten()

                # update gae & tdlamret
                # Optimize
                with self.timer_manager.get_timer("Optimize"):
                    self.optimize(next_state, next_value, done)
                    
                # action std decaying
                if self.config.env.is_continuous:
                    while self.timesteps > next_action_std_decay_step:
                        next_action_std_decay_step +=  self.config.network.action_std_decay_freq
                        self.policy.action_decay(
                            self.config.network.action_std_decay_rate,
                            self.config.network.min_action_std
                        )

                # scheduling learning rate
                if self.config.train.scheduler:
                    self.scheduler1.step()
                    self.scheduler2.step()

            # ------------- Logging training state -------------

            avg_score       = np.round(np.mean(reward_queue), 4)
            std_score       = np.round(np.std(reward_queue), 4)
            avg_duration    = np.round(np.mean(duration_queue), 4)

            # Writting for tensorboard
            self.writer.add_scalar("train/score", avg_score, self.timesteps)
            self.writer.add_scalar("train/duration", avg_duration, self.timesteps)
            if self.config.train.scheduler:
                # TODO: Clean up your code 
                for idx, lr in enumerate(self.scheduler1.get_lr()):
                    self.writer.add_scalar(f"train/learning_rate{idx}", lr, self.timesteps)
                for idx, lr in enumerate(self.scheduler2.get_lr()):
                    self.writer.add_scalar(f"train/learning_rate{idx + 1}", lr, self.timesteps)

            # Printing for console
            remaining_num_of_optimize = int(math.ceil((self.config.train.total_timesteps - self.timesteps) /
                                                      (self.config.env.num_envs * self.config.train.max_episode_len)))
            remaining_training_time_min = int(self.timer_manager.get_timer('Total').get() * remaining_num_of_optimize // 60)
            remaining_training_time_sec = int(self.timer_manager.get_timer('Total').get() * remaining_num_of_optimize % 60)
            logger.info(f"[{datetime.now().replace(microsecond=0) - start_time}] {self.timesteps}/{self.config.train.total_timesteps} - score: {avg_score} +-{std_score} \t duration: {avg_duration}")
            for k, v in self.timer_manager.timers.items():
                logger.info(f"\t\t {k} time: {v.get()} sec")
            logger.info(f"\t\t Estimated training time remaining: {remaining_training_time_min} min {remaining_training_time_sec} sec")

            # Save best model
            if avg_score >= best_score:
                self.save(f'best', envs)
                best_score = avg_score

            self.save(f"timesteps{self.timesteps}", envs)

        envs.close()
        self.save('last')
        return best_score
    

    def prepare_data(self, data):       
        s        = data['state'].float()
        a        = data['action']
        logp     = data['logprob'].float()
        v_target = data['v_target'].float()
        adv      = data['advant'].float()
        v        = data['value'].float()

        # # normalize advant a.k.a atarg
        # adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        return s, a, logp, adv, v_target, v   

    def copy_network_param(self):
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.old_policy.set_action_std(self.policy.action_std)  
    

    def optimize(self, next_state, next_value, done):
        
        self.copy_network_param()
        
                    
        self.memory.finish(next_state, next_value, done)
                    

        fraction = self.config.train.fraction
        with self.timer_manager.get_timer("\toptimize_ppo"):
            
            # -------- PPO Training Loop --------
                
            self.policy.train()         
            for _ in range(self.config.train.ppo.optim_epochs):
                   
                avg_policy_loss = 0
                avg_entropy_loss = 0
                avg_value_loss = 0
                
                # ------------- Uniform sampling -------------
                
                with self.timer_manager.get_timer("\tuniform sampling"):
                    data, inds  = self.memory.uniform_sample(int(self.config.env.num_envs * (1 - fraction)), (self.old_policy, self.critic))
                    data        = self.prepare_data(data)
                    
                    data_loader     = data_iterator(self.config.train.ppo.batch_size, data)                    
                    v_loss          = self.optimize_critic(data_loader)
                    avg_value_loss += v_loss   
                    self.memory.update_priority((self.old_policy, self.critic), inds)
                    
                    data_loader        = data_iterator(self.config.train.ppo.batch_size, data)
                    p_loss, e_loss     = self.optimize_actor(data_loader)  
                    avg_policy_loss   += p_loss
                    avg_entropy_loss  += e_loss                  
                    
                    
                # ------------- Critic prioritized sampling -------------
                                    
                with self.timer_manager.get_timer("\tcritic prioritized sampling"):
                    data, inds  = self.memory.priority_sample(int(self.config.env.num_envs * fraction), (self.old_policy, self.critic))
                    data        = self.prepare_data(data)
                    
                    data_loader     = data_iterator(self.config.train.ppo.batch_size, data)
                    v_loss          = self.optimize_critic(data_loader)
                    avg_value_loss += v_loss
                    self.memory.update_priority((self.old_policy, self.critic), inds)
                    
                # ------------- Actor inverse prioritized sampling -------------
                
                with self.timer_manager.get_timer("\tactor prioritized sampling"):
                    data, _  = self.memory.priority_sample(int(self.config.env.num_envs * fraction), (self.old_policy, self.critic), inverse=True)
                    data     = self.prepare_data(data)
                    
                    data_loader        = data_iterator(self.config.train.ppo.batch_size, data)
                    p_loss, e_loss     = self.optimize_actor(data_loader)  
                    avg_policy_loss   += p_loss
                    avg_entropy_loss  += e_loss      
                    
                # ------------- Recording -------------
                
                avg_policy_loss /= 2
                avg_entropy_loss /= 2
                avg_value_loss /= 2
                    
                self.writer.add_scalar("train/policy_loss", avg_policy_loss, self.trained_epoch)
                self.writer.add_scalar("train/entropy_loss", avg_entropy_loss, self.trained_epoch)
                self.writer.add_scalar("train/value_loss", avg_value_loss, self.trained_epoch)
                self.writer.add_scalar("train/total_loss", avg_policy_loss + avg_entropy_loss + avg_value_loss, self.trained_epoch)                 

                self.trained_epoch += 1
                    
    def optimize_actor(self, data_loader):

        policy_losses   = []
        entropy_losses  = []
        
        c2 = self.config.train.ppo.coef_entropy_penalty
        for batch in data_loader:
            bev_ob, bev_ac, bev_logp, bev_adv, _, _ = batch
            bev_adv = (bev_adv - bev_adv.mean()) / (bev_adv.std() + 1e-7)

            _, cur_logp, cur_ent = self.policy(bev_ob, action=bev_ac)

            with torch.no_grad():
                _, old_logp, _ = self.old_policy(bev_ob, action=bev_ac)

            # -------- Policy Loss --------

            # ratio = pi(x|s) / mu(x|s)
            ratio   = torch.exp(cur_logp - bev_logp)

            # loss  = ratio * advantage
            surr1   = ratio * bev_adv

            if self.config.train.ppo.loss_type == "clip":
                # clipped loss
                lower           = (1 - self.config.train.ppo.eps_clip) * torch.exp(old_logp - bev_logp)
                upper           = (1 + self.config.train.ppo.eps_clip) * torch.exp(old_logp - bev_logp)
                clipped_ratio   = torch.clamp(ratio, lower, upper)

                surr2       = clipped_ratio * bev_adv
                policy_surr = torch.min(surr1, surr2)

            elif self.config.train.ppo.loss_type == "kl":
                # kl-divergence loss
                policy_surr = surr1 - 0.01 * torch.exp(bev_logp) * (bev_logp - cur_logp)
            else:
                # simple ratio loss
                policy_surr = surr1

            # policy loss
            policy_surr = -policy_surr.mean()

            # entropy loss
            policy_ent  = -cur_ent.mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss = policy_surr + c2 * policy_ent
            policy_loss.backward()
            if self.config.train.clipping_gradient:
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.policy_optimizer.step()

            # ---------- For recoding training loss data ----------

            policy_losses.append(policy_surr.item())
            entropy_losses.append(policy_ent.item())
            
        return np.mean(policy_losses), np.mean(entropy_losses)

    def optimize_critic(self, data_loader):

        value_losses    = []
        
        c1 = self.config.train.ppo.coef_value_function

        for batch in data_loader:
            bev_ob, _, _, _, bev_vtarg, bev_v = batch

            cur_v = self.critic(bev_ob)
            cur_v = cur_v.reshape(-1)
            
            # --------------- Value Loss ---------------

            if self.config.train.ppo.value_clipping:
                cur_v_clipped = bev_v + torch.clamp(
                    cur_v - bev_v,
                    -self.config.train.ppo.eps_clip,
                    self.config.train.ppo.eps_clip
                )

                vloss1  = (cur_v - bev_vtarg) ** 2
                vloss2  = (cur_v_clipped - bev_vtarg) ** 2
                vf_loss = torch.max(vloss1, vloss2)
            else:
                vf_loss = (cur_v - bev_vtarg) ** 2

            vf_loss = 0.5 * vf_loss.mean()
            

            self.critic_optimizer.zero_grad()
            critic_loss = c1 * vf_loss
            critic_loss.backward()
            if self.config.train.clipping_gradient:
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.critic_optimizer.step()
            
            # ---------- For recording training loss data ----------
            
            value_losses.append(vf_loss.item())

        return np.mean(value_losses)


    
    def play(self, env, max_ep_len, num_episodes=10):  

        rewards = []
        durations = []

        for episode in range(num_episodes):

            episodic_reward = 0
            duration = 0
            state, _ = env.reset()

            for t in range(max_ep_len):

                # ------------- Collect Trajectories -------------

                with torch.no_grad():
                    action, _, _, _ = self.network(torch.Tensor(state).unsqueeze(0).to(self.device))
                next_state, reward, terminated, truncated, info = env.step(action.cpu().numpy().squeeze(0))
                done = terminated + truncated

                episodic_reward += reward
                duration += 1

                if done:
                    break

                # update state
                state = next_state

            rewards.append(episodic_reward)
            durations.append(duration)
            logger.info(f"Episode {episode}: score - {episodic_reward} duration - {t}")

        avg_reward = np.mean(rewards)
        avg_duration = np.mean(durations)
        logger.info(f"Average score {avg_reward}, duration {avg_duration} on {num_episodes} games")               
        env.close()

