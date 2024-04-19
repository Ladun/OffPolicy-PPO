
import omegaconf

import torch
from torch import nn
from torch.distributions import MultivariateNormal, Normal
from torch.distributions import Categorical


def init_normal_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

def init_orthogonal_weights(m):
    if isinstance(m, nn.Linear):
        orthogonal_init(m.weight)
        nn.init.constant_(m.bias, 0.1)

def orthogonal_init(tensor, gain=1):
    '''
    https://github.com/implementation-matters/code-for-paper/blob/094994f2bfd154d565c34f5d24a7ade00e0c5bdb/src/policy_gradients/torch_utils.py#L494
    Fills the input `Tensor` using the orthogonal initialization scheme from OpenAI
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor
    Examples:
        >>> w = torch.empty(3, 5)
        >>> orthogonal_init(w)
    '''
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


class Critic(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        # -------- Initialize variables --------

        self.is_cont    = config.env.is_continuous
        self.device     = device
        
        self.m = nn.Sequential(
            nn.Linear(config.env.state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )          
        
        self.apply(init_orthogonal_weights)

    def forward(self, state):

        return self.m(state)
    


class Actor(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        # -------- Initialize variables --------

        self.is_cont    = config.env.is_continuous
        self.device     = device

        if self.is_cont:
            # if action space is defined as continuous, make variance
            self.action_dim = config.env.action_dim
            self.action_std = config.network.action_std_init
            self.action_var = torch.full((self.action_dim, ), config.network.action_std_init ** 2).to(self.device)
            
        self.m = nn.Sequential(
            nn.Linear(config.env.state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, config.env.action_dim),
            nn.Tanh()
        )
        
        self.apply(init_orthogonal_weights)

    def action_decay(self, action_std_decay_rate, min_action_std):
        # Change the action variance

        if self.is_cont:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std

            self.action_var = torch.full((self.action_dim, ), self.action_std ** 2).to(self.device)
        else:
            print("[Warning] Calling Actor::set_action_std() on discrete action space policy")

    def set_action_std(self, action_std):
        self.action_std = action_std
        self.action_var = torch.full((self.action_dim, ), self.action_std ** 2).to(self.device)

    def forward(self, state, action=None):

        if self.is_cont:
            # continuous space action 
            action_mean = self.m(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            # discrete space action
            action_probs = self.m(state)
            dist = Categorical(action_probs)

        # Get (action, action's log probs, estimated Value)
        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy()

