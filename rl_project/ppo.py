import torch
from torch import nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, obs):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(self, obs, actions):
        logits, values = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values
