import gym
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils import tensorboard


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # state dimension is 4 for the following:
            # cart pos, cart velocity, pole angle, pole angular velocity
            nn.Linear(4, 128),
            nn.Mish(),
            nn.Linear(128, 128),
            nn.Mish(),
            nn.Linear(128, 128),
            nn.Mish(),
            nn.Linear(128, 2),
            # softmax to convert to probabilities of going left/right
            nn.Softmax()
        )

    def forward(self, X):
        return self.model(X)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.Mish(),
            nn.Linear(128, 128),
            nn.Mish(),
            nn.Linear(128, 128),
            nn.Mish(),
            # output is 1 because we want to predict the reward for being in a state
            nn.Linear(128, 1)
        )

    def forward(self, X):
        return self.model(X)


# configuration
actor = Actor()
critic = Critic()
actor_optim = torch.optim.Adam(actor.parameters())
critic_optim = torch.optim.Adam(critic.parameters())
gamma = 0.98
eps = 0.2


def policy_loss(old_log_prob, log_prob, adv):
    ratio = (log_prob - old_log_prob).exp()
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv

    likelihood = torch.min(ratio * adv, clipped)
    return -likelihood


env = gym.make('CartPole-v1', render_mode="human")

for i in range(10000):
    prev_policy_action_probability = None
    state = env.reset()[0]
    done = False
    total_reward = 0

    while True:
        # Sample action from policy network
        distribution = torch.distributions.Categorical(actor(torch.from_numpy(state)))
        action = distribution.sample()
        prob_action = distribution.log_prob(action)

        next_state, reward, done, info, _ = env.step(action.numpy())
        total_reward += reward
        if done:
            break
        # Calculate discounted reward estimated based on critic network
        # Here we only perform one timestep before updating reward -> TD(0) algorithm
        discounted_rewards = reward + gamma * critic(torch.from_numpy(next_state))
        # Calculate advantage
        baseline_rewards = critic(torch.from_numpy(state))
        advantage = discounted_rewards - baseline_rewards

        state = next_state

        if prev_policy_action_probability:
            # Actor loss is based on clipped objective
            actor_loss = policy_loss(prev_policy_action_probability.detach(), prob_action, advantage.detach())
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            # Critic loss is based on squared advantage
            critic_loss = advantage.pow(2).mean()
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

        prev_policy_action_probability = prob_action
    print(total_reward)
