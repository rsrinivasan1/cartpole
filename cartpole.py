import gym
import numpy as np
import torch
import torch.nn as nn
from memory import PPOMemory
from actor import Actor
from critic import Critic


# interface to store info from step taken
def remember(state, action, probs, vals, reward, done):
    memory.store_memory(state, action, probs, vals, reward, done)

def calculate_actor_loss(old_log_probs, log_probs, adv_batch):
    ratio = (log_probs - old_log_probs).exp()
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv_batch

    loss = torch.min(ratio * adv_batch, clipped).mean()
    return -loss

# takes in numpy array of state, returns output of actor and critic
def choose_action(state):
    state = torch.tensor(state, dtype=torch.float).to(device).unsqueeze(0)
    distribution = actor(state)
    value = critic(state)
    action = distribution.sample()

    prob_action = distribution.log_prob(action).item()
    action = action.item()
    value = value.item()

    return action, prob_action, value

def step(actor_optim, critic_optim, loss):
    actor_optim.zero_grad()
    critic_optim.zero_grad()
    loss.backward()
    actor_optim.step()
    critic_optim.step()

def learn():
    for _ in range(n_epochs):
        # create batches from stored memory
        # numpy arrays
        states_arr, actions_arr, old_probs_arr, values_arr, rewards_arr, dones_arr, batches = memory.generate_batches()

        # calculate advantage for every state in memory
        advantage = np.zeros_like(rewards_arr)
        # get each A_t BEFORE using shuffled batches, so that continuity of states is not broken
        for t in range(len(rewards_arr) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards_arr) - 1):
                # discount = (gamma * gae_lambda) ^ (k - t)
                # A_t = sum of discount * (r_t + gamma * V(s_t+1) * (1 - done_t))
                # no more extra rewards if done, just discount * rewards_arr[k] - values[k]
                a_t += discount * (rewards_arr[k] + gamma * values_arr[k + 1] * (1 - int(dones_arr[k])) - values_arr[k])
                discount *= gamma * gae_lambda
                if dones_arr[k]:
                    # reset discount to 1 if episode ends
                    discount = 1
            advantage[t] = a_t

        advantage = torch.tensor(advantage).to(device)
        values = torch.tensor(values_arr).to(device)

        for batch in batches:
            states = torch.tensor(states_arr[batch], dtype=torch.float).to(device)
            old_probs = torch.tensor(old_probs_arr[batch], dtype=torch.float).to(device)
            actions = torch.tensor(actions_arr[batch], dtype=torch.long).to(device)

            distribution = actor(states)
            critic_value = critic(states)

            new_probs = distribution.log_prob(actions)
            
            actor_loss = calculate_actor_loss(old_probs, new_probs, advantage[batch])

            # total predicted reward of the state = advantage + value
            returns = advantage[batch] + values[batch]
            critic_loss = (returns - critic_value).pow(2).mean()

            c_1 = 0.5
            total_loss = actor_loss + c_1 * critic_loss

            step(actor_optim, critic_optim, total_loss)

    memory.clear_memory()


# configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor = Actor(device, 4, 256, 256, 2)
critic = Critic(device, 4, 256, 256)
actor_optim = torch.optim.Adam(actor.parameters(), lr=0.0003)
critic_optim = torch.optim.Adam(critic.parameters(), lr=0.0003)

gamma = 0.99
eps = 0.2
gae_lambda = 0.95
batch_size = 5
N = 20
n_epochs = 4
n_games = 300

memory = PPOMemory(batch_size)

env = gym.make('CartPole-v1')


def run():
    best_score = env.reward_range[0]
    prev_scores = []
    num_steps = 0

    # want to learn every N games
    for i in range(n_games):
        state = env.reset()[0]
        done = False
        score = 0
        while not done:
            action, prob, val = choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            num_steps += 1
            score += reward
            # store this observation
            remember(state, action, prob, val, reward, done)
            if num_steps % N == 0:
                # actually backpropagate
                learn()

            state = next_state
        prev_scores.append(score)
        mean_score = np.mean(prev_scores[-100:])

        print(f"Episode {i}, score: {score}, mean score: {mean_score}")
        if mean_score > best_score:
            best_score = mean_score
            print(f"Best average score over 100 trials: {best_score}")


if __name__ == "__main__":
    run()