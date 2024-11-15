import gym
import numpy as np
import torch
import torch.nn as nn
from memory import PPOMemory
from network import Actor, Critic
from config import batch_size, learning_rate, n_epochs, gamma, gae_lambda, c_1, eps, N, n_games, layer_dims_actor, layer_dims_critic


def calculate_actor_loss(old_log_probs, log_probs, adv_batch):
    ratio = (log_probs - old_log_probs).exp()
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv_batch

    loss = torch.min(ratio * adv_batch, clipped).mean()
    return -loss


# takes in numpy array of states in env, returns output of actor and critic
def choose_actions(states, actor, critic):
    states = torch.tensor(states, dtype=torch.float).to(device)
    distributions = actor(states)
    values = critic(states)

    actions = distributions.sample()
    prob_actions = distributions.log_prob(actions)

    actions = actions.tolist()
    prob_actions = prob_actions.tolist()
    values = values.squeeze(-1).tolist()

    return actions, prob_actions, values


def step(actor_optim, critic_optim, loss):
    actor_optim.zero_grad()
    critic_optim.zero_grad()
    loss.backward()
    actor_optim.step()
    critic_optim.step()


# assumes you have N observations in memory, for each batch makes a step
def learn(actor, critic, actor_optim, critic_optim, memory, lr):
    # update lr for both optimizers:
    actor_optim.param_groups[0]['lr'] = lr
    critic_optim.param_groups[0]['lr'] = lr
    
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

            total_loss = actor_loss + c_1 * critic_loss

            step(actor_optim, critic_optim, total_loss)

    memory.clear_memory()


def make_env(gym_id):
    def thunk():
        return gym.make(gym_id)
    return thunk


def run(envs, actor, critic, actor_optim, critic_optim, memory, device, anneal_lr=True):
    best_score = envs.reward_range[0]
    prev_scores = []
    num_steps = 0

    # want to learn every N games
    for i in range(n_games):
        states = envs.reset()[0]
        done = False
        score = 0
        lr = learning_rate
        while not done:
            actions, probs, vals = choose_actions(states, actor, critic)
            next_states, rewards, dones, _, _ = envs.step(actions)
            num_steps += 1
            score += rewards[0]
            # store this observation
            memory.store_memory(states[0], actions[0], probs[0], vals[0], rewards[0], dones[0])

            if num_steps % N == 0:
                # anneal learning rate if specified
                if anneal_lr:
                    frac = 1 - (i / n_games)
                    lr = learning_rate * frac
                # actually backpropagate
                learn(actor, critic, actor_optim, critic_optim, memory, lr)
            states = np.expand_dims(next_states[0], axis=0)
            done = dones[0]
            
        prev_scores.append(score)
        mean_score = np.mean(prev_scores[-100:])

        print(f"Episode {i}, lr: {round(lr, 5)}, score: {score}, mean score: {mean_score}")
        if mean_score > best_score:
            best_score = mean_score
            print(f"Best average score over 100 trials: {best_score}")


if __name__ == "__main__":
    envs = gym.vector.SyncVectorEnv([make_env('CartPole-v1')])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(device, np.array(envs.single_observation_space.shape).prod(), layer_dims_actor, envs.single_action_space.n)
    critic = Critic(device, np.array(envs.single_observation_space.shape).prod(), layer_dims_critic)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=learning_rate, eps=1e-5)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=learning_rate, eps=1e-5)

    memory = PPOMemory(batch_size)

    run(envs, actor, critic, actor_optim, critic_optim, memory, device)