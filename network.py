import torch
import torch.nn as nn

def layer_init(layer, std=1.41, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, device, input_dim, first_layer_dim, second_layer_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            # state dimension is 4 for the following:
            # cart pos, cart velocity, pole angle, pole angular velocity
            layer_init(nn.Linear(input_dim, first_layer_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(first_layer_dim, second_layer_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(second_layer_dim, output_dim), std=0.01),
            # softmax to convert to probabilities of going left/right
            nn.Softmax(-1)
        )

        self.to(device)

    def forward(self, X):
        return torch.distributions.Categorical(self.model(X))


class Critic(nn.Module):
    def __init__(self, device, input_dim, first_layer_dim, second_layer_dim):
        super().__init__()
        self.model = nn.Sequential(
            layer_init(nn.Linear(input_dim, first_layer_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(first_layer_dim, second_layer_dim)),
            nn.ReLU(),
            # output is 1 because we want to predict the reward for being in a state
            layer_init(nn.Linear(second_layer_dim, 1), std=1) 
        )

        self.to(device)

    def forward(self, X):
        return self.model(X)