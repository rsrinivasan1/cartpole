import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, device, input_dim, first_layer_dim, second_layer_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            # state dimension is 4 for the following:
            # cart pos, cart velocity, pole angle, pole angular velocity
            nn.Linear(input_dim, first_layer_dim),
            nn.ReLU(),
            nn.Linear(first_layer_dim, second_layer_dim),
            nn.ReLU(),
            nn.Linear(second_layer_dim, output_dim),
            # softmax to convert to probabilities of going left/right
            nn.Softmax(-1)
        )

        self.to(device)

    def forward(self, X):
        return torch.distributions.Categorical(self.model(X))