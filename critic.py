import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, device, input_dim, first_layer_dim, second_layer_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, first_layer_dim),
            nn.ReLU(),
            nn.Linear(first_layer_dim, second_layer_dim),
            nn.ReLU(),
            # output is 1 because we want to predict the reward for being in a state
            nn.Linear(second_layer_dim, 1)
        )

        self.to(device)

    def forward(self, X):
        return self.model(X)