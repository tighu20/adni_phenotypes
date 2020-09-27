import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):

    def __init__(self, dim_in, dim_inner=128):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_inner)
        self.fc2 = nn.Linear(dim_inner, dim_inner)
        self.fc3 = nn.Linear(dim_inner, 1)

        self.activation = nn.Tanh()
        self.dropout_rate : float = 0.9

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.activation(self.fc2(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc3(x)

        return torch.sigmoid(x)
