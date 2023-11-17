import torch.nn as nn


class MusicRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dense):
        super(MusicRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        # Two fully connected layers
        self.fc1 = nn.Linear(hidden_dim, dense)
        self.fc2 = nn.Linear(dense, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        # Apply the first fully connected layer and a ReLU activation
        out = F.relu(self.fc1(out))
        # Apply the second fully connected layer
        out = self.fc2(out)
        # Apply softmax on the last dimension to get probability distribution
        out = F.softmax(out, dim=-1)

        return out, hidden
