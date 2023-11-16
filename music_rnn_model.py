import torch.nn as nn


class MusicRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(MusicRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)

        return out, hidden
