import torch
import torch.optim as optim
from music_rnn_model import MusicRNN
from data_preprocessing import prepare_sequences, create_dataloader


def train(model, dataloader, criterion, optimizer, n_epochs):
    model.train()

    for epoch in range(n_epochs):
        for inputs, targets in dataloader:
            # Convert data to suitable tensor format
            inputs = torch.from_numpy(inputs).float()
            targets = torch.from_numpy(targets).long()

            # Forward pass
            outputs, _ = model(inputs, None)
            loss = criterion(outputs.squeeze(), targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}/{n_epochs} Loss: {loss.item()}')
