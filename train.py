import torch
import torch.optim as optim
from music_rnn_model import MusicRNN
from data_preprocessing import prepare_sequences, create_dataloader


def train(model, dataloader, criterion, optimizer, n_epochs):
    model.train()

    for epoch in range(n_epochs):
        for inputs, targets in dataloader:
            inputs = inputs.float()  # Ensure inputs are float32
            targets = targets.long()  # Targets usually are long for classification

            # Forward pass
            outputs, _ = model(inputs, None)
            loss = criterion(outputs.squeeze(), targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}/{n_epochs} Loss: {loss.item()}')
