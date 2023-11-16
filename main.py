from data_preprocessing import get_notes, prepare_sequences, create_dataloader
from music_rnn_model import MusicRNN
from train import train
import torch.nn as nn
import torch.optim as optim


def main():
    # Load and preprocess the data
    notes = get_notes()
    network_input, network_output = prepare_sequences(
        notes, sequence_length=100)
    dataloader = create_dataloader(network_input, network_output)

    # Define the model architecture parameters
    input_size = 1  # This depends on how you preprocess your data
    output_size = len(set(notes))  # The total number of unique notes
    hidden_dim = 512  # Size of the hidden layer, can be adjusted
    n_layers = 2  # Number of LSTM layers, can be adjusted

    # Initialize the model
    model = MusicRNN(input_size, output_size, hidden_dim, n_layers)
    model = model.float()  # Explicitly setting model to float32
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    n_epochs = 50
    train(model, dataloader, criterion, optimizer, n_epochs)


if __name__ == "__main__":
    main()
