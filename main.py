from data_preprocessing import get_notes, prepare_sequences, create_dataloader
from music_rnn_model import MusicRNN
from train import train
from generater import generate_music, create_midi
import torch.nn as nn
import torch.optim as optim
import torch
import os


def main():
    # Load or prepare your dataset
    notes = get_notes()  # This function should return all notes and chords from your MIDI files
    pitchnames = sorted(set(notes))  # Get all unique notes and chords
    n_vocab = len(pitchnames)  # The number of unique notes and chords
    network_input, network_output = prepare_sequences(
        notes, sequence_length=100)
    dataloader = create_dataloader(network_input, network_output)

    # Define the model architecture parameters
    input_size = 1  # This depends on how you preprocess your data
    output_size = len(set(notes))  # The total number of unique notes
    hidden_dim = 512  # Size of the hidden layer, can be adjusted
    n_layers = 2  # Number of LSTM layers, can be adjusted

    # Initialize the model
    # model = MusicRNN(input_size, output_size, hidden_dim, n_layers)
    model = MusicRNN(input_size, n_vocab, hidden_dim, n_layers)
    # model.load_state_dict(torch.load('./model_state_dict.pth'))
    model_path = './model_state_dict.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
        print("Loaded model state from '{}'".format(model_path))
    else:
        print("No model state file found at '{}'. Starting with a fresh model.".format(
            model_path))

    model = model.float()  # Explicitly setting model to float32
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    n_epochs = 50
    # train(model, dataloader, criterion, optimizer, n_epochs)

    # Generate music
    prediction_output = generate_music(
        model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output, 'generated_music.mid')


if __name__ == "__main__":
    main()
