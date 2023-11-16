import numpy as np
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.utils import to_categorical
import glob
from music21 import converter, instrument, note, chord


def get_notes():
    notes = []

    for file in glob.glob("./dataset/*.mid"):
        midi = converter.parse(file)
        notes_to_parse = None

        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes


def prepare_sequences(notes, sequence_length):
    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number)
                       for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # Create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    normalized_input = np.reshape(
        network_input, (n_patterns, sequence_length, 1))

    # Normalize input
    normalized_input = normalized_input / float(len(pitchnames))

    # One-hot encode the output
    network_output = to_categorical(network_output)

    return (normalized_input, network_output)


class MusicDataset(Dataset):
    def __init__(self, network_input, network_output):
        self.network_input = network_input
        self.network_output = network_output

    def __len__(self):
        return len(self.network_input)

    def __getitem__(self, idx):
        return self.network_input[idx], self.network_output[idx]


def create_dataloader(network_input, network_output, batch_size=64):
    dataset = MusicDataset(network_input, network_output)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
