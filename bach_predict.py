import numpy as np
import sys
import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler

from LSTM_bach import LSTM_model, NotesDataset

from augmentation import createEncodedData


def predictNextNotes(input, steps, lstm_model, voices, scaler):
    # predicted notes
    predicted_notes = np.zeros((1, 4))

    # all unique notes for each voice
    unique_voice1 = np.unique(voices[:, 0])
    unique_voice2 = np.unique(voices[:, 1])
    unique_voice3 = np.unique(voices[:, 2])
    unique_voice4 = np.unique(voices[:, 3])

    # prepare input
    input = torch.tensor(input, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        for i in tqdm(range(steps)):
            # print(input.shape)
            pred1, pred2, pred3, pred4 = lstm_model(input, stateful=False)
            # print(output)
            pred1 = pred1.detach().numpy().squeeze()
            pred2 = pred2.detach().numpy().squeeze()
            pred3 = pred3.detach().numpy().squeeze()
            pred4 = pred4.detach().numpy().squeeze()

            # get the notes of the indices with highest value from model forward output
            note_voice1 = unique_voice1[np.argmax(pred1)]
            note_voice2 = unique_voice2[np.argmax(pred2)]
            note_voice3 = unique_voice3[np.argmax(pred3)]
            note_voice4 = unique_voice4[np.argmax(pred4)]
            # print(note_voice1, note_voice2, note_voice3, note_voice4)

            # add to array
            next_notes = np.array(
                [[note_voice1, note_voice2, note_voice3, note_voice4]])
            predicted_notes = np.concatenate(
                (predicted_notes, next_notes), axis=0)

            # change input
            # drop oldest notes
            input = input[0][1:]
            # encode and scale
            next_notes = createEncodedData(next_notes)
            next_notes = scaler.transform(next_notes)
            # concat predicted notes
            input = torch.cat((input, torch.Tensor(next_notes)))
            input = input.unsqueeze(0)

    return (predicted_notes.astype(np.int32)[1:])


def main(argv):
    if (argv == None or len(argv) != 1):
        print("Usage: python3 bach_inference.py *path/to/model.pth*")
        sys.exit()

    # define parameters used here
    # sliding window size
    # window_size = 100
    # hidden_size = 128
    window_size = 100
    hidden_size = 128
    input_size = 20
    output_size = [22, 27, 23, 26]
    num_layers = 2
    # train/test split, to continue predicting
    split_size = 0.0
    batch_size = 64

    # initialize model
    # model = LSTM_model(input_size, output_size,
    #                    hidden_size, num_layers, batch_size)
    model = LSTM_model(input_size=20, output_sizes=[22, 27, 23, 26],
                       hidden_size=256, num_layers=2, batch_size=32)
    # load model file, I use CPU, can be done on gpu
    model.load_state_dict(torch.load(
        argv[0], map_location=torch.device("cpu")))

    # load data, 4 voices of instruments
    voices = np.loadtxt("input/contrapunctusXIV.txt")
    # voices = np.load('input/Jsb16thSeparated.npz',
    #                  allow_pickle=True, encoding='latin1')["train"][0]

    # encode voices
    encoded_voices = createEncodedData(voices)

    # Train/test split (needed for correct scaling of new data)
    dataset_size = voices.shape[0]
    indices = list(range(dataset_size))
    split = int(np.floor((1 - split_size) * dataset_size))
    train_indices = indices[:split]
    # create split in data
    train_voices = encoded_voices[train_indices, :]

    # fit the scaler to the train data
    scaler = StandardScaler()
    scaler.fit(train_voices)
    # scale voices
    # voices = scaler.transform(voices)
    train_voices = scaler.transform(train_voices)

    # take last sliding window in data and infer from there
    input = train_voices[-window_size:]
    steps = 1000
    new_music = predictNextNotes(input, steps, model, voices, scaler)

    # save new music
    model_name = argv[0].replace("models/", "")
    os.makedirs("output/" + model_name)
    np.savetxt(fname="output/" + model_name +
               "/output.txt", X=new_music, fmt="%d")


if __name__ == '__main__':
    torch.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(precision=3)
    torch.set_printoptions(sci_mode=False)
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(precision=3)

    main(sys.argv[1:])
