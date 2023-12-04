import numpy as np
import math
import sys
import time
from collections import OrderedDict
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from tensorboard.plugins.hparams import api as hp
from augmentation import createEncodedData
from tqdm import tqdm

# gpu if available (global variable for convenience)
device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))

# to find float index in unique float list of standard scaled array
# works also for ints when not scaled


def uniqueLocation(uniques, note):
    for index, unique in enumerate(uniques):
        if (math.isclose(unique, note, abs_tol=0.0001)):
            return index
    return None


# find unique notes in a set of voices
def findUniques(voices: np.ndarray) -> list:
    uniques = []

    # for each voice
    for voice in voices.T:
        unique_notes = np.unique(voice)
        uniques.append(unique_notes)

    return uniques

# find unique note union of all sequences that are prepared for the dataloader


def findUniquesSequences(sequences):
    print(sequences.shape)
    pass

# returns concatenated onehot encoding for each note


# returns concatenated onehot encoding for each note
def one_hot_encode(y: np.ndarray, uniques: list) -> np.ndarray:
    # initialize return arrays
    one_hot_voice1 = np.zeros((y.shape[0], len(uniques[0])), dtype=np.float32)
    one_hot_voice2 = np.zeros((y.shape[0], len(uniques[1])), dtype=np.float32)
    one_hot_voice3 = np.zeros((y.shape[0], len(uniques[2])), dtype=np.float32)
    one_hot_voice4 = np.zeros((y.shape[0], len(uniques[3])), dtype=np.float32)

    # one hot encode each note
    for timestep, notes in enumerate(y):
        for voice, note in enumerate(notes):
            if (voice == 0):
                # get location in uniques of current note
                one_hot_location = uniqueLocation(uniques[0], note)
                one_hot_voice1[timestep][one_hot_location] = 1
            elif (voice == 1):
                one_hot_location = uniqueLocation(uniques[1], note)
                one_hot_voice2[timestep][one_hot_location] = 1
            elif (voice == 2):
                one_hot_location = uniqueLocation(uniques[2], note)
                one_hot_voice3[timestep][one_hot_location] = 1
            elif (voice == 3):
                one_hot_location = uniqueLocation(uniques[3], note)
                one_hot_voice4[timestep][one_hot_location] = 1

    return one_hot_voice1, one_hot_voice2, one_hot_voice3, one_hot_voice4

# encoded voices used as input data (x), non_encoded_voices as target data (y),
#   voices used to one hot encode (necessary for nr of uniques in data),
#   as encoded and non_encoded is a train/test subset of the complete set and might miss some unique values


class NotesDataset(Dataset):
    def __init__(self, window_size: int, skip_steps: int, input_size: int, uniques: list):
        # keep window size and skip steps consistent for each sequence
        self.window_size = window_size
        self.skip_steps = skip_steps
        # make sure every sequence has same shape
        self.input_size = input_size
        self.uniques = uniques

        # init nr_samples as 0
        self.nr_samples = 0

        # init tensors
        # self.x = torch.empty((1, self.window_size, input_size), dtype=torch.float32).to(device)
        self.x = torch.empty((0, window_size, input_size),
                             dtype=torch.float32).to(device)
        self.y1 = torch.empty(
            (0, len(uniques[0])), dtype=torch.float32).to(device)
        self.y2 = torch.empty(
            (0, len(uniques[1])), dtype=torch.float32).to(device)
        self.y3 = torch.empty(
            (0, len(uniques[2])), dtype=torch.float32).to(device)
        self.y4 = torch.empty(
            (0, len(uniques[3])), dtype=torch.float32).to(device)

    # add sequences through this function, which allows the dataset to store multiple sequences in sliding window fashion (with skip steps)
    def addSequence(self, encoded_voices: np.ndarray, non_encoded_voices: np.ndarray):
        # nr of samples, and nr of voices
        nr_samples = encoded_voices.shape[0] - self.window_size
        input_width = encoded_voices.shape[1]
        output_width = non_encoded_voices.shape[1]

        # shape check for input
        if (self.input_size != input_width):
            raise ValueError(
                f"Added sequence input dimensions do not match with initialized ({self.input_size} != {input_width})")

        # initialize x data -> window_size amount of notes of 4 (=20 when encoded) voices each per prediction
        x = np.zeros((nr_samples, self.window_size,
                     input_width), dtype=np.float32)
        for i in range(x.shape[0]):
            x[i] = encoded_voices[i: i + self.window_size]

        # initialize y data -> 4 following target notes per time window
        y = np.zeros((nr_samples, output_width), dtype=np.float32)
        for j in range(y.shape[0]):
            y[j] = non_encoded_voices[j + self.window_size]

        # one hot encode different task (differnt voices) target values
        y1, y2, y3, y4 = one_hot_encode(y, self.uniques)

        # create tensors
        self.x = torch.cat((self.x, torch.from_numpy(x).to(device)), 0)
        self.y1 = torch.cat((self.y1, torch.from_numpy(y1).to(device)), 0)
        self.y2 = torch.cat((self.y2, torch.from_numpy(y2).to(device)), 0)
        self.y3 = torch.cat((self.y3, torch.from_numpy(y3).to(device)), 0)
        self.y4 = torch.cat((self.y4, torch.from_numpy(y4).to(device)), 0)

        # add nr of samples to total
        self.nr_samples += nr_samples

    def __getitem__(self, index: int):
        sample = {'x': self.x[index], 'y1': self.y1[index],
                  'y2': self.y2[index], 'y3': self.y3[index], 'y4': self.y4[index]}
        return sample

    def __len__(self):
        return self.nr_samples


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        """
        lstm_output: [batch_size, seq_len, hidden_size]
        """
        # Applying linear layer and tanh activation
        energy = self.tanh(self.linear(lstm_output))
        # Calculating attention scores
        attention_scores = self.softmax(energy)
        # Producing the weighted sum of lstm_output (context vector)
        context_vector = torch.sum(attention_scores * lstm_output, dim=1)
        return context_vector


# LSTM model with four output heads, one for each voice next note prediction (task)
# The model can be set to stateful, meaning the internal hidden state and cell state is passed
#   into the model each batch and reset once per epoch.


class LSTM_model(nn.Module):
    def __init__(self, input_size, output_sizes, hidden_size, num_layers, batch_size):
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # lstm layer(s)
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, dropout=0.2, batch_first=True)

        # task head: voice 1
        self.head1 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(hidden_size, hidden_size)),
             ('relu', nn.ReLU()),
             ('dropout', nn.Dropout(0.5)),
             ('final', nn.Linear(hidden_size, output_sizes[0]))]
        ))

        # task head: voice 2
        self.head2 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(hidden_size, hidden_size)),
             ('relu', nn.ReLU()),
             ('dropout', nn.Dropout(0.5)),
             ('final', nn.Linear(hidden_size, output_sizes[1]))]
        ))

        # task head: voice 3
        self.head3 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(hidden_size, hidden_size)),
             ('relu', nn.ReLU()),
             ('dropout', nn.Dropout(0.5)),
             ('final', nn.Linear(hidden_size, output_sizes[2]))]
        ))

        # task head: voice 4
        self.head4 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(hidden_size, hidden_size)),
             ('relu', nn.ReLU()),
             ('dropout', nn.Dropout(0.5)),
             ('final', nn.Linear(hidden_size, output_sizes[3]))]
        ))

        print("LSTM initialized with {} input size, {} hidden layer size, {} number of LSTM layers, and an output size of {}".format(
            input_size, hidden_size, num_layers, output_sizes))
        # reset states in case of stateless use
        self.reset_states(batch_size)

    # reset hidden state and cell state, should be before each new sequence
    #   In our problem: every epoch, as it is one long sequence
    def reset_states(self, batch_size):
        # def reset_states(self):
        # hidden state and cell state for LSTM
        self.hn = torch.zeros(self.num_layers,  batch_size,
                              self.hidden_size).to(device)
        self.cn = torch.zeros(self.num_layers, batch_size,
                              self.hidden_size).to(device)

    def forward(self, input, stateful):
        # simple forward function
        # stateful = keep hidden states entire sequence length
        # only use when 2 samples follow temporally (first timepoint from 2nd sample follows from last timepoint 1st sample)
        if stateful:
            # for last batch which might not be the same shape
            if (input.size(0) != self.hn.size(1)):
                self.reset_states(input.size(0))

            # lstm layer
            out, (self.hn, self.cn) = self.lstm(
                input, (self.hn.detach(), self.cn.detach()))
            # linear output layers for each head
            task_head1_out = self.head1(out[:, -1, :])
            task_head2_out = self.head2(out[:, -1, :])
            task_head3_out = self.head3(out[:, -1, :])
            task_head4_out = self.head4(out[:, -1, :])
        else:
            # initiaze hidden and cell states
            hn = torch.zeros(self.num_layers,  input.size(0),
                             self.hidden_size).to(device)
            cn = torch.zeros(self.num_layers, input.size(0),
                             self.hidden_size).to(device)
            # lstm layer
            out, (hn, cn) = self.lstm(input, (hn, cn))
            # linear output layers for each head
            task_head1_out = self.head1(out[:, -1, :])
            task_head2_out = self.head2(out[:, -1, :])
            task_head3_out = self.head3(out[:, -1, :])
            task_head4_out = self.head4(out[:, -1, :])

        return task_head1_out, task_head2_out, task_head3_out, task_head4_out


def training(model, train_loader: DataLoader, test_loader: DataLoader, nr_epochs, optimizer, loss_func, scheduler, stateful, writer):
    # lowest train/test loss, train/test loss lists
    lowest_train_loss = np.inf
    lowest_test_loss = np.inf
    train_losses = []
    test_losses = []

    # test_loss declaration untill assigned in model evaluation (used in progress bar print)
    test_loss = "n/a"

    # training loop
    for epoch in (progress_bar := tqdm(range(1, nr_epochs))):
        # add epoch info to progress bar
        progress_bar.set_description(f"Epoch {epoch}")

        # reset lstm hidden and cell state (stateful lstm = reset states once per sequence)
        # if not, reset automatically each forward call
        if stateful:
            model.reset_states(train_loader.batch_size)

        # reset running loss
        running_loss_train = 0
        running_loss_test = 0

        # train loop
        model.train()
        for data in train_loader:
            # reset gradient function of weights
            optimizer.zero_grad()
            # forward
            voice1_pred, voice2_pred, voice3_pred, voice4_pred = model(
                data["x"], stateful)
            # calculate loss
            loss = loss_func(voice1_pred, data["y1"]) + loss_func(voice2_pred, data["y2"]) + loss_func(
                voice3_pred, data["y3"]) + loss_func(voice4_pred, data["y4"])
            # backward, retain_graph = True needed for hidden lstm states
            loss.backward(retain_graph=True)
            # step
            optimizer.step()
            # add to running loss
            running_loss_train += loss.item()

        # learning rate scheduler step
        scheduler.step()

        # calc running loss
        train_loss = running_loss_train/len(train_loader)
        train_losses.append(train_loss)

        # add loss to tensorboard
        writer.add_scalar("Running train loss", train_loss, epoch)

        # check if lowest loss
        if (train_loss < lowest_train_loss):
            # Save model
            torch.save(model.state_dict(), "models/model" +
                       str(train_loader.dataset.x.shape[1]) + str(model.hidden_size) + ".pth")
            # torch.save(model.state_dict(), "drive/MyDrive/colab_outputs/lstm_bach/models/model" + str(train_loader.dataset.x.shape[1]) + str(model.hidden_size) + ".pth")

        # Test evaluation
        if (test_loader):
            # model.eval()
            with torch.no_grad():
                for data in test_loader:
                    # forward pass
                    voice1_pred, voice2_pred, voice3_pred, voice4_pred = model(
                        data["x"], stateful)
                    # calculate loss
                    loss = loss_func(voice1_pred, data["y1"]) + loss_func(voice2_pred, data["y2"]) + loss_func(
                        voice3_pred, data["y3"]) + loss_func(voice4_pred, data["y4"])
                    # add to running loss
                    running_loss_test += loss.item()

            # calc running loss
            test_loss = running_loss_test/len(test_loader)
            test_losses.append(test_loss)

            # add test loss to tensorboard
            writer.add_scalar("Running test loss", test_loss, epoch)

            # if lowest till now, save model (checkpointing)
            if (test_loss < lowest_test_loss):
                torch.save(model.state_dict(), "models/model" + str(
                    train_loader.dataset.x.shape[1]) + str(model.hidden_size) + "test" + ".pth")
                # torch.save(model.state_dict(), "drive/MyDrive/colab_outputs/lstm_bach/models/model" + str(train_loader.dataset.x.shape[1]) + str(model.hidden_size) + "test" + ".pth")

        # before next epoch: add last epoch info to progress bar
        progress_bar.set_postfix(
            {"train_loss": train_loss, "test_loss": test_loss})

    return train_losses, test_losses

# create train and test dataset based on window size where one window of timesteps
#   will predict the subsequential single timestep
# Data is created without any information leak between test/train (either scaling leak or time leak)


def createDataLoadersSequences(sequences, encoded_voices, split_size, window_size, skip_steps, batch_size):
    # get unique notes for each voices, for one-hot-encoding/output size
    uniques = findUniquesSequences(sequences)

# create train and test dataset based on window size where one window of timesteps
#   will predict the subsequential single timestep
# Data is created without any information leak between test/train (either scaling leak or time leak)


def createDataLoaders(voices, encoded_voices, split_size, window_size, skip_steps, batch_size):
    # get unique notes for each voices, for one-hot-encoding/output size
    uniques = findUniques(voices)

    # Train/test split
    dataset_size = voices.shape[0]
    indices = list(range(dataset_size))
    split = int(np.floor((1 - split_size) * dataset_size))
    train_indices, test_indices = indices[:split], indices[split:]

    # create split in data, using encoded data for x (input), and non encoded for y (target)
    train_voices_x = encoded_voices[train_indices, :]
    train_voices_y = voices[train_indices, :]
    test_voices_x = encoded_voices[test_indices, :]
    test_voices_y = voices[test_indices, :]

    # scale both sets, using training data as fit (no leaks)
    scaler = StandardScaler()
    scaler.fit(train_voices_x)
    train_voices_x = scaler.transform(train_voices_x)
    # all_voices = scaler.transform(voices)

    # set input/output sizes
    input_size = encoded_voices.shape[1]
    output_sizes = [len(uniques[0]), len(uniques[1]),
                    len(uniques[2]), len(uniques[3])]

    # create train dataset
    train_dataset = NotesDataset(window_size, skip_steps, input_size, uniques)
    train_dataset.addSequence(train_voices_x, train_voices_y)

    # create train dataloader
    train_loader = DataLoader(train_dataset, batch_size)

    # Do the same for test set
    if (split_size > 0):
        # scale test set
        test_voices_x = scaler.transform(test_voices_x)
        # create test dataset
        test_dataset = NotesDataset(
            window_size, skip_steps, input_size, uniques)
        test_dataset.addSequence(test_voices_x, test_voices_y)

        # create test dataloader
        test_loader = DataLoader(test_dataset, batch_size)
    else:
        test_loader = None

    return train_loader, test_loader


def main():
    # load data, 4 voices of instruments
    voices = np.loadtxt("input/contrapunctusXIV.txt")
    # print(voices0)
    # voices = np.load('input/Jsb16thSeparated.npz',
    #                  allow_pickle=True, encoding='latin1')["train"][0]
    # voices = np.load('input/js-fakes-16thSeparated.npz',
    #                  allow_pickle=True, encoding='latin1')

    # remove starting silence, does not promote learning
    # data shape is (3816, 4) after
    voices = np.delete(voices, slice(8), axis=0)
    print("Data shape (4 voices):", voices.shape)

    # encode data with additional harmonic information
    encoded_voices = createEncodedData(voices)

    # batch_size for training network
    batch_size = 32

    # split size of test/train data
    split_size = 0.1

    # hyperparameters for fine-tuning
    # window_size = sliding window on time-sequence data for input
    # skip steps = steps the window slides each time
    # hidden_size = hidden units of lstm layer(s)
    # conv_channels = number of channels in the first conv layer (multiplied by 2 every next layer)
    # nr_layers = number of lstm layers stacked after each other
    hyperparams = dict(
        window_size=[96],
        skip_steps=[16],
        hidden_size=[256],
        nr_layers=[2],
        l2=[0.06]
    )
    # sets of combinations of hparams
    hyperparam_value_sets = product(*[value for value in hyperparams.values()])

    # Loop through different combinations of the hyperparameters
    for run_id, (window_size, skip_steps, hidden_size, nr_layers, l2) in enumerate(hyperparam_value_sets):
        # tensorboard summary writer
        writer = SummaryWriter(
            f'runs/E_window_size={window_size}_hidden_size={hidden_size}_l2={l2}')
        # writer = SummaryWriter(f'drive/MyDrive/colab_outputs/lstm_bach/runs/window_size={window_size}_hidden_size={hidden_size}_l2={l2}')

        # Split data in train and test, scale, create datasets and create dataloaders
        train_loader, test_loader = createDataLoaders(
            voices, encoded_voices, split_size, window_size, skip_steps, batch_size)

        # some informational print statements
        print("\nNew run window/hidden/l2/batch_size:", window_size,
              "/", hidden_size, "/", l2, "/", batch_size)
        data = next(iter(train_loader))
        print("Input size:", data["x"].size(),
              "- Output size:[", data["y1"].size(), data["y2"].size(
        ), data["y3"].size(), data["y4"].size(), "]\n",
            "TRAIN batches:", len(train_loader),
            "- TEST batches:", len(test_loader) if test_loader else "Not available")
        # Input/output dimensions
        input_size = encoded_voices.shape[1]
        output_sizes = [data["y1"].size(1), data["y2"].size(
            1), data["y3"].size(1), data["y4"].size(1)]

        # create model
        lstm_model = LSTM_model(input_size, output_sizes,
                                hidden_size, nr_layers, batch_size)

        # loss function and optimizer
        #   Output of each head is multi-class classification -> cross entropy
        loss_func = nn.CrossEntropyLoss()
        # AdamW = Adam with fixed weight decay (weight decay performed after controlling parameter-wise step size)
        optimizer = optim.AdamW(lstm_model.parameters(),
                                lr=0.001, weight_decay=l2)
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.5, total_iters=750)

        # to gpu if possible
        lstm_model = lstm_model.to(device)

        # training loop
        epochs = 1000

        # In this example we should not use a stateful lstm, as the next samples (subsequent sliding windows) do not follow directly from the current.
        # This is only the case when the first sample is (for Ex.) [1:10] which is the first window, and [11:20] the next, and so on.
        # With our data it would be: [1:10] and the next [2:11]. Target value does not matter necessarily.
        # More explanation: https://stackoverflow.com/questions/58276337/proper-way-to-feed-time-series-data-to-stateful-lstm
        #   unfortunately I implemented stateful before knowing these in and outs.
        # One possible way to use stateful is to use window_size = 1
        stateful = True
        train_losses, test_losses = training(
            lstm_model, train_loader, test_loader, epochs, optimizer, loss_func, scheduler, stateful, writer)

        # flush tensorboard writer
        writer.flush()


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(precision=4, suppress=True, linewidth=np.nan)
    torch.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(precision=4, sci_mode=False)

    main()
