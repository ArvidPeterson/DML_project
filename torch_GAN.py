from __future__ import print_function, division
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
from music21 import converter, instrument, note, chord, stream
# from keras.layers import Input, Dense, Reshape, Dropout, CuDNNLSTM, Bidirectional, LSTM
# from keras.layers import BatchNormalization, Activation, ZeroPadding2D
# from keras.layers.advanced_activations import LeakyReLU
# from keras.models import Sequential, Model
# from keras.optimizers import Adam
# from keras.utils import np_utils
import torch
import torch_GAN
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import os

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """
    # stolen KERAS sourcecode

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def get_notes(n_notes=3):
    """ Get all the notes and chords from the midi files """
    notes = []
    # import pdb; pdb.set_trace()
    for ii, file in enumerate(glob.glob("data/maestro-v2.0.0/2004/*.midi")):
        if ii > n_notes:
            break
        pickle_file_name = file[:-4] + 'pkl'

        if os.path.isfile(pickle_file_name):
            print(f'Reading parsed file: {pickle_file_name}')
            with open(pickle_file_name, 'rb') as handle:
                midi = pickle.load(handle)
        else:
            midi = converter.parse(file)

            with open(pickle_file_name, 'wb') as handle:
                print(f'writing parsed file: {pickle_file_name}')
                unserialized_data = pickle.dump(midi, 
                    handle, 
                    protocol=pickle.HIGHEST_PROTOCOL
                    )


        print("Parsing %s" % file)

        notes_to_parse = None
        # TODO: files are too long
        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
            
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
        
    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # Normalize input between -1 and 1
    network_input = (network_input - float(n_vocab)/2) / (float(n_vocab)/2)
    # import pdb; pdb.set_trace()
    network_output = to_categorical(network_output)

    return (network_input, network_output)

def generate_notes(model, network_input, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(network_input)-1)
    
    # Get pitch names and store in a dictionary
    pitchnames = sorted(set(item for item in notes))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        
        pattern = np.append(pattern,index)
        #pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output
  
def create_midi(prediction_output, filename):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for item in prediction_output:
        pattern = item[0]
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='{}.mid'.format(filename))


class Discriminator(nn.Module):
    def __init__(self, n_units):
        super(Discriminator, self).__init__()
        self.sequence_length = n_units
        self.LSTM_hidden_dim = 512

        # self.LSTM_1 = nn.LSTM(
        #     self.sequence_length, 
        #     self.LSTM_hidden_dim
        #     )
        # self.LSTM_2 = nn.LSTM(
        #     self.LSTM_hidden_dim,
        #     self.LSTM_hidden_dim,
        #     bidirectional=True
        #     )

        self.LSTM = nn.LSTM(
            input_size=self.sequence_length, 
            hidden_size=self.LSTM_hidden_dim,
            num_layers=2,
            batch_first=True
            )

        self.linear_layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
            )
    
    def forward(self, x):
        # import pdb; pdb.set_trace()
        hidden_1 = torch.zeros(self.sequence_length)
        hidden_2 = torch.zeros(self.sequence_length)
        out, (h1, h2) = self.LSTM(x, (hidden_1, hidden_2))

        x = self.linear_layers(out)
        return x


class Generator(nn.Module):
    def __init__(self, n_units):
        super(Generator, self).__init__()
        self.sequence_length = n_units

        self.generating_layers = nn.Sequential(
            nn.Linear(self.sequence_length, 256),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(num_features=256), # TODO: Make batchnorm work at some time
            nn.Linear(256, 1024),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(num_features=1024),
            nn.Linear(1024, self.sequence_length)
        )
        
    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.generating_layers(x)
        return x
        
class torchGAN():

    def __init__(self, n_units):
        self.sequence_length = n_units
        
        # create discriminator and generator 
        self.discriminator = Discriminator(self.sequence_length)
        self.generator = Generator(self.sequence_length)

    
    def train(self, n_epochs, batch_size=128, sample_interval=50):
        notes = get_notes(n_notes=3)
        n_vocab = len(set(notes))
        X_train, _ = prepare_sequences(notes, n_vocab)
    
        # Adversarial ground truths
        label_real = np.ones((batch_size, 1))
        label_fake = np.zeros((batch_size, 1))
        
        # Training the model
        for i_epoch in range(n_epochs):

            # Training the discriminator
            # Select a random batch of note sequences
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_seqs = X_train[idx]
            import pdb; pdb.set_trace()
            print(f'epoch: {i_epoch}')
    
    def _generate(self):
        pass

    def _discriminate(self):
        pass
    
if __name__ == '__main__':

    gan = torchGAN(n_units=50)
    gan.train(n_epochs=3)

    print('done')
    

