#!/usr/bin/env python


import numpy as np
from AminoAcid import aminoAcidDict
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from numpy.random import seed

# Make sure random numbers are always the same (for reproducibility)
randomFix = 100
seed(randomFix)


def createModel(pathToTrainingData):
    # Create a list of sequences and a 0/1-list where position i is 1 if sequence i is a binder and 0 if it isn't
    sequencesList = []
    isBinderList = []
    with open(pathToTrainingData, "r") as i:
        for e in i:
            lineSplit = e.split()
            if lineSplit[0] == "Peptide":  # skip the first line
                pass
            else:
                sequencesList.append(lineSplit[0])
                isBinderList.append(1 if lineSplit[2] is "1" else 0)
                # Create a 3D array of all the data necessary for the ANN
                # Its shape is 726*10*6: There are 726 sequences in the training set,
                # each sequence consists of 9 residues plus a boolean telling us if it's a binder
                # and 6 properties are considered for each residue:
                # Its one-letter-code, weight, iep, hydrophobicity, polarity, and its area
    data = np.array([[(aminoAcidDict.get(residue).one_letter_code,
                       aminoAcidDict.get(residue).weight,
                       aminoAcidDict.get(residue).iep,
                       aminoAcidDict.get(residue).hydrophobicity,
                       aminoAcidDict.get(residue).polarity,
                       aminoAcidDict.get(residue).area)
                      for residue in sequence] for sequence in sequencesList])
    # Generate actual ANN input: one-letter-code is cut away, now there are only five numbers for each residue
    annInput = data[:, :, 1:6].reshape(726, 45)

    # Setup model
    model = Sequential()  # Input layer with 9*5=45 input nodes - each property at every position
    model.add(Dense(45, kernel_initializer='uniform', activation='softplus', input_shape=(45,)))

    # Hidden layer with 17 nodes
    model.add(Dense(17, kernel_initializer='uniform', activation='softplus'))

    # Output layer with 1 node
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(annInput, isBinderList, epochs=100, verbose=0, batch_size=100)
    return model
