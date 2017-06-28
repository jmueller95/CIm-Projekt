#!/usr/bin/env python

import argparse
import numpy as np
from AminoAcid import aminoAcidDict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef
from numpy.random import seed  # Needed for seeding random numbers
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import os

# Make sure random numbers are always the same
seed(100)
# Parse the input file
parser = argparse.ArgumentParser(description="Occurence Matrix Generator")
parser.add_argument('input', metavar='input file')
args = parser.parse_args()
# Create a list of sequences and a 0/1-list where position i is 1 if sequence i is a binder and 0 if it isn't
sequencesList = []
isBinderList = []
with open(args.input, "r") as i:
    for e in i:
        lineSplit = e.split()
        if lineSplit[0] == "Peptide":  # skip the first line
            pass
        else:
            sequencesList.append(lineSplit[0])
            isBinderList.append(1 if lineSplit[2] is "1" else 0)
# Create a 3D array of all the data necessary for the ANN
# Its shape is 726*10*6, since there are 726 sequences in the training set, each sequence consists of 9 residues plus
# a boolean telling us if it's a binder and
# 6 properties are considered for each residue:
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

# Split data up into training and test sets
x_train, x_test, y_train, y_test = train_test_split(annInput, isBinderList, test_size=0.33, random_state=0)


# Function for creating a model - needed for the keras classifier
def create_model(neurons=1, init_mode='uniform', epochs=1, batch_size=1):
    model = Sequential()

    # Input layer with 9*5=45 input nodes - each property at every position
    model.add(Dense(45, kernel_initializer=init_mode, activation='softplus', input_shape=(45,)))

    # Hidden layer with 17 nodes
    model.add(Dense(17, kernel_initializer=init_mode, activation='softplus'))

    # Output layer with 1 node
    model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam')
    model.fit(x_train, y_train, epochs=epochs, verbose=1, batch_size=batch_size)
    return model


# Create classifier
classifier = KerasClassifier(build_fn=create_model)

# Define grid search parameters
epochs = [1, 10, 25, 50, 100]
batch_size = [100, 200, 300, 400, len(x_train)]
param_grid = dict(epochs=epochs, batch_size=batch_size)
grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1, scoring='roc_auc')
grid_result = grid.fit(x_train, y_train)

# Write results to file
with open(os.path.join(os.getcwd(), 'GridSearchResults2.txt'), 'w') as output:
    output.write("Best: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        output.write("%f (%f) with: %r\n" % (mean, stdev, param))


# Aus Effizienzgruenden nur kleines Trainingset (0.2) und nur 20 Durchlaeufe (epochs),
# aber jetzt haben wir eine grobe Ahnung, welche Parameter wir nehmen sollen
