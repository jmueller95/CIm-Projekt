#!/usr/bin/env python

import argparse
import numpy as np
from AminoAcid import aminoAcidDict
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from numpy.random import seed  # Needed for seeding random numbers
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import os

# Make sure random numbers are always the same
seed(100)
# Parse the input file
parser = argparse.ArgumentParser()
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
x_train, x_test, y_train, y_test = train_test_split(annInput, isBinderList, test_size=0.8, random_state=0)


# Function for creating a model - needed for the keras classifier
def create_model(neurons=1, activation='relu'):
    model = Sequential()

    # Input layer with 9*5=45 input nodes - each property at every position
    model.add(Dense(45, kernel_initializer='uniform', activation=activation, input_shape=(45,)))

    # Hidden layer with 5 nodes
    model.add(Dense(neurons, kernel_initializer='uniform', activation=activation))

    # Output layer with 1 node
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam')
    model.fit(x_train, y_train, epochs=20, verbose=1)
    return model


# Create classifier
classifier = KerasClassifier(build_fn=create_model)

# Define grid search parameters
batch_size = [1, 10, 100, len(x_train)]
neurons = range(5, 20)
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

param_grid = dict(batch_size=batch_size, neurons=neurons, activation=activation)
grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1, scoring='roc_auc')
grid_result = grid.fit(x_train, y_train)

# Write results to file
with open(os.path.join(os.getcwd(), 'GridSearchResults1.txt'), 'w') as output:
    output.write("Best: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        output.write("%f (%f) with: %r\n" % (mean, stdev, param))


# For efficiency reasons only a small dataset was used (0.2 of the data available, which is 145 sequences)
# and only 20 training epochs were performed, but now we know how to refine our ANN and do a more detailed grid search
