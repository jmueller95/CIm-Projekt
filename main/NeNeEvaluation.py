#!/usr/bin/env python

import argparse
import numpy as np
import os
from AminoAcid import aminoAcidDict
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense
from numpy.random import seed

# Make sure random numbers are always the same (for reproducibility)
randomFix = 100
seed(randomFix)
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
# We want to evaluate our model with a stratified 10-fold stratified cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=randomFix)
stratifiedCVScores = []
# Define a string to store the results in (string is written to an output file afterwards
outputString = ""
iterations = 100
for i in range(iterations):
    outputString += "\nIteration No." + str(i+1) + "\n"
    currentCVScores = []
    for train, test in kfold.split(annInput, isBinderList):
        # Create model
        model = Sequential()  # Input layer with 9*5=45 input nodes - each property at every position
        model.add(Dense(45, kernel_initializer='uniform', activation='softplus', input_shape=(45,)))

        # Hidden layer with 17 nodes
        model.add(Dense(17, kernel_initializer='uniform', activation='softplus'))

        # Output layer with 1 node
        model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Split up data into training and test set
        x_train = [list(annInput[i]) for i in train]
        y_train = np.array([isBinderList[i] for i in train])
        x_test = [list(annInput[i]) for i in test]
        y_test = np.array([isBinderList[i] for i in test])

        model.fit(x_train, y_train, epochs=100, verbose=0, batch_size=100)
        # evaluate
        scores = model.evaluate(x_test, y_test, verbose=0)
        # Generate next part of output (percentage accuracy of the current model)
        nextOutput = "%.2f%%" % (scores[1] * 100)
        # Append it to the output string and directly print it
        outputString += nextOutput + "\n"
        print(nextOutput)
        # Save scores for later
        currentCVScores.append(scores[1] * 100)
    # Calculate mean and standard deviation and put it in a string
    meanAndStdOutput = "Mean: %.2f%% (+/- %.2f%%)" % (np.mean(currentCVScores), np.std(currentCVScores))
    outputString += meanAndStdOutput + "\n"
    print(meanAndStdOutput)
    stratifiedCVScores += currentCVScores
overallMeanAndStdOutput = "Overall Mean: %.2f%% (+/- %.2f%%)" % (np.mean(stratifiedCVScores), np.std(stratifiedCVScores))
outputString += overallMeanAndStdOutput + "\n"
# Write outputString to file
with open(os.path.join(os.getcwd(), 'CrossValidationResults.txt'), 'w') as output:
    output.write(outputString)
