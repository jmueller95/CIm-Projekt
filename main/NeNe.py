#!/usr/bin/env python

import argparse
import numpy as np
from AminoAcid import aminoAcidDict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

#Temporary import
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

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

print(annInput)
# Split data up into training and test sets
x_train, x_test, y_train, y_test = train_test_split(annInput, isBinderList, test_size=0.33, random_state=0)

# TODO: Is scaling necessary?



# TODO: What other models/layers are there?
model = Sequential()

# First try: Input layer with 9*5=45 input nodes - each property at every position
# TODO: What activation function should we use? For the moment it's 'relu'
model.add(Dense(45, activation='relu', input_shape=(45,)))

# Hidden layer with 5 nodes
model.add(Dense(5, activation='relu'))

# Output layer with 1 node
model.add(Dense(1, activation='sigmoid'))

# print(model.output_shape)
# print(model.summary())
# print(model.get_config())
# print(model.get_weights())

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=0)

y_pred = model.predict(x_test)
print(y_test[:10])
print(y_pred[:10])
#print(confusion_matrix(y_test,y_pred))
