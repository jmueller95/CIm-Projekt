#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
from AminoAcid import aminoAcidDict
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, accuracy_score
from numpy.random import seed

# Make sure random numbers are always the same (for reproducibility)
randomFix = 100
seed(randomFix)
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
x_train, x_test, y_train, y_test = train_test_split(annInput, isBinderList, test_size=0.33, random_state=0)

# Create model
model = Sequential()  # Input layer with 9*5=45 input nodes - each property at every position
model.add(Dense(45, kernel_initializer='uniform', activation='softplus', input_shape=(45,)))

# Hidden layer with 17 nodes
model.add(Dense(17, kernel_initializer='uniform', activation='softplus'))

# Output layer with 1 node
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=100)

# Prediction
y_pred = model.predict(x_test).reshape(len(x_test))

# Find threshold with maximal MCC
thresholds = [i / 100.0 for i in range(100)]
MCCs = []
for t in thresholds:
    y_pred_binary = [1 if value > t else 0 for value in y_pred]
    MCCs.append(matthews_corrcoef(y_test, y_pred_binary))

threshold = np.argmax(MCCs) / 100.0
y_pred_binary = [1 if value > threshold else 0 for value in y_pred]

# Evaluation
confMatrix = confusion_matrix(y_test, y_pred_binary)
print("Confusion matrix with following shape:\n"
      "[[TN FP]\n"
      " [FN TP]]")
print(confusion_matrix(y_test, y_pred_binary))
print("Accuracy=" + str(accuracy_score(y_test, y_pred_binary)))
print("Sensitivity/Recall=" + str(recall_score(y_test, y_pred_binary)))
specificity = float(confMatrix[0][0]) / (
    confMatrix[0][0] + confMatrix[0][1])  # Couldn't find specificity in Scikit-learn
print("Specificity=" + str(specificity))
print("MCC=" + str(matthews_corrcoef(y_test, y_pred_binary)))

# Create CSV output for ROC Analysis with KNIME

roc_data = zip(y_pred, y_test)
df = pd.DataFrame(roc_data)
df.to_csv("ROC_Data.csv")
