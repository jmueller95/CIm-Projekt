#!/usr/bin/env python

import argparse
import os
from random import random
import ModelCreator
from AminoAcid import aminoAcidDict
import numpy as np

if __name__ == "__main__":
    # Generate model
    model = ModelCreator.createModel(os.path.join(os.getcwd(), 'project_training.txt'))

    parser = argparse.ArgumentParser(description="Neural Network Epitope Binding Predictor")

    parser.add_argument('input', metavar='input file')
    parser.add_argument('output', metavar='output file')

    args = parser.parse_args()
    with open(args.input, "r") as i:
        sequencesList = [e.strip() for e in i]

    # Create input data from the sequences
    data = np.array([[(aminoAcidDict.get(residue).one_letter_code,
                       aminoAcidDict.get(residue).weight,
                       aminoAcidDict.get(residue).iep,
                       aminoAcidDict.get(residue).hydrophobicity,
                       aminoAcidDict.get(residue).polarity,
                       aminoAcidDict.get(residue).area)
                      for residue in sequence] for sequence in sequencesList])
    annInput = data[:, :, 1:6].reshape(len(sequencesList), 45)
    predictedValues = model.predict(annInput).reshape(len(annInput))
    threshold = 0.7  # TODO: Set best threshold from ROC analysis here!
    prediction = [1 if value > threshold else 0 for value in predictedValues]
    with open(args.output, "w") as o:
        for (seq, pred) in zip(sequencesList, prediction):
            o.write("%s\t%i\n" % (seq, pred))
