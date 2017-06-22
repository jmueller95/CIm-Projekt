#!/usr/bin/env python

from AminoAcid import aminoAcidDict
import numpy as np
import argparse

# Parse the input file
parser = argparse.ArgumentParser(description="Occurence Matrix Generator")
parser.add_argument('input', metavar='input file')
args = parser.parse_args()
# Create a list of sequences and a boolean list where position i is True if sequence i is a binder
sequencesList = []
isBinderList = []
with open(args.input, "r") as i:
    for e in i:
        lineSplit = e.split()
        if lineSplit[0] == "Peptide":  # skip the first line
            pass
        else:
            sequencesList.append(lineSplit[0])
            isBinderList.append(lineSplit[2] is "1")

# Create a 2D-array of sequences, residue by residue
residues = np.array([[residue for residue in sequence] for sequence in sequencesList])
# residues' shape is 726*9

# Create 2D arrays for each of the properties (not sure if we're actually gonna use this, 'residue' might be enough)
# Weight
weights = np.array(
    [[aminoAcidDict.get(residue).weight for residue in sequence] for sequence in sequencesList])

# IEP
ieps = np.array(
    [[aminoAcidDict.get(residue).iep for residue in sequence] for sequence in sequencesList])

# Hydrophobicity
hydrophobicities = np.array(
    [[aminoAcidDict.get(residue).hydrophobicity for residue in sequence] for sequence in sequencesList])

# Polarity
polarities = np.array(
    [[aminoAcidDict.get(residue).polarity for residue in sequence] for sequence in sequencesList])

# Accessible Area
areas = np.array(
    [[aminoAcidDict.get(residue).area for residue in sequence] for sequence in sequencesList])
