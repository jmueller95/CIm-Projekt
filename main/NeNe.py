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
        if lineSplit[0] == "Peptide": #skip the first line
            pass
        else:
            sequencesList.append(lineSplit[0])
            isBinderList.append(lineSplit[1] is 1)

#Create numpy arrays of the sequence properties