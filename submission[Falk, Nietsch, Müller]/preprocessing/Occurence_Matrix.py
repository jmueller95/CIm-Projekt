#!/usr/bin/env python
from __future__ import division

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Occurence Matrix Generator")

    parser.add_argument('input', metavar='input file')

    args = parser.parse_args()

    # Parse the sequences into two String lists of binder and non-binder sequences
    binders = []
    nonBinders = []
    with open(args.input, "r") as i:
        for e in i:
            lineSplit = e.split()
            if lineSplit[0] == "Peptide":
                pass
            else:
                if int(lineSplit[2]) == 1:
                    binders.append(lineSplit[0])
                else:
                    nonBinders.append(lineSplit[0])
    # Count the occurences at each position
    # First, create two dictionaries with one-letter-codes as keys and integer lists of length 9 as values
    binderOccurenceDict = {}
    nonBinderOccurenceDict = {}
    oneLetterCodes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                      'Y']
    for letter in oneLetterCodes:
        binderOccurenceDict[letter] = [0] * 9
        nonBinderOccurenceDict[letter] = [0] * 9

    # Now iterate over the two sequence lists and count the occurrences of every AA at every position
    for seq in binders:
        index = 0
        for residue in seq:
            binderOccurenceDict[residue][index] += 1
            index += 1

    for seq in nonBinders:
        index = 0
        for residue in seq:
            nonBinderOccurenceDict[residue][index] += 1
            index += 1

    # Print the dictionaries as tables
    print("BINDER OCCURRENCE MATRIX")
    for key, value in sorted(binderOccurenceDict.items()):
        print(key + ": " + str([x / len(binders) for x in value]))

    print("-----------------------------")
    print("NON-BINDER OCCURRENCE MATRIX")
    for key, value in sorted(nonBinderOccurenceDict.items()):
        print(key + ": " + str([x / len(nonBinders) for x in value]))
