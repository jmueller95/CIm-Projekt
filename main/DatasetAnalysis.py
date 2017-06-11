import AminoAcid
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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
            if lineSplit[0] == "Peptide":  # skip the first line
                pass
            else:
                if int(lineSplit[2]) == 1:
                    binders.append(lineSplit[0])
                else:
                    nonBinders.append(lineSplit[0])

    # Create lists of summed up molecular weights of each sequence
    binderWeights = [sum([AminoAcid.aminoAcidDict.get(aa).weight for aa in sequence]) for sequence in binders]
    nonBinderWeights = [sum([AminoAcid.aminoAcidDict.get(aa).weight for aa in sequence]) for sequence in nonBinders]

    # Create lists of mean iep
    binderIeps = [sum([AminoAcid.aminoAcidDict.get(aa).iep for aa in sequence]) / len(sequence) for sequence in binders]
    nonBinderIeps = [sum([AminoAcid.aminoAcidDict.get(aa).iep for aa in sequence]) /
                     len(sequence) for sequence in nonBinders]

    # Create lists of mean hydrophobicity
    binderHydrophobicity = [sum([AminoAcid.aminoAcidDict.get(aa).hydrophobicity for aa in sequence]) /
                            len(sequence) for sequence in binders]
    nonBinderHydrophobicity = [sum([AminoAcid.aminoAcidDict.get(aa).hydrophobicity for aa in sequence]) /
                               len(sequence) for sequence in nonBinders]

    # Visualize
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(binderWeights, binderIeps, binderHydrophobicity, c="green")
    # ax.scatter(nonBinderWeights, nonBinderIeps, nonBinderHydrophobicity, c="red")
    # ax.set_xlabel("Molecular Weight")
    # ax.set_ylabel("Isoelectric Point")
    # ax.set_zlabel("Hydrophobicity")
    # plt.show()
    # Doesn't show anything useful, however...

    # Let's try it position-wise
    binderWeights_POSITIONS = []
    nonBinderWeights_POSITIONS = []
    binderIeps_POSITIONS = []
    nonBinderIeps_POSITIONS = []
    binderHydrophobicity_POSITIONS = []
    nonBinderHydrophobicity_POSITIONS = []
    for i in range(9):
        binderWeights_POSITIONS.append([AminoAcid.aminoAcidDict.get(sequence[i]).weight for sequence in binders])
        nonBinderWeights_POSITIONS.append([AminoAcid.aminoAcidDict.get(sequence[i]).weight for sequence in nonBinders])

        binderIeps_POSITIONS.append([AminoAcid.aminoAcidDict.get(sequence[i]).iep for sequence in binders])
        nonBinderIeps_POSITIONS.append([AminoAcid.aminoAcidDict.get(sequence[i]).iep for sequence in nonBinders])

        binderHydrophobicity_POSITIONS.append(
            [AminoAcid.aminoAcidDict.get(sequence[i]).hydrophobicity for sequence in binders])
        nonBinderHydrophobicity_POSITIONS.append(
            [AminoAcid.aminoAcidDict.get(sequence[i]).hydrophobicity for sequence in nonBinders])

    # ----Code copied from Occurence_Matrix.py----------
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
    # -------------------------------------------------#

    # Visualize one position at a time
    for i in range(9):
        position = i  # Change this to view another position
        # ax.set_xlabel("Molecular Weight")
        # ax.set_ylabel("Isoelectric Point")
        # ax.set_zlabel("Hydrophobicity")
        plt.clf()
        plt.title("Nonbinder sequences at position " + str(position + 1))
        plt.xlabel("Residue mass in u")
        plt.ylabel("Isoelectric Point")

        scaling = 500
        binderAreas = [scaling * np.pi * binderOccurenceDict[sequence[position]][position] / len(binders)
                       for sequence in binders]
        nonBinderAreas = [scaling * np.pi * nonBinderOccurenceDict[sequence[position]][position] / len(nonBinders)
                          for sequence in nonBinders]

        plt.scatter(nonBinderWeights_POSITIONS[position], nonBinderIeps_POSITIONS[position], s=nonBinderAreas,
                    c="red")
        #plt.scatter(binderWeights_POSITIONS[position], binderIeps_POSITIONS[position], s=binderAreas,
        #            c="green")
        #plt.ylim(-0.1, 1.1)
        #plt.yticks(np.arange(0, 1.1, 0.2))
        plt.savefig('../resources/scatterplots/WeightIEP_Nonbinder' + str(position + 1) + '.png')
