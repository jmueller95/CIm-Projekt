import AminoAcid
import argparse
import matplotlib.pyplot as plt
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

    # Start position-wise analysis
    binderWeights_POSITIONS = []
    nonBinderWeights_POSITIONS = []
    binderIeps_POSITIONS = []
    nonBinderIeps_POSITIONS = []
    binderHydrophobicity_POSITIONS = []
    nonBinderHydrophobicity_POSITIONS = []
    binderPolarities_POSITIONS = []
    nonBinderPolarities_POSITIONS = []
    binderAreas_POSITIONS = []
    nonBinderAreas_POSITIONS = []
    for i in range(9):
        binderWeights_POSITIONS.append([AminoAcid.aminoAcidDict.get(sequence[i]).weight for sequence in binders])
        nonBinderWeights_POSITIONS.append([AminoAcid.aminoAcidDict.get(sequence[i]).weight for sequence in nonBinders])

        binderIeps_POSITIONS.append([AminoAcid.aminoAcidDict.get(sequence[i]).iep for sequence in binders])
        nonBinderIeps_POSITIONS.append([AminoAcid.aminoAcidDict.get(sequence[i]).iep for sequence in nonBinders])

        binderHydrophobicity_POSITIONS.append(
            [AminoAcid.aminoAcidDict.get(sequence[i]).hydrophobicity for sequence in binders])
        nonBinderHydrophobicity_POSITIONS.append(
            [AminoAcid.aminoAcidDict.get(sequence[i]).hydrophobicity for sequence in nonBinders])

        binderPolarities_POSITIONS.append(
            [AminoAcid.aminoAcidDict.get(sequence[i]).polarity for sequence in binders])
        nonBinderPolarities_POSITIONS.append(
            [AminoAcid.aminoAcidDict.get(sequence[i]).polarity for sequence in nonBinders])

        binderAreas_POSITIONS.append(
            [AminoAcid.aminoAcidDict.get(sequence[i]).area for sequence in binders])
        nonBinderAreas_POSITIONS.append(
            [AminoAcid.aminoAcidDict.get(sequence[i]).area for sequence in nonBinders])

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

    # Visualize one position at a time
    for i in range(9):
        position = i  # Change this to view another position
        plt.clf()
        plt.title("Binder sequences at position " + str(position + 1))
        plt.xlabel("Accessible contact area in Angstrom^2")
        plt.ylabel("Polarity in Debye")

        scaling = 500
        binderRadii = [scaling * np.pi * binderOccurenceDict[sequence[position]][position] / len(binders)
                       for sequence in binders]
        nonBinderRadii = [scaling * np.pi * nonBinderOccurenceDict[sequence[position]][position] / len(nonBinders)
                          for sequence in nonBinders]

        # plt.scatter(nonBinderAreas_POSITIONS[position], nonBinderPolarities_POSITIONS[position], s=nonBinderRadii,
        #           c="red")
        plt.scatter(binderAreas_POSITIONS[position], binderPolarities_POSITIONS[position], s=binderRadii,
                    c="green")
        # plt.ylim(-0.1, 1.1)
        # plt.yticks(np.arange(0, 1.1, 0.2))
        plt.savefig('../resources/scatterplots/AreaPolarity_Binder' + str(position + 1) + '.png')
