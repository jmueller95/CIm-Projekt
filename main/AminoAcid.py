class AminoAcid(object):
    # Konstruktor
    def __init__(self, one_letter_code, flexibility, weight, iep, hydrophobicity):
        self.one_letter_code = one_letter_code

        self.flexibility = flexibility  # Flexibility of the side chain,
        # one of ["None", "Low", "Limited", "Moderate", "High", "Restricted"]

        self.weight = weight  # Molecular weight (in units, I guess...)
        self.iep = iep  # Isoelectric Point
        self.hydrophobicity = hydrophobicity  # TODO: What unit does hydrophobicity have?


# TODO: Gibt es sinnvolle Methoden?
# End of class definition

# Instances for all possible 20 amino acids
# Data resource: https://www.ncbi.nlm.nih.gov/Class/Structure/aa/aa_explorer.cgi
ala = AminoAcid('A', "Limited", 71, 6.0, 0.806)
cys = AminoAcid('C', "Low", 103, 5.0, 0.721)
asp = AminoAcid('D', "Moderate", 115, 3.0, 0.417)
glu = AminoAcid('E', "High", 129, 3.2, 0.458)
phe = AminoAcid('F', "Moderate", 147, 5.5, 0.951)
gly = AminoAcid('G', "None", 57, 6.0, 0.770)
his = AminoAcid('H', "Moderate", 137, 7.6, 0.548)
ile = AminoAcid('I', "Moderate", 113, 6.0, 1.000)
lys = AminoAcid('K', "High", 128, 9.7, 0.263)
leu = AminoAcid('L', "Moderate", 113, 6.0, 0.918)
met = AminoAcid('M', "High", 131, 5.7, 0.811)
asn = AminoAcid('N', "Moderate", 114, 5.4, 0.448)
pro = AminoAcid('P', "Restricted", 97, 6.3, 0.678)
gln = AminoAcid('Q', "High", 128, 5.7, 0.430)
arg = AminoAcid('R', "High", 156, 10.8, 0.000)
ser = AminoAcid('S', "Low", 87, 5.7, 0.601)
thr = AminoAcid('T', "Low", 101, 5.6, 0.634)
val = AminoAcid('V', "Low", 99, 6.0, 0.923)
trp = AminoAcid('W', "Moderate", 186, 5.9, 0.854)
tyr = AminoAcid('Y', "Moderate", 163, 5.7, 0.714)

# List all amino acids and their one-letter-codes, then zip them to a dictionary
# This dictionary, 'aminoAcidDict', will be accessed from the main program
aminoAcidList = [ala, cys, asp, glu, ]  # TODO: Complete
oneLetterCodeList = [acid.one_letter_code for acid in aminoAcidList]
aminoAcidDict = {key: value for (key, value) in zip(oneLetterCodeList, aminoAcidList)}
