class AminoAcid(object):
    # Konstruktor
    def __init__(self, one_letter_code, flexibility, weight, iep, hydrophobicity, polarity, area):
        self.one_letter_code = one_letter_code

        self.flexibility = flexibility  # Flexibility of the side chain,
        # one of ["None", "Low", "Limited", "Moderate", "High", "Restricted"]

        self.weight = weight  # Molecular weight (in units, I guess...)
        self.iep = iep  # Isoelectric Point
        self.hydrophobicity = hydrophobicity  # TODO: What unit does hydrophobicity have?
        self.polarity = polarity # in Debye
        self.area = area # Accessible contact area in Angstrom^2

# TODO: Gibt es sinnvolle Methoden?
# End of class definition

# Instances for all possible 20 amino acids
# Data resource: https://www.ncbi.nlm.nih.gov/Class/Structure/aa/aa_explorer.cgi
ala = AminoAcid('A', "Limited", 71, 6.0, 0.806,0.0,10.0)
cys = AminoAcid('C', "Low", 103, 5.0, 0.721, 1.5,3.8)
asp = AminoAcid('D', "Moderate", 115, 3.0, 0.417, 49.7, 18.3)
glu = AminoAcid('E', "High", 129, 3.2, 0.458, 49.9, 15.8)
phe = AminoAcid('F', "Moderate", 147, 5.5, 0.951, 0.4, 8.1)
gly = AminoAcid('G', "None", 57, 6.0, 0.770,0.0, 7.6)
his = AminoAcid('H', "Moderate", 137, 7.6, 0.548,51.6, 12.3)
ile = AminoAcid('I', "Moderate", 113, 6.0, 1.000,0.2, 7.0)
lys = AminoAcid('K', "High", 128, 9.7, 0.263,49.5, 34.3)
leu = AminoAcid('L', "Moderate", 113, 6.0, 0.918, 0.1, 7.7)
met = AminoAcid('M', "High", 131, 5.7, 0.811, 1.4, 7.2)
asn = AminoAcid('N', "Moderate", 114, 5.4, 0.448, 3.4, 20.4)
pro = AminoAcid('P', "Restricted", 97, 6.3, 0.678, 1.6, 12.8)
gln = AminoAcid('Q', "High", 128, 5.7, 0.430, 3.5, 24.7)
arg = AminoAcid('R', "High", 156, 10.8, 0.000, 52.0, 32.6)
ser = AminoAcid('S', "Low", 87, 5.7, 0.601, 1.7, 12.9)
thr = AminoAcid('T', "Low", 101, 5.6, 0.634, 1.6, 15.1)
val = AminoAcid('V', "Low", 99, 6.0, 0.923, 0.1, 7.2)
trp = AminoAcid('W', "Moderate", 186, 5.9, 0.854, 2.1, 10.3)
tyr = AminoAcid('Y', "Moderate", 163, 5.7, 0.714, 1.6, 18.3)

# List all amino acids and their one-letter-codes, then zip them to a dictionary
# This dictionary, 'aminoAcidDict', will be accessed from the main program
aminoAcidList = [ala, cys, asp, glu, phe, gly, his, ile, lys, leu, met, asn, pro, gln, arg, ser, thr, val, trp, tyr]
oneLetterCodeList = [aa.one_letter_code for aa in aminoAcidList]
aminoAcidDict = {key: value for (key, value) in zip(oneLetterCodeList, aminoAcidList)}

