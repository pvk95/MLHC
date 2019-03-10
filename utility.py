import numpy as np

mapping = {'A': [1, 0, 0, 0], 
           'T': [0, 1, 0, 0],
           'C': [0, 0, 1, 0],
           'G': [0, 0, 0, 1]
}


def map_dna_into_vector(string):
    '''
        This function maps a DNA string (ATCG) into a one-hot encoded vector
    '''
    vector = []
    for c in string:
        vector.append(mapping[c])
    vector = np.hstack(vector)
    return vector