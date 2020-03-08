import numpy as np


def count_non_sparse_values(matrix):
    '''Function Counts the Number of sparse values in a flatten 1 dimensional matrix'''
    values = 0  # Size of Non-Discrete Values
    for i in range(len(matrix)):  # This part is to calculate how many values are not null values
        if matrix[i] != 0:
            values += 1
    return values

def convert_C_to_B(values, matrix):
    ''' Function transforms the C matrix into the B Matrix
        The B matrix should be smaller than the C matrix.
        More importantly the B matrix represents the non-sparse C matrix after we mask the C Matrix'''
    B_Matrix = np.zeros((values))
    assert (len(B_Matrix) == values)  # Make sure the matrix size is correct to fit the values
    j = 0  # Index Number of B Matrix
    for i in range(len(matrix)):  # Go Through the C Matrix
        if matrix[i] != 0:  # if value is not discrete we put it into the new B
            B_Matrix[j] = matrix[i]
            j += 1
    return B_Matrix

def convert_T_to_A(mask, T_matrix):
