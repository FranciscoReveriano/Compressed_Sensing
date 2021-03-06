import numpy as np
from MOSEK.mosek import *
from sklearn.feature_extraction import image
from scipy.signal import medfilt2d

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

def convert_T_to_A(mask, T_Matrix):
    # Now we Need to Convert the T Matrix into Smaller A Matrix
    A_matrix = (T_Matrix.T * mask).T                                                                                    # Multiply the matrix by mask to make sparse
    # We Shrink the Matrix
    A_Values = count_non_sparse_values(A_matrix.T[0])
    #print("A: Matrix\n", A_matrix)
    #print("T Non-Sparse Values:", A_Values)
    ## Now I need to Convert To Smaller Matrix
    A_matrix_small = np.zeros((A_Values, T_Matrix.shape[0]))                                                            # Create Empty Numpy Array To Get Proper Dimensions
    A = A_matrix[A_matrix != 0]                                                                                         # Get All the Non-Zero Values
    A = A.reshape(A_matrix_small.shape)                                                                                 # Reshape To Correct Dimensions
    return A                                                                                                            # Return A Matrix

def transform_Patch(dimension,mask, patch, T_Matrix,solver="L2",lambda1=0):
    # Turn Patch into C Matrix
    C = patch.flatten()
    # Get the New Sparse matrix
    new_matrix = mask * C                                                                                               # Taking Dot Product
    ## The Next Part is minimizing this matrix
    C_values = count_non_sparse_values(new_matrix)  # Count the Values in the C Matrix that are non-sparse

    ## Next part is creating the new sparse C matrix
    B_Matrix = convert_C_to_B(C_values, new_matrix)
    B_values = count_non_sparse_values(B_Matrix)
    #print("B Non-Sparse Values:", B_values)
    assert (B_values == B_Matrix.shape[0])

    # Convert the T Matrix to A
    A_Matrix = convert_T_to_A(mask, T_Matrix)
    #A_values = count_non_sparse_values(A_Matrix.T[0])
    #print("A Non-Sparse Values:", A_values)

    # Use Mosek
    if solver == "L1":
        alpha, res = l1norm(A_Matrix, B_Matrix)
    if solver == "L2":
        with mosek.Env() as env:
            with env.Task(0,0) as task:
                alpha, res = l2norm(A_Matrix,B_Matrix)
            task.__del__()
        env.__del__()
    if solver == "Lasso":
        with mosek.Env() as env:
            with env.Task(0,0) as task:
                try:
                    M = lseReg(A_Matrix,B_Matrix, lambda1,0)
                    M.solve()
                    alpha = M.getVariable("w").level()
                finally:
                    M.dispose()
            task.__del__()
        env.__del__


    new_C = np.matmul(T_Matrix,alpha)
    new_C = new_C.reshape((dimension))
    new_C = np.around(new_C)
    return new_C

def transform_Patch2(dimension,mask, patch, T_Matrix):
    # Turn Patch into C Matrix
    C = patch.flatten()
    # Get the New Sparse matrix
    new_matrix = mask * C                                                                                               # Taking Dot Product
    ## The Next Part is minimizing this matrix
    C_values = count_non_sparse_values(new_matrix)  # Count the Values in the C Matrix that are non-sparse

    ## Next part is creating the new sparse C matrix
    B_Matrix = convert_C_to_B(C_values, new_matrix)
    B_values = count_non_sparse_values(B_Matrix)
    #print("B Non-Sparse Values:", B_values)
    assert (B_values == B_Matrix.shape[0])

    # Convert the T Matrix to A
    A_Matrix = convert_T_to_A(mask, T_Matrix)
    A_values = count_non_sparse_values(A_Matrix.T[0])
    #print("A Non-Sparse Values:", A_values)
    #print(A_Matrix.shape)
    return A_Matrix, B_Matrix



def create_mask(num_sample, size):
    ''' Function Creates the Mask
        Receives a Number of Samples and Proceeds to Create The mask'''
    mask = np.zeros(size)                                                                                               # Initialize the Mask to Zero
    value = 0
    while value != num_sample:                                                                                          # Sample from the requested distribution
        index = np.random.randint(size,size=1)
        mask[index] = 1
        value = count_non_sparse_values(mask)
    values = count_non_sparse_values(mask)                                                                              # Count How many Non-Sparse Values there are
    assert(values == num_sample)                                                                                        # Error Check To Make sure Values Equal the Desired Amount
    return mask

def medianFilter(matrix,dimension, P,Q):
    patches = image.extract_patches_2d(matrix, dimension)  # Turn into Patches the main Matrix
    new_image = []
    for patch in patches:
        new_patch = medfilt2d(patch, kernel_size=3)
        new_image.append(new_patch)
    new_image = np.asarray(new_image)
    reconstructed_image = image.reconstruct_from_patches_2d(new_image, image_size=(P, Q))
    return reconstructed_image