import numpy as np
from math import cos, sqrt, pi

''' This Formula Calculates the DCT Matrix for an Image Declared
by: Francisco Reveriano
Class: ECE 580
Duke University '''

def DCT_Value(x,y,u,v, P, Q):
    ''' Calculates the Individual Value'''
    # Calculate Alpha
    if u == 1:
        alpha = sqrt(1/P)
    else:
        alpha = sqrt(2/P)
    # Calculate Beta
    if v == 1:
        beta = sqrt(1/Q)
    else:
        beta = sqrt(2/Q)
    value = alpha * beta * cos((pi*(2*x -1)*(u-1))/(2*P)) * cos((pi*(2*y - 1)*(v-1))/(2*Q))
    return value

def DCT_Matrix(P,Q):
    T_dim = P * P, Q * Q
    DCT = np.zeros(T_dim)
    DCT_Values = []
    for x in range(1,P+1):
        for y in range(1,Q+1):
            for u in range(1,P+1):
                for v in range(1,Q+1):
                    value = DCT_Value(x,y,u,v, P, Q)
                    DCT_Values.append(value)
    # Convert Values into matrix
    k = 0
    for i in range(len(DCT)):
        for j in range(len(DCT[0])):
            DCT[i][j] = DCT_Values[k]
            k += 1
    return DCT
