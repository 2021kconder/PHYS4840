#!/usr/bin/python3.8
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import conder_functions_lib as cfl

#!/usr/bin/python3.8
#####################################
#
# Class 14: Matrices and Linear algebra 
# Author: M Joyce
#
#####################################
import numpy as np
from numpy import array,empty

A = np.array([ [2, -1, 3,],\
			   [-1, 4, 5], 
			   [3,  5, 6] ],float)

eigenvector_1 =  np.array([-0.5774,\
						   -0.5774,\
						   0.5774],float)

LHS = np.dot(A, eigenvector_1)

## Bonus: Why doesn't this line work??
#LHS = A*eigenvector_1

RHS = -2.0*eigenvector_1

print("LHS:\n",LHS, "\n\nRHS:\n",RHS)
print()


#!/usr/bin/python3.8
#####################################
#
# Class 14: Matrices and Linear algebra 
# Author: M Joyce
#
#####################################

'''
by importing and using the QR decomposition 
algorithm in my_functions_lib.py:
1) Find Q and R
2) Confirm that Q is orthogonal
3) Confirm that R is upper triangular
4) Confirm that the matrix A introduced in eigenvalues.py
can indeed be reconstructed by the dot product 
of matrices Q and R
'''
#finding Q and R

Q = (cfl.qr_decomposition(A))[0]
R = (cfl.qr_decomposition(A))[1]

print('Q', Q)
print()
print('R', R)
print()

#checking to see that Q is orthogonal

identity_matrix_check_one = np.dot(np.transpose(Q), Q)
identity_matrix_check_two = np.dot(Q, np.transpose(Q))

'''Since the output below is an identity matrix, we know that Qi s orthogonal 
(approximately since our answers below give values like 1e-17 where there should 
be zeros)'''

print('QQ$^T$ =', identity_matrix_check_one)
print()
print('Q$^T$T =',identity_matrix_check_two)
print()

#checking to see that R is an upper triangle 

N = 3 #dimension of the matrix 

print('If any other condition is returned besides "Upper Triange!" then we dont have an upper triangle.')

for i in range(N):
	if i == 0 and R[0].all() != 0:
		print('Upper Triangle!')

	else: 
		if R[i,i-1] == 0:
			print('Upper triangle!')

		else: 
			print('Oops... its not')


'''R is an upper triangle!'''



#checking that matrix A introduced in eigenvalues.py can indeed be reconstructed by the dot product of matrices Q and R

A_check = np.dot(Q,R)

print('QR Dot Product =', A_check)
print()
print(A)

'''QR does return A!'''

