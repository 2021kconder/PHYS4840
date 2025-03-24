#!usr/local/Anaconda2023/bin/python3.11

import numpy as np
import matplotlib.pyplot as plt
import math 
from math import *

#making sure that the font of my plots is Times
plt.rcParams['font.family']= 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

######################################################################################

#HOMEWORK FOUR <Kaycee Conder> Spring 2025 PHYS4840

'''IMPORTANT NOTE: 
This homework is NOT entirely complete, and will be used as my 'drop' homework.'''

######################################################################################
'''PROBLEM ONE: QUADRATIC VS LINEAR ERROR'''

'''The answers to this problem are derived in the written portion of the homework 
attached separately.'''


######################################################################################
'''PROBLEM TWO: BOOK 5.17, GAMMA FUNCTION'''

#Part A: Plotting gamma integrand 

integrand_a_two   = []
integrand_a_three = []
integrand_a_four  = []

x_range=np.linspace(0,6,1000)

for i in x_range:
	integrand_a_two.append( (i**(2-1) * ((math.e)**(-i))))
	integrand_a_three.append( (i**(3-1) * ((math.e)**(-i))))
	integrand_a_four.append( (i**(4-1) * ((math.e)**(-i))))



fig, ax = plt.subplots()

ax.plot(x_range, integrand_a_two, color='red', label='a=2')
ax.plot(x_range, integrand_a_three, color='orange', label='a=3')
ax.plot(x_range, integrand_a_four, color='green', label='a=4')
ax.set_xlabel('X')
ax.set_ylabel('X$^{a-1}$ e$^{-x}$')
ax.set_title('Gamma Function Interand Visualized ')
ax.legend()


plt.show()
plt.close()


#Part B: Analytical Maximum 

'''Work done on sheet. Numerical proof shown here as an additional
confirmation of the claim of B.'''


#Part C: Change of Variables


######################################################################################
'''PROBLEM THREE: EQUILIBRIUM OF SUSPENDED BEAM'''

#can actually find the answers for the tensions lol 


base_mass  = 10. #kg
extra_mass = 20. #kg

length       = 4. #meters
cable_one_x  = 2. #meters from left end
cable_two_x  = 4. #meters from left end 
extra_mass_x = 1. #meters from left end

beam_weight  = 098.1 #Newtons, acts at 2m
extra_weight = 196.2 #Newtons, acts at 1m

A = np.array([ [1, 0, 0, 0],\
			   [0, 1, 1, -1], 
			   [0, 2, 4, 0], 
			   [ 0, 2, -1, 2] ],float)

vector = np.array([0, 294.3, 392.4, 196.2])


########################################
#
print('Problem Three, Part A Answer')
print()
#
########################################

three_part_a = 'This matrix is unique in that we do not have a symmetrical, \
or Hermitian, matrix. This means that our eigenvalues may be \
complex, and our eigenvectors arent necessarily orthogonal \
i.e. their dot product ISNT zero. Despite this, we can still \
use QR matrix decomposition, as this method applies to ANY\
square matrix, symmetric or not.'

print(three_part_a)
print()


########################################
#
print('Problem Three, Part B Practice QR Answers:')
print()
#
########################################

'''First QR Decomposition'''

col_vec_one   = np.array([ [1], [0], [0], [0]], float)
col_vec_two   = np.array([ [0], [1], [2], [2]], float)
col_vec_three = np.array([ [0], [1], [4], [-1]], float)
col_vec_four  = np.array([ [0], [-1], [0], [2]], float)


'''The below is a QR decomposition PRACTICE RUN using the example matrix
in Dr. Joyce's PHYS4840_class14_notes.pdf, following along
to ensure that my end results are correct. This is a means 
of checking the mathematical validity of my code prior to 
running it on a separate matrix, as the problem requires'''

A_example = np.array([ [2, -1, 3],\
			   [-1, 4, 5], 
			   [3, 5, 6]],float)


col_practice_one   = np.array([ [2], [-1], [3]], float)
col_practice_two   = np.array([ [-1], [4], [5]], float)
col_practice_three = np.array([ [3], [5], [6]], float)

#Finding q1

#defining a normalization function 
def normalization(vector):
	'''
	Function meant to normalize the 
	entirety of a given vector.
	
	Parameters: 
		vector = array, either column or row vector 
		that we want to normalize

	Returns: 
		normalized vec, array, the original vector 
		divided by the vector's magnitude 
	'''
	summ = []
	for i in vector:
		summ.append(i**2)

	magnitude = abs(sqrt(sum(summ)))

	normalized_vec = vector/magnitude

	return normalized_vec 

q_one = normalization(col_practice_one)


#Finding q2

#defining a function to project our given q values onto a new vector, 
#, i.e. the proceeding column of our matrix 

def projection(v, q):
	projected_vector = q*((np.dot(q.T, v))/((np.dot(q.T, q))))

	return projected_vector

proj_q1_v2= projection(col_practice_two, q_one)
u2 = col_practice_two - proj_q1_v2
q_two = normalization(u2)



#Finding Q3

proj_q1_v3 = projection(col_practice_three, q_one)
proj_q2_v3 = projection(col_practice_three, q_two)

u3 = col_practice_three - proj_q1_v3 - proj_q2_v3
q_three = normalization(u3)

print('Q Three', q_three)
Q_practice_final = np.array( [ [q_one[0,0], q_two[0,0], q_three[0,0]], \
					  [q_one[1,0], q_two[1,0], q_three[1,0]], 
					  [q_one[2,0], q_two[2,0], q_three[2,0]]], float)

print('Q Example Calculation', Q_practice_final)
print()

#Find R from Q

'''MATH NOTE: 
The R vector can be constructed via
the dot product between the transpose
of our original matrix columns and the
column vectors of our Q matrix. The 
original matrix vectors vary from column
to column, whereas the Q vectors vary from 
row to row, to create an upper diagonal.'''

Q_practice_col_one = np.array([ [q_one[0,0]], [q_one[1,0]], [q_one[2,0]]], float)
Q_practice_col_two = np.array([ [q_two[0,0]], [q_two[1,0]], [q_two[2,0]]], float)
Q_practice_col_three = np.array([ [q_three[0,0]], [q_three[1,0]], [q_three[2,0]]], float)

R_practice_final = np.array( [ [float(np.dot(col_practice_one.T,Q_practice_col_one)), float(np.dot(col_practice_two.T,Q_practice_col_one)), float(np.dot(col_practice_three.T, Q_practice_col_one)) ], \
					  [0., float(np.dot(col_practice_two.T, Q_practice_col_two)), float(np.dot(col_practice_three.T, Q_practice_col_two))],
					  [0., 0., float(np.dot(col_practice_three.T, Q_practice_col_three))]], float )


print('R Example Calculation ', R_practice_final)
print()
print('Example Matrix np.linalg.qr Solution', np.linalg.qr(A_example))
print()

'''These numbers are the same as Dr. Joyce's example (with some changes that 
are correct via the np.linalg calculator). Now understanding the process, let
us solve the ACTUAL PROBLEM BELOW:'''


########################################
#
print('Problem Three, Part B ACTUAL QR Answers:')
print()
#
########################################


#Finding q1

q_one_final = normalization(col_vec_one)


#Finding q2

proj_q1_v2_final= projection(col_vec_two, q_one_final)
u2_final = col_vec_two - proj_q1_v2_final
q_two_final = normalization(u2_final )

#Finding Q3

proj_q1_v3_final = projection(col_vec_three, q_one_final)
proj_q2_v3_final = projection(col_vec_three, q_two_final)

u3_final = col_vec_three - proj_q1_v3_final - proj_q2_v3_final
q_three_final = normalization(u3_final)

#Finding Q4

proj_q1_v4_final = projection(col_vec_four, q_one_final)
proj_q2_v4_final = projection(col_vec_four, q_two_final)
proj_q3_v4_final = projection(col_vec_four, q_three_final)

u4_final = col_vec_four - proj_q1_v4_final - proj_q2_v4_final - proj_q3_v4_final
q_four_final = normalization(u4_final)



#Finding Q

Q_final = np.array( [ [q_one_final[0,0], q_two_final[0,0], q_three_final[0,0], q_four_final[0,0]], \
					  [q_one_final[1,0], q_two_final[1,0], q_three_final[1,0], q_four_final[1,0]], 
					  [q_one_final[2,0], q_two_final[2,0], q_three_final[2,0], q_four_final[2,0]],
					  [q_one_final[3,0], q_two_final[3,0], q_three_final[3,0], q_four_final[3,0]]], float)


print('Q Final', Q_final)
print()

#Finding R

Q_col_one = np.array([ [q_one_final[0,0]], [q_one_final[1,0]], [q_one_final[2,0]], [q_one_final[3,0]]], float)
Q_col_two = np.array([ [q_two_final[0,0]], [q_two_final[1,0]], [q_two_final[2,0]], [q_two_final[3,0]]], float)
Q_col_three = np.array([ [q_three_final[0,0]], [q_three_final[1,0]], [q_three_final[2,0]], [q_three_final[3,0]]], float)
Q_col_four = np.array([ [q_four_final[0,0]], [q_four_final[1,0]], [q_four_final[2,0]], [q_four_final[3,0]]], float)


R_final = np.array( [ [float(np.dot(col_vec_one.T,Q_col_one)), float(np.dot(col_vec_two.T,Q_col_one)), float(np.dot(col_vec_three.T, Q_col_one)), float(np.dot(col_vec_four.T, Q_col_one)) ], \
					  [0, float(np.dot(col_vec_two.T, Q_col_two)), float(np.dot(col_vec_three.T, Q_col_two)), float(np.dot(col_vec_four.T, Q_col_two))],
					  [0, 0, float(np.dot(col_vec_three.T, Q_col_three)), float(np.dot(col_vec_four.T, Q_col_three))],
					  [0, 0, 0, float(np.dot(col_vec_four.T, Q_col_four))]], float )


print('R Final', R_final)
print()

print('Final np.linalg.qr Calculation', np.linalg.qr(A) )
print()

########################################
#
print('Problem Three, Part B LU Answers')
print()
#
########################################



def identity(dimension):
	base = np.empty(shape=(dimension,dimension))

	for i in range(dimension):
		for j in range(dimension):

			if i == j: 

				base[i,j] = 1

			else: 
				base[i,j] = 0

	return base 


#Initiallty setting L as the identity matrix 
L = identity(len(A))


# initalize U as a copy of A
U = A.copy()


## this double loop will transform L
## into the lower-diagonal form we need
for m in range(len(A)):
    for i in range(m+1, len(A)):        
        
        # Compute the multiplier for the current row operation
        L[i, m] = U[i, m] / U[m, m]
        
        # Subtract the appropriate multiple of the pivot row from the current row
        U[i, :] -= L[i, m] * U[m, :]

print('L Final', L)
print('U Final', U)

########################################
#
print('Problem Three, Part C Answers')
print()
#
########################################

print('Proving Q Orthogonality:')
print((np.dot(Q_final, Q_final.T)))
print((np.dot(Q_final.T, Q_final)))
print()

print('Proving Q Normality:')
print(((np.linalg.inv(Q_final))))
print(Q_final.T)


three_part_c = 'We can see that, assuming our e-16 and below values to be approximately zero \
that the transpose of our Q matrix multiplied by the Q matrix, and vice versa, is \
equal to the identity matrix, thus proving its orthonormality.'

print()
print(three_part_c)
print()

######################################################################################
'''PROBLEM FOUR: INTERPOLATION VS NUMERICAL DIFFERENCE'''

'''Answer presented in written version of homework.'''


######################################################################################
'''PROBLEM FIVE: EIGENVALUES'''

#could add a determinant definition 

#Part A

matrix_A = np.array( [ [4,1], \
					   [2,3]], float)

for i in range(-100,100):
	if np.linalg.det(matrix_A - (identity(2)*i)) == 0:
		print('Matrix A Eigenvalue =', i)

print('Matrix A linalg.eig Check', np.linalg.eig(matrix_A)[0])
print()





#Part B

matrix_B = np.array( [ [1,2,3], \
	                   [0,1,4],
	                   [0,0,1]], float)


for i in range(-100,100):
	if np.linalg.det(matrix_B - (identity(3)*i)) == 0:
		print('Matrix B Eigenvalue =', i)

print('Matrix B linalg.eig Check', np.linalg.eig(matrix_B)[0])
print()

#Part C

matrix_C = np.array( [ [1,2,3], \
	                   [4,5,6],
	                   [7,8,9]], float)


for i in range(-100,100):
	if np.linalg.det(matrix_C - (identity(3)*i)) == 0:
		print('Matrix C Eigenvalue =', i)

print('Matrix C linalg.eig Check', np.linalg.eig(matrix_C)[0])
print()



#Part D
matrix_D = np.array( [ [1,2,3], \
	                   [4,5,6],
	                   [7,8,10]], float)


for i in range(-100,100):
	if np.linalg.det(matrix_D - (identity(3)*i)) == 0:
		print('Matrix D Eigenvalue =', i)
	else:
		print('We cannot find the Matrix D Eigenvalues via this Method!')
		break

print('Matrix D eigenvalues', np.linalg.eig(matrix_D)[0])
print()

five_part_d = 'We cannot find the eigenvalue for Matrix D via our \
method of iterating through possible values because the eigenvalues are \
not integers. Therefore, via the numerical method implemented here, as opposed \
to solving via a method such as sympy with specific packages, we cannot find the \
eigenvalues for this matrix. '

print(five_part_d)


	