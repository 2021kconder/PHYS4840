#!usr/local/Anaconda2023/bin/python3.11

'''In class Problems/Work''' 

############################################################################

import numpy as np
#from scipy import sqrt

#PollEv Exercise 

vector = [10,11,12]
def normalize_vector(vector):
	answer = np.linalg.norm(vector)
	return answer, normalize_vector

print('Output of my function:',\
	normalize_vector(vector))

###########################################################################

#Loading different packages, different definitions 
#my_number = 16

#from_numpy = np.sqrt(my_number)
#from_scipy = sqrt(my_number)

#print('from numpy', from_numpy)
#print('from scipy', from_scipy)


'''Lab Exercises'''


############################################################################

#Exercise #1

##Diagram loaded separately, made in Google Drawings


###########################################################################

#Exercise #2

#Diagaram loaded separately, made in Google Drawings.


###########################################################################

#Exercise 3

def my_function(vector): 
	a = vector[0]
	b = vector[1]
	c = vector[2]

	return np.linalg.norm(vector)


vector = [1,2,3]
print('My function output', my_function(vector))


#3a answer = 3.7416573867739413
#3b answer: With 2 entries we get a IndexError, and with 4 entries we get
#an integer, 5.477225575051661\

##########################################################################

#Exercise 4

import conder_functions_lib as cfl

vector = [1,2,3]
print('My Own Function Library! Vector Check:', cfl.my_function(vector))

#Answer: This import works and gives me the same answer as when written 
#within this .py file. 

#Bonus Answer: It needs to be imported in the my_functions_lib file, as 
#the function is running through the library and what is imported there
#NOT using imports that I already have within this .py file. 

########################################################################

#Exercise 5

cookie_price = 10 #dollars

print('My Own Function Library! Cookie Check:', cfl.single_cookie(10))

#Output: 3 Chocolate chip bookies with leftover change of 0.4 cents. 

#######################################################################

#Advanced Exercise 6
#DIDN'T FINISH THIS ONE YET 

print('Square Root of 25 and -3:', cfl.square_root(25), cfl.square_root(-3))

#Output: 5, 'Error! Value is a Negative Number'

print('Floor Value of 25.9, 25, and 25j:', cfl.floor(25.9), cfl.floor(25), cfl.floor(25j))

#output: 25, Input is already an Integer!, Error! Input is complex!

#print('Rounding value of 3 and 3.191919191 to 2 decimal points:', cfl.round(3), cfl.round(3.191919191))




#######################################################################

#Advanced Exercise 7
#DIDN'T FINISH THIS ONE YET 

