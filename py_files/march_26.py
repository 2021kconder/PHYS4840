#!/usr/bin/python3.8
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, sin
import conder_functions_lib as cfl
import sys

#!/usr/bin/python3.8
#####################################
#
# March 26th, 2025
# Author: Kaycee Conder
# Course: PHYS4840
#
#####################################

###################
#
# Q1: Prelim Questions
#
###################

#Question 1

'''
Parameters of euler_method function:

f: our ACTUAL function f(x,t) that 
we want to implement this method with respect to.
x0: x(t) at t=t0. Initial condition.
t0: start of interval
t_end: end of interval
dt: Step size h

'''

#Question 2

'''
I assume that the step size h is 
determined by a similar manner as that of our 
numerical integration h step sizes, in that we 
find an h value that corresponds to the 
highest level of Python precision. We will say
that this is (b-a)/N, just as with Simpsons and 
Trapz, where N is the numer of steps.
'''

###################
#
# Q1: Solving 8.1
#
###################


def book_f(x,t):
	return -x**3 + sin(t)

#Conditions for our function

a = 0.0
b = 10.0
N = 1000
h = (b-a)/N
x = 0.0

book_euler_times = cfl.euler_method(book_f, x, a, b, h)[0]
book_euler_xvals = cfl.euler_method(book_f, x, a, b, h)[1]

fig, ax = plt.subplots(2)

ax[0].plot(book_euler_times, book_euler_xvals, color='green')
ax[0].set_xlabel('t')
ax[0].set_ylabel('x(t)')
ax[0].set_title('Example 8.1 Function')



###################
#
# Q2: In Class
# Exercise Function
#
###################

def class_f(x,t):
	return x**2 -x 

a_2 = 0.0
b_2 = 10.0
N_2 = 1000
h_2 = (b_2-a_2)/N_2 
x_2 = 0.5

class_euler_times = cfl.euler_method(class_f, x_2, a_2, b_2, h_2)[0]
class_euler_xvals = cfl.euler_method(class_f, x_2, a_2, b_2, h_2)[1]



ax[1].plot(class_euler_times, class_euler_xvals, color='green')
ax[1].set_xlabel('t')
ax[1].set_ylabel('x(t)')
ax[1].set_title('In Class Function')

plt.show()


###################
#
# Q2: In Class
# RK 2
#
###################

#MORE WORK WILL BE DONE ON THIS LAST PART ON CLASS THURDAY 


a_3 = 0.0
b_3 = 10.0
N_3_one = 10
N_3_two = 20
N_3_three = 50
N_3_four = 100

h_3_one = (b_3-a_3)/N_3_one 
h_3_two = (b_3 -a_3)/N_3_two
h_3_three = (b_3 - a_3)/N_3_three
h_3_four = (b_3 - a_3)/N_3_four

tpoints_3_one = np.arange(a_3, b_3, h_3_one)
tpoints_3_two = np.arange(a_3, b_3, h_3_two)
tpoints_3_three = np.arange(a_3, b_3, h_3_three)
tpoints_3_four = np.arange(a_3, b_3, h_3_four)

xpoints_3_one = []
xpoints_3_two = []
xpoints_3_three = []
xpoints_3_four = []

x_3_one = 0.0
x_3_two = 0.0
x_3_three = 0.0
x_3_four = 0.0

for t in tpoints_3_one:
	xpoints_3_one.append(x_3)
	k1_one = h_3_one * class_f(x_3,t)
	k2_one = h_3_one * class_f(x_3+ 0.5*k1_one,t+0.5*h_3_one)
	x_3_one += k2


fig, ax = plt.subplots()

ax.plot(tpoints_3_one, xpoints_3_one, color='red', label='N=10')
ax.set_xlabel('t')
ax.set_ylabel('x(t)')
ax.legend()

plt.show()