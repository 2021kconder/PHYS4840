#!/usr/bin/python3.8
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, sin
import conder_functions_lib as cfl
import sys



#####################################
#
# April 1st, 2025
# Author: Kaycee Conder
# Course: PHYS4840
#
#####################################


#In Class Question Responses 

'''

$which gfortran yields:

 /bin//gfortran



$gfortran --version yields:

GNU Fortran (GCC) 11.4.1 20231218 (Red Hat 11.4.1-3)
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


Comments in fortran are !

Output of $./hello.exe i: 

Hello, Fortran!

'''

##################
#
# Q1: RK2
# Ex. 8.2
#
###################



def book_f(x,t):
	return -x**3 + sin(t)

a = 0.0
b = 10.0

N_ten = 10.0
N_twenty = 20.0
N_fifty = 50.0
N_hundred = 100.0

h_ten = (b-a)/N_ten
h_twenty = (b-a)/N_twenty
h_fifty = (b-a)/N_fifty
h_hundred = (b-a)/N_hundred



tpoints_ten = np.arange(a,b,h_ten)
tpoints_twenty = np.arange(a,b,h_twenty)
tpoints_fifty = np.arange(a,b,h_fifty)
tpoints_hundred = np.arange(a,b,h_hundred)

xpoints_ten = []
xpoints_twenty = []
xpoints_fifty = []
xpoints_hundred = []

x = 0.0

for t in tpoints_ten:
	xpoints_ten.append(x)
	k1 = h_ten * book_f(x,t)
	k2 = h_ten * book_f(x+0.5*k1, t+0.5*h_ten)
	x += k2

for t in tpoints_twenty:
	xpoints_twenty.append(x)
	k1 = h_twenty * book_f(x,t)
	k2 = h_twenty * book_f(x+0.5*k1, t+0.5*h_twenty)
	x += k2

for t in tpoints_fifty:
	xpoints_fifty.append(x)
	k1 = h_fifty * book_f(x,t)
	k2 = h_fifty * book_f(x+0.5*k1, t+0.5*h_fifty)
	x += k2

for t in tpoints_hundred:
	xpoints_hundred.append(x)
	k1 = h_hundred * book_f(x,t)
	k2 = h_hundred * book_f(x+0.5*k1, t+0.5*h_hundred)
	x += k2



import matplotlib.pyplot as pyplot

fig, ax = plt.subplots()

ax.plot(tpoints_ten, xpoints_ten, color='red', label='N=10')
ax.plot(tpoints_twenty, xpoints_twenty, color='orange', label='N=20')
ax.plot(tpoints_fifty, xpoints_fifty, color='green', label='N=50')
ax.plot(tpoints_hundred, xpoints_hundred, color='purple', label='N=100')
ax.set_xlabel('Time Points')
ax.set_ylabel('X Points')
ax.set_title('Python RK2')
ax.legend()
plt.show()
plt.close()
##################
#
# Q2: RK4
# Ex. 8.3
#
###################
def book_f(x,t):
	return -x**3 + sin(t)



a = 0.0
b = 10.0

N_ten = 10.0
N_twenty = 20.0
N_fifty = 50.0
N_hundred = 100.0

h_ten = (b-a)/N_ten
h_twenty = (b-a)/N_twenty
h_fifty = (b-a)/N_fifty
h_hundred = (b-a)/N_hundred

tpoints_ten = np.arange(a,b,h_ten)
tpoints_twenty = np.arange(a,b,h_twenty)
tpoints_fifty = np.arange(a,b,h_fifty)
tpoints_hundred = np.arange(a,b,h_hundred)

xpoints_ten_rk4 = []
xpoints_twenty_rk4 = []
xpoints_fifty_rk4 = []
xpoints_hundred_rk4 = []

x = 0.0

for t in tpoints_ten:
	xpoints_ten_rk4.append(x)
	k1 = h_ten * book_f(x,t)
	k2 = h_ten * book_f(x+0.5*k1, t+0.5*h_ten)
	k3 = h_ten * book_f(x+0.5*k2, t+0.5*h_ten)
	k4 = h_ten * book_f(x+k3, t+h_ten)
	x += (k1+2*k2+2*k3+k4)/6

for t in tpoints_twenty:
	xpoints_twenty_rk4.append(x)
	k1 = h_twenty * book_f(x,t)
	k2 = h_twenty * book_f(x+0.5*k1, t+0.5*h_twenty)
	k3 = h_twenty * book_f(x+0.5*k2, t+0.5*h_twenty)
	k4 = h_twenty * book_f(x+k3, t+h_twenty)
	x += (k1+2*k2+2*k3+k4)/6

for t in tpoints_fifty:
	xpoints_fifty_rk4.append(x)
	k1 = h_fifty * book_f(x,t)
	k2 = h_fifty * book_f(x+0.5*k1, t+0.5*h_fifty)
	k3 = h_fifty * book_f(x+0.5*k2, t+0.5*h_fifty)
	k4 = h_fifty * book_f(x+k3, t+h_fifty)
	x += (k1+2*k2+2*k3+k4)/6

for t in tpoints_hundred:
	xpoints_hundred_rk4.append(x)
	k1 = h_hundred * book_f(x,t)
	k2 = h_hundred * book_f(x+0.5*k1, t+0.5*h_hundred)
	k3 = h_hundred * book_f(x+0.5*k2, t+0.5*h_hundred)
	k4 = h_hundred * book_f(x+k3, t+h_hundred)
	x += (k1+2*k2+2*k3+k4)/6



fig, ax = plt.subplots()

ax.plot(tpoints_ten, xpoints_ten_rk4, color='red', label='N=10')
ax.plot(tpoints_twenty, xpoints_twenty_rk4, color='orange', label='N=20')
ax.plot(tpoints_fifty, xpoints_fifty_rk4, color='green', label='N=50')
ax.plot(tpoints_hundred, xpoints_hundred_rk4, color='purple', label='N=100')
ax.set_xlabel('Time Points')
ax.set_ylabel('X Points')
ax.set_title('Python RK4')
ax.legend()
plt.show()
plt.close()

##################
#
# Q3: Fortran
# RK4
#
###################

filepath_one = 'rk2_results.dat'
filepath_two = 'rk2_results_two.dat'

t,x = np.loadtxt(filepath_one, usecols=(0,1), skiprows=1,unpack=True)
t_10000, x_10000 = np.loadtxt(filepath_two, usecols=(0,1), skiprows=1,unpack=True)

import matplotlib.pyplot as pyplot

fig, ax = plt.subplots(2)

ax[0].plot(t,x, color='green', label='n as a function')
ax[0].legend()
ax[0].set_title('Fortran RK2, N Function')
ax[0].set_xlabel('t')
ax[0].set_ylabel('x')
ax[1].plot(t_10000,x_10000, color='blue', label='n=10,000')
ax[1].set_title('Fortran RK2, N=10000')
ax[1].set_xlabel('t')
ax[1].set_ylabel('x')
ax[1].legend()

plt.show()
plt.close()

'''From the above plots, we can see oscillatory behavior, with the key difference
between the top and bottom plot being the range over which we are viewing this 
oscillatory behavior.'''



##################
#
# Q4: Fortran
# RK4
#
###################

filepath_three = 'rk4_results.dat'

t_RK4,x_RK4 = np.loadtxt(filepath_three, usecols=(0,1), skiprows=1,unpack=True)

fig, ax = plt.subplots()

ax.plot(t_RK4, x_RK4, color='purple', label='RK4, n=10,000')
ax.plot(t_10000,x_10000, color='blue', label='RK2, n=10,000')
ax.set_title('Fortran: RK4 vs RK2')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.legend()

plt.show()
plt.close()





