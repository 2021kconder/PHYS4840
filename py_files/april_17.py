#!/usr/bin/python3.11
#####################################
#
# April 17th PHYS4840 Lab
# Kaycee Conder
#
#####################################

# ------------------------------
# Example 9.3 in Textbook
# ------------------------------

import numpy as np 
from pylab import plot, xlabel, ylabel, show

# Constants
L = 0.01 		#Thickness of the steel in meters
D = 4.25e-6 	#Thermal diffusivity 
N = 100			#Number of divisions in grid
a = L/N 		#Grid spacing
h = 1e-4 		#Time step
epsilon = h/100

t_low = 00.0 	#low temperature in C
t_med = 20.0 	#mid temperature in C
t_hig = 50.0 	#high temperature in C

t1 = 0.01
t2 = 0.1
t3 = 0.4
t4 = 1.0
t5 = 10.0
tend= t5 + epsilon

#Create arrays
T = np.empty(N+1, dtype=float)
T[0]=t_hig
T[N] = t_low
T[1:N] = t_med

Tp = np.empty(N+1, dtype=float)
Tp[0] = t_hig
Tp[N] = t_low

#Main loop
t = 0.0
c = h*D/(a*a)

while t<tend: 

	#Calculate the new values of T
	for i in range(1,N):
		Tp[i] = T[i] + c*(T[i+1] + T[i-1] - 2*T[i])
	T, Tp = Tp, Tp
	t+=h

	#Make plots at these given times
	if abs(t-t1)<epsilon:
		plot(T, color='red', label='t-t1<epsilon')
	if abs(t-t2)<epsilon:
		plot(T, color='orange', label='t-t2<epsilon')
	if abs(t-t3)<epsilon:
		plot(T, color='green', label='t-t3<epsilon')
	if abs(t-t4)<epsilon: 
		plot(T, color='blue', label='t-t4<epsilon')
	if abs(t-t5)<epsilon:
		plot(T, color='purple', label='t-t5<epsilon')

xlabel('X')
ylabel('T')
show()