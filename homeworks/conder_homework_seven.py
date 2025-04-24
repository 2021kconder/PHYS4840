#!usr/local/Anaconda2023/bin/python3.11

import numpy as np 
import matplotlib.pyplot as plt
from math import *



plt.rcParams['font.family']= 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

################################################################
#
# PHYS 4840 HOMEWORK SEVEN 
# <Kaycee Conder> 
# Spring 2025 PHYS4840
#
################################################################


# ---------------------------------
# PROBLEM ZERO: FINISH LABS
# ---------------------------------

'''The completed form of these labs are in the py_files
folder in my GitHub repository.'''

# ---------------------------------
# PROBLEM ONE: TEXTBOOK 9.4
# ---------------------------------

#Following the format of our in class lab...

from pylab import plot, xlabel, ylabel, show, legend


# Defining mean daily temperature function 
def temp_0(t): 
	'''
	Function calculates the mean daily
	temperature at a particular point
	on the surface of the Earth. Equation
	Given in Textbook Problem 9.4

	Parameters: 
	A = float, temperature [C]
	B = float, temperature [C]
	tau = float, time [days]

	Returns: 
	T_0 = float, value calculated via equation 
	'''

	A = 10.0 #Celsius
	B = 12.0 #Celsius
	tau = 365.0 #days

	T_0 = A + ((B * sin(2*np.pi*t/tau)))

	return T_0

# Defining our constants 

L = 20.0 		#Depth [m]
D = 0.1 		#Thermal Diffusivity [m^2/day]
h = 0.01		#Time and Depth Step
N = 100			#Grid Divisions 
a = L/N 		#Grid Spacing
epsilon = h/1000 

#Setting our temperature and time conditions
T_low = 10.0
T_med = 11.0
T_hig = 12.0

t1 = 3285 	#zero months into the 10th year
t2 = 3376	#three months into the 10th year
t3 = 3467	#six months into the 10th year
t4 = 3558	#nine months into the 10th year
t5 = 3650	#twelve months into the 10th year
t_end = t5 + epsilon


#Setting empty arrays
T = np.empty(N+1, dtype=float)
T[0]=T_hig
T[N] = T_low
T[1:N] = T_med

T_p = np.empty(N+1, dtype=float)
T_p[0] = T_hig
T_p[N] = T_low

t = 0.0
c = h * D/(a*a)

#looping through...
while t < t_end: 
	for i in range(1,N): 
		
		T_p[i] = T[i] + c * (T[i+1] + T[i-1] - 2*T[i])
	

	T_p[0] = temp_0(t)
	T_p[-1]= 11

	T, T_p = T_p, T
	t+=h

	#Making plots at our time steps
	if abs(t-t1) < epsilon: 
		plot(np.linspace(0,L,N+1),T,color='red', label='Day 0 of 10th Year')
	if abs(t-t2)<epsilon:
		plot(np.linspace(0,L,N+1),T, color='orange', label='Day 91 of 10th Year')
	if abs(t-t3)<epsilon:
		plot(np.linspace(0,L,N+1),T, color='green', label = 'Day 182 of 10th Year')
	if abs(t-t4)<epsilon:
		plot(np.linspace(0,L,N+1),T, color='blue', label = 'Day 273 of 10th Year ')
	if abs(t-t5)<epsilon:
		plot(np.linspace(0,L,N+1),T, color='purple', label='Day 365 of 10th Year')

xlabel('Depth [m]')
ylabel('Temperature [C]')
legend()
show()


# ---------------------------------
# PROBLEM TWO: MESA TECHNIQUES
# ---------------------------------

'''
Referring to 'Modules for Experiments in Stellar Astrophysics (MESA): Convective Boundaries,
Element Diffusion, and Massive Star Explosions' 2018 from Paxton et al., we find that one 
of the methods utilized within the program that we also discussed in class is interpolation. 
Specifically, referring to Appendix A of the paper, we learn that in order to calculate 
bolometric corrections (BC), a linear interpolation is performed over log(T_eff/K), log(g/cm s^-2)
and [M/H]. The interpolation is done over these specific values given BC as a function of stellar
photosphere is defined by these parameters. 

'''

# ---------------------------------
# PROBLEM THREE: RK4 STELLAR MODELING
# ---------------------------------

'''
The Fortran file stellar_RK2.f90 solves the Mass Conservation Equation 
of stellar structure [Equation 1 in Problem 2]. This is evident from the 
k values within the RK4 integration loop, specially the lines: 

rho = density_profile(r, rho_c, R_star)
k1 = dr * (4.0d0 * pi * r**2 * rho)

The boundary conditions of ths problem are enforced in lines 13, 14, and 16 
of the problem, specifically via the lines:

r = 0.0d0
m = 0.0d0
r_end = R_star

These boundary conditions are enforced via n and the loop running from 1,n, 
as n directly depends on the boundary conditions of r and r_end in the line:

n = int((r_end - r) / dr)

'''
filepath_one = '/d/cha1/kconder/PHYS4840_labs/homeworks/profile.dat'
radius, mass = np.loadtxt(filepath_one, usecols=(0,1), skiprows=1, unpack=True)

fig, ax = plt.subplots()

ax.plot(radius, mass, color='green')
ax.set_xlabel('Radius')
ax.set_ylabel('Mass')
ax.set_title('Conservation of Mass Equation, Visualized')

plt.savefig('/d/cha1/kconder/PHYS4840_labs/homeworks/figures/mass_conservation.png')
plt.show()
plt.close()


'''
The x axis should be radius and the y axis should be mass, given that 
we have solved the mass conservation equation, where mass is a function of 
radius. On the plot above, the axes are labeled accordingly.

As an additional analysis note, we can see that the mass goes as radius
cubed (roughly). 
'''

# ---------------------------------
# PROBLEM FOUR: PDE CLASSES
# ---------------------------------

'''
Elliptical PDE
--------------
Boundary conditions instantly affect the entire
domain, and are time independent. The systems that are described
by elliptical PDEs are usually in a 'steady state' in the sense 
that they are not evolving any further over time and are thus
time independent.  

An example of an elliptical PDE is the Laplace Equation. As we 
demonstrated within lecture,this PDE can be solved utilizing 
different relaxation methods for boundary value problems such 
as The Method of Finite Distances, aka the Jacoby Method. 
More specifically, this method turns the PDE into a set of 
N grid points of linear, simultaneous eqns which can be 
solved through matrix techniques. 




Parabolic PDE
-------------
Boundary conditions effects gradually spread out 
over time. 

An example of a parabolic equation is the Diffusion Equation. 
As we demonstrated in lecture, this PDE can be solved utilizing 
not relaxation methods, but forward integration (since time is 
directional). In more specific terms, this process involves 
underlying Euler Methods and is ultimately refered to as the 
FTCS, or Fowrd Time Centered Space Method. More specifically, 
this method creates a segments of a line ID, and if we know
the value @ every point on this line at out initial time, 
we know the time at every value of dt later.



Hyperbolic PDE
--------------
Information propagates at a finite speed in waveforms. 
In other words, the changes in initial conditions do not INSTANTLY
affect everything everywhere all at once in the domain (insert movie
reference here). 

One very pertinent example of a hyperbolic PDE is that of Maxwell's Equations. 
I would assume that this PDE can be solved in a similar manner as the Diffusion
Equation, or parabolic PDEs, in that the FCTS method is impelemnted (given 
that our dependence is on time which is directional, and we are therefore
working with an IVP instead of BVP.)

'''