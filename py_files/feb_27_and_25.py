#!usr/local/Anaconda2023/bin/python3.11

from math import sqrt
from math import sin, tanh
import numpy as np
import matplotlib.pyplot as plt
import conder_functions_lib as cfl
import time 


########################################

'''@author: Kaycee Conder
   Lab: February 27th, 2025'''

########################################

''' IMPORTED FROM THE interpolate_exercise.py FILE '''

#####################################
#
# Class 12: Numerical Differentiation II
# Author: M Joyce
#
#####################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline

# some data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([0, 2, 1, 3, 7, 8, 6, 9, 7, 10, 12])  

# Define fine-grained x-values for interpolation
x_domain = np.linspace(min(x), max(x), 100)

# Linear Interpolation
linear_interp = interp1d(x, y, kind='quadratic')
y_linear = linear_interp(x_domain)

# Cubic Spline Interpolation
cubic_spline = CubicSpline(x, y)
y_cubic = cubic_spline(x_domain)

# Plot the results
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='red', label='Data Points', zorder=3)
plt.plot(x_domain, y_linear, '--', label='Linear Interpolation', linewidth=2)
plt.plot(x_domain, y_cubic, label='Cubic Spline Interpolation', linewidth=2)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear vs. Cubic Spline Interpolation')
plt.grid(True)
plt.show()

##################
# Question One:

# Q: 

''' How does the smoothness of the two interpolation methods compare? '''

# A: 

''' The cubic spline interpolation is much smoother than that of the linear,
given the nature of the linear vs cubic methods. '''

##################
# Question Two:

# Q: 

''' What happens if you add more data points? Does one method improve more than the other? '''

# A: 

''' It appears that the cubic spline improves more than the linear when we increase the number
of data points in an oscillatory fashion.'''

##################
# Question Three:

# Q: 

''' Try changing y values to represent a sinusoidal function. Which method approximates the 
function better? '''

# A: 

''' It appears that the cubic function is a better approximation'''

##################
# Question Four:

# Q: 

''' Experiment with kind='quadratic' in interp1d. How does it compare to cubic splines?  '''

# A: 

''' It appears the quadratic spines either slightly overshoot or undershoot the interpolation of the points. 
When we are moving from a lower y value to a higher y value, we undershoot, and when we move from a higher
y value to a lower y value, we undershoot (COMPARED TO THE CUBIC SPLICE).


'''


#####################################
#
# Class 12: Numerical Differentiation II
# Author: M Joyce
#
#####################################

import numpy as np
import matplotlib.pyplot as plt
from math  import tanh, cosh


import conder_functions_lib as cfl

## compute the instantaneous derivatives
## using the central difference approximation
## over the interval -2 to 2

x_lower_bound = -2.0
x_upper_bound = 2.0

N_samples = 100

#####################
#
# Try different values of h
# What did we "prove" h should be
# for C = 10^(-16) in Python?
#
#######################
h_my = 0.000464158883361 ## what goes here?
h_two = 2
h_one = 1
h_small = 1e-14
h_max = 0.09  #when we put the value of 1e-16, as instructed by the problem, we go beyond the precision of C and 
                  # end up with weird results 

xdata = np.linspace(x_lower_bound, x_upper_bound, N_samples)

central_diff_values_my_h = []
central_diff_values_2_h = []
central_diff_values_1_h = []
central_diff_values_1eneg16_h =[]
central_diff_values_max = []

for x in xdata:
   central_difference = ( cfl.tan_function(x + 0.5*h_my) - cfl.tan_function(x - 0.5*h_my) ) / h_my
   central_diff_two = ( cfl.tan_function(x + 0.5*h_two) - cfl.tan_function(x - 0.5*h_two) ) / h_two
   central_diff_one = ( cfl.tan_function(x + 0.5*h_one) - cfl.tan_function(x - 0.5*h_one) ) / h_one
   central_diff_small = ( cfl.tan_function(x + 0.5*h_small) - cfl.tan_function(x - 0.5*h_small) ) / h_small
   central_diff_max = ( cfl.tan_function(x + 0.5*h_max) - cfl.tan_function(x - 0.5*h_max) ) / h_max
  
   central_diff_values_my_h.append(central_difference)
   central_diff_values_2_h.append(central_diff_two)
   central_diff_values_1_h.append(central_diff_one)
   central_diff_values_1eneg16_h.append(central_diff_small)
   central_diff_values_max.append(central_diff_max)

## Add the analytical curve
## let's use the same xdata array we already made for our x values

analytical_values = []

for x in xdata:
   dfdx = cfl.df_dx_analytical(x)
   analytical_values.append(dfdx)


plt.plot(xdata, analytical_values, linestyle='-', color='black')
plt.plot(xdata, central_diff_values_my_h, "*", color="green", markersize=8, alpha=0.5)
plt.show()
#plt.savefig('numerical_vs_analytic_derivatives.png')
plt.close()

##################
# Question Five:

# Q: 

''' Assuming the functions and derivatives are a good order of unity, what is a good guess for 
an appropriate value of h? '''

# A: 000464158883361




##################
# Question Six:

# Q: 

'''Once you have reproduced the figure on GitHub, try h=2, h=1, and h=1e-16. Plot all of these on the same graph as the result. '''

fig, ax = plt.subplots()

ax.plot(xdata, analytical_values, linestyle='-', color='black', label='Original Function')
ax.scatter(xdata, central_diff_values_my_h, color='red', marker='*', label='h=0.00464158883361')
ax.scatter(xdata, central_diff_values_2_h, color='orange', marker='*', label='h=2')
ax.scatter(xdata, central_diff_values_1_h, color='green', marker='*', label='h=1')
ax.scatter(xdata, central_diff_values_1eneg16_h, color='blue', marker='*', label='h=1e-14')
ax.scatter(xdata, central_diff_values_max, color='purple', marker='*', label='Max Accurateh, h=0.09')
ax.legend()

plt.show()



##################
# Question Six:

# Q: 

'''What is the largest value of h that still provides a good approximation to the analytical derivative? '''

# A:

''' About h = 0.09 '''