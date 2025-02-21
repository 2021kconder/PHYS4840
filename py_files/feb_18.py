#!usr/local/Anaconda2023/bin/python3.11

import sys 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import norm 
import statistics 
from sklearn.mixture import GaussianMixture


########################################

'''@author: Kaycee Conder
   Lab: February 18th, 2024'''

########################################

def trapezoidal_rule(f, a, b, N):
    """
    Approximates the integral using the trapezoidal rule with a loop.

    Parameters:
        f (function or array-like): A function, it's evaluated at N+1 points.
                                    
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
        N (int): Number of intervals (trapezoids).

    Returns:
        float: The approximated integral.
    """
    
    h = (b-a)/N  # Step size

    integral = (1/2) * (f(a) + f(b)) * h  # Matches the first & last term in the sum

    # Loop through k=1 to N-1 to sum the middle terms
    for k in range(1, N):
        xk = a + k * h  # Compute x_k explicitly (matches the formula)
        integral += f(xk) * h  # Normal weight (multiplied by h directly)

    return integral


def function(x):
    return np.exp(-x**2)

a = 0  # Integration bounds
b = 1  # Integration bounds
N= 2  # Number of trapezoids

integral_approx= trapezoidal_rule(function, a, b, N)
print(f"Approximated Integral with N={N}: {integral_approx}")

N= 10  # Number of trapezoids

integral_approx= trapezoidal_rule(function, a, b, N)
print(f"Approximated Integral with N={N}: {integral_approx}")

N= 100  # Number of trapezoids

integral_approx= trapezoidal_rule(function, a, b, N)
print(f"Approximated Integral with N={N}: {integral_approx}")

#GO THROUGH AND CALCULATE THE ERROR ABOVE 

###############################################
#
# Simpsons Rule via Dr. Miller's code
print()
# 
###############################################

def simpsons_rule(f, a, b, N):
    """
    Approximates the integral using Simpson's rule.

    Parameters:
        f (function): The function to integrate.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
        N (int): Number of intervals (must be even).

    Returns:
        float: The approximated integral.
    """

    h = (b-a)/N # Step size
    integral = f(a) + f(b)  # First and last terms
    
    # Loop through k=1 to N-1
    for k in range(1, N, 2):  # Odd indices (weight 4)
        xk = a + k * h
        integral += 4 * f(xk)

    for k in range(2, N-1, 2):  # Even indices (weight 2)
        xk = a + k * h
        integral += 2 * f(xk)

    return (h / 3) * integral  # Final scaling


def function(x):
    return np.exp(-x**2)

a = 0  # Integration bounds
b = 1  # Integration bounds
N = 2  # Number of sections (what happens if this is odd, why?)

integral_approx = simpsons_rule(function, a, b, N)
print(f"Simpson Approximated Integral with N={N}: {integral_approx}")

N = 10  # Number of sections (what happens if this is odd, why?)

integral_approx = simpsons_rule(function, a, b, N)
print(f"Simpson Approximated Integral with N={N}: {integral_approx}")

N = 200 # Number of sections (what happens if this is odd, why?)

integral_approx = simpsons_rule(function, a, b, N)
print(f"Simpson Approximated Integral with N={N}: {integral_approx}")




#Hint: the integral of e^(-x**2) between 0 and 1 is 0.746824132812427


###############################################
#
# Combining All Methods via Dr. Miller's Code
print()
# 
###############################################

import numpy as np
import time
import matplotlib.pyplot as plt



####################
#
# Trapezoid 
#
####################

# Example usage with array data
def trapezoidal(y_values, x_values, N):
    """
    Approximates the integral using trapezoidal rule for given y_values at given x_values.
    
    Parameters:
        y_values (array-like): The function values at given x points.
        x_values (array-like): The x values corresponding to y_values.
        N (int): Number of intervals.

    Returns:
        float: The approximated integral.
    """

    a = x_values[0]
    b = x_values[-1]
    h = (b-a)/N



    integral = (1/2) * (y_values[0] + y_values[-1]) * h  # First and last terms

    for k in range(1, N):
        xk = a + k * h  # Compute x_k explicitly
        yk = np.interp(xk, x_values, y_values)  # Interpolate y at x_k manually in loop
        integral += yk * h

    return integral

####################
#
# Simpsons
#
####################

# Simpson's rule for array data
def simpsons(y_values, x_values, N):
    """
    Approximates the integral using Simpson's rule for given y_values at given x_values.

    Parameters:
        y_values (array-like): The function values at given x points.
        x_values (array-like): The x values corresponding to y_values.
        N (int): Number of intervals (must be even).

    Returns:
        float: The approximated integral.
    """

    a = x_values[0]
    b = x_values[-1]
    h = (b-a)/N

    integral = (1/2) * (y_values[0] + y_values[-1]) * h # First and last y_value terms 
    #not sure about the above? 

    for k in range(1, N, 2):  # Odd indices (weight 4)
        xk = a + k * h
        yk = np.interp(xk, x_values, y_values)
        integral += 4 * yk

    for k in range(2, N, 2):  # Even indices (weight 2)
        xk = a + k * h
        yk = np.interp(xk, x_values, y_values)
        integral += 2 * yk

    return (h / 3) * integral  # Final scaling


####################
#
# Rombergs
#
####################

# Romberg integration for array data
def romberg(y_values, x_values, max_order):
    """
    Approximates the integral using Romberg's method for given y_values at given x_values.

    Parameters:
        y_values (array-like): The function values at given x points.
        x_values (array-like): The x values corresponding to y_values.
        max_order (int): Maximum order (controls accuracy).

    Returns:
        float: The approximated integral.
    """
    R = np.zeros((max_order, max_order))
    a = x_values[0]
    b = x_values[-1]
    N = 1
    h = (b - a)

    # First trapezoidal estimate
    R[0, 0] = (h / 2) * (y_values[0] + y_values[-1])

    for i in range(1, max_order):
        N = 2**i     #Remember: we are recomputing the integral with different N (and therefore h)
        a = x_values[0]
        b = x_values[-1]

        h = (b-a)/(2**i) #Look at the github derivation for richardson extrapolation


        sum_new_points = sum(np.interp(a + k * h, x_values, y_values) for k in range(1, N, 2))
        R[i, 0] = 0.5 * R[i - 1, 0] + h * sum_new_points

        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4**j - 1)

    return R[max_order - 1, max_order - 1]


    ####################
    # 
    # Timing Function
    #
    ####################

def timing_function(integration_method, x_values, y_values, integral_arg):
    """
    Times the execution of an integration method.

    Parameters:
        integration_method (function): The numerical integration function.
        x_values (array-like): The x values.
        y_values (array-like): The corresponding y values.
        integral_arg (int, optional): EITHER Number of intervals to use (Simpson/Trapz) OR the maximum order of extrapolation (Romberg).

    Returns:
        tuple: (execution_time, integration_result)
    """
    start_time = time.perf_counter()
    result = integration_method(y_values, x_values, integral_arg)
    end_time = time.perf_counter()
    
    return end_time - start_time, result

####################
#
# Function to Integrate
#
####################

# Function to integrate
def function(x):
    return x * np.exp(-x)

end_numpy = time.perf_counter()

# Precompute data for fair comparisons
x_data = np.linspace(0, 1, 100000000)  # High-resolution x values
y_data = function(x_data)

# Testing parameters
N = range(1,100,1) # Number of intervals
max_order = 3 # Romberg's accuracy level






'''trap_time, trap_result = timing_function(trapezoidal, x_data, y_data, N)
simp_time, simp_result = timing_function(simpsons, x_data, y_data, N)
romb_time, romb_result = timing_function(romberg, x_data, y_data, max_order)'''

# True integral value
true_value = 0.26424111765711535680895245967707826510837773793646433098432639660507700851

# Compute errors (in percent)
'''trap_error = abs(((trap_result - true_value)/true_value) * 100)
simp_error = abs(((simp_result - true_value)/true_value) * 100)
romb_error = abs(((romb_result - true_value)/true_value) * 100)'''

# Measure timing for custom methods

n = range(1,20)
trap_results=[]
simp_results=[]
romb_results=[]



for i in n: 
    trap_results.append(timing_function(trapezoidal, x_data, y_data, i)[1])
    simp_results.append(timing_function(simpsons, x_data, y_data, i)[1])
    romb_results.append(timing_function(romberg, x_data, y_data, i)[1])

print(len(trap_results))
print(len(simp_results))
print(len(romb_results))

trap_error = []
simp_error = []
romb_error = []

for i in range(len(trap_results)):
    trap_error.append((abs((trap_results[i]-true_value)/true_value))*100)
    simp_error.append((abs((simp_results[i]-true_value)/true_value))*100)
    romb_error.append((abs((romb_results[i]-true_value)/true_value))*100)

print(len(trap_error))
print(len(simp_error))
print(len(romb_error))

# Print results with error analysis
'''print("\nIntegration Method Comparison")
print("=" * 80) # why 80? https://peps.python.org/pep-0008/
print(f"{'Method':<25}{'Result':<20}{'Error (%)':<20}{'Time (sec)':<15}")
print("-" * 80)
print(f"{'Custom Trapezoidal':<25}{trap_result:<20.8f}{trap_error:<20.8e}{trap_time:<15.6f}")
print(f"{'Custom Simpsons':<25}{simp_result:<20.8f}{simp_error:<20.8e}{simp_time:<15.6f}")
print(f"{'Custom Romberg':<25}{romb_result:<20.8f}{romb_error:<20.8e}{romb_time:<15.6f}")'''
#print("=" * 80)


fig, ax = plt.subplots(figsize=(15,20))

ax.scatter(n, trap_error, color='red', label='Trapezoidal')
ax.scatter(n, simp_error, color='blue', label='Simpsons')
ax.scatter(n, romb_error, color='green', label='Romberg')
ax.set_xlabel('Number of Bins')
ax.set_ylabel('Error %')
ax.set_title('Number of Bins vs Error of Approximation')

ax.legend()

plt.show()

