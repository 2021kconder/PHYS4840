#!usr/local/Anaconda2023/bin/python3.11

from math import sqrt
from math import sin, tanh
import numpy as np
import matplotlib.pyplot as plt
import conder_functions_lib as cfl
import time 

######################################################################################

#HOMEWORK THREE <Kaycee Conder> Spring 2025 PHYS4840

#ASK WHETHER OR NOT TO SCATTER IT OR PLOT IT, I THINK PLOT? 

######################################################################################
'''PROBLEM ZERO: FINISHING LAB FROM FEBRUARY 20TH'''


########################################
#
# Trapezoidal integration method via 
# Dr Miller's code. Alterations to 
# complete the code made by me. 
#
########################################



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




########################################
#
# Simpson's Rule integration method via 
# Dr Miller's code. Alterations to 
# complete the code made by me. 
#
########################################


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


    for k in range(1, N, 2):  # Odd indices (weight 4)
        xk = a + k * h
        yk = np.interp(xk, x_values, y_values)
        integral += 4 * yk

    for k in range(2, N, 2):  # Even indices (weight 2)
        xk = a + k * h
        yk = np.interp(xk, x_values, y_values)
        integral += 2 * yk

    scaled_integral = (h/3) * integral




    return scaled_integral  # Final scaling




########################################
#
# Romberg integration method via 
# Dr Miller's code. Alterations to 
# complete the code made by me. 
#
########################################


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



########################################
#
# Timing function code from Dr. Miller.   
#
########################################

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



########################################
#
# Problem Specific Specifications
#
########################################

# Function to integrate
def function(x):
    return x * np.exp(-x)


# Precompute data for fair comparisons
x_data = np.linspace(0, 1, 100000000)  # High-resolution x values
y_data = function(x_data)

# Testing parameters
N = range(1,100,1) # Number of intervals
max_order = 3 # Romberg's accuracy level

# True integral value
true_value = 0.26424111765711535680895245967707826510837773793646433098432639660507700851


########################################
#
# Previous work in Dr. Miller's code
# that has been commented out given that
# we are looking at a range of N values, 
# not a single N value.
#
########################################


'''trap_time, trap_result = timing_function(trapezoidal, x_data, y_data, N)
simp_time, simp_result = timing_function(simpsons, x_data, y_data, N)
romb_time, romb_result = timing_function(romberg, x_data, y_data, max_order)'''



# Compute errors (in percent)
'''trap_error = abs(((trap_result - true_value)/true_value) * 100)
simp_error = abs(((simp_result - true_value)/true_value) * 100)
romb_error = abs(((romb_result - true_value)/true_value) * 100)'''

# Measure timing for custom methods



########################################
#
# Looping through range of N values 
# for all functions. 
#
########################################


'''In this first iteration of the code, we are accounting for the issues that occur with Simpson's Rule. 
Specially, Simpson's rule does not work when your N is an even number. Therefore, for our range of N values, we 
only want to look at N values that are divisible by 2. This can be remedied a separate way by means of the 
3/8 Rule, but this will be examined later in the homework.'''


n_even = range(2,21,2) #make sure that this is in intervals of 2 so that we can make simpson's work



trap_results_even = []
simp_results_even = []
romb_results_even = []


for i in n_even: 
    trap_results_even.append(timing_function(trapezoidal, x_data, y_data, i)[1])
    simp_results_even.append(timing_function(simpsons, x_data, y_data, i)[1])
    romb_results_even.append(timing_function(romberg, x_data, y_data, i)[1])


trap_error_even = []
simp_error_even = []
romb_error_even = []

for i in range(len(trap_results_even)):
    trap_error_even.append((abs((trap_results_even[i]-true_value)/true_value))*100)
    simp_error_even.append((abs((simp_results_even[i]-true_value)/true_value))*100)
    romb_error_even.append((abs((romb_results_even[i]-true_value)/true_value))*100)


trap_time_even = []
simp_time_even = []
romb_time_even = []

for i in n_even: 
    trap_time_even.append(timing_function(trapezoidal,x_data, y_data, i)[0])
    simp_time_even.append(timing_function(simpsons,x_data, y_data, i)[0])
    romb_time_even.append(timing_function(romberg,x_data, y_data, i)[0])


########################################
#
# Plotting our Even N Results. 
#
########################################


#making sure that the font of my plots is Times
plt.rcParams['font.family']= 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

fig, ax = plt.subplots(1,3, figsize=(15,5))

ax[0].plot(n_even, trap_error_even, color='red', label='Trapezoidal')
ax[0].plot(n_even, simp_error_even, color='blue', label='Simpsons')
ax[0].plot(n_even, romb_error_even, color='green', label='Romberg')
ax[0].set_xlabel('Number of Bins')
ax[0].set_ylabel('Absolute Error %')
ax[0].set_title('Even Ns: Number of Bins vs Absolute Error')
ax[0].legend()

ax[1].plot(n_even, np.log10(trap_time_even), color='red', label='Trapezoidal')
ax[1].plot(n_even, np.log10(simp_time_even), color='blue', label='Simpsons')
ax[1].plot(n_even, np.log10(romb_time_even), color='green', label='Romberg')
ax[1].set_xlabel('Number of Bins')
ax[1].set_ylabel('log(Computation Time) [s]')
ax[1].set_title('Even Ns: Number of Bins vs Computation Time')
ax[1].legend()

ax[2].plot(np.log10(trap_time_even), trap_error_even, color='red', label='Trapezoidal')
ax[2].plot(np.log10(simp_time_even), simp_error_even, color='blue', label='Simpsons')
ax[2].plot(np.log10(romb_time_even), romb_error_even, color='green', label='Romberg')
ax[2].set_xlabel('log(Computation Time) [s]')
ax[2].set_ylabel('Aboslute Error %')
ax[2].set_title('Even Ns: Computation Time vs Absolute Error')
ax[2].legend()


plt.show()



########################################
#
# Looping through range of N values 
# for all functions. 
#
########################################


'''In this first iteration of the code, we are NOT accounting for the issues that occur with Simpson's Rule. 
As a result, our Simpson's Rule Error will spike at even N vaules.'''

n_all = range(1,21) #make sure that this is in intervals of 2 so that we can make simpson's work



trap_results_all=[]
simp_results_all=[]
romb_results_all=[]


for i in n_all: 
    trap_results_all.append(timing_function(trapezoidal, x_data, y_data, i)[1])
    simp_results_all.append(timing_function(simpsons, x_data, y_data, i)[1])
    romb_results_all.append(timing_function(romberg, x_data, y_data, i)[1])


trap_error_all = []
simp_error_all = []
romb_error_all = []

for i in range(len(trap_results_all)):
    trap_error_all.append((abs((trap_results_all[i]-true_value)/true_value))*100)
    simp_error_all.append((abs((simp_results_all[i]-true_value)/true_value))*100)
    romb_error_all.append((abs((romb_results_all[i]-true_value)/true_value))*100)


trap_time_all = []
simp_time_all = []
romb_time_all = []

for i in n_all: 
    trap_time_all.append(timing_function(trapezoidal,x_data, y_data, i)[0])
    simp_time_all.append(timing_function(simpsons,x_data, y_data, i)[0])
    romb_time_all.append(timing_function(romberg,x_data, y_data, i)[0])


########################################
#
# Plotting our Full N Range Results. 
#
########################################


#making sure that the font of my plots is Times
plt.rcParams['font.family']= 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

fig, ax = plt.subplots(1,3, figsize=(15,5))

ax[0].plot(n_all, trap_error_all, color='red', label='Trapezoidal')
ax[0].plot(n_all, simp_error_all, color='blue', label='Simpsons')
ax[0].plot(n_all, romb_error_all, color='green', label='Romberg')
ax[0].set_xlabel('Number of Bins')
ax[0].set_ylabel('Absolute Error %')
ax[0].set_title('All Ns: Number of Bins vs Absolute Error')
ax[0].legend()

ax[1].plot(n_all, np.log10(trap_time_all), color='red', label='Trapezoidal')
ax[1].plot(n_all, np.log10(simp_time_all), color='blue', label='Simpsons')
ax[1].plot(n_all, np.log10(romb_time_all), color='green', label='Romberg')
ax[1].set_xlabel('Number of Bins')
ax[1].set_ylabel('log(Computation Time) [s]')
ax[1].set_title('All Ns: Number of Bins vs Computation Time')
ax[1].legend()

ax[2].plot(np.log10(trap_time_all), trap_error_all, color='red', label='Trapezoidal')
ax[2].plot(np.log10(simp_time_all), simp_error_all, color='blue', label='Simpsons')
ax[2].plot(np.log10(romb_time_all), romb_error_all, color='green', label='Romberg')
ax[2].set_xlabel('log(Computation Time) [s]')
ax[2].set_ylabel('Aboslute Error')
ax[2].set_title('All Ns: Computation Time vs Absolute Error')
ax[2].legend()


plt.show()



######################################################################################
'''PROBLEM ONE: GAIA & SED NUMERICAL INTEGRATION'''

#Part A

print('Problem One, Part A Answers:')

########################################
#
# Initial Setup
#
########################################



#loading in all of our data 
filepath_one = '/d/cha1/kconder/PHYS4840_labs/homeworks/figures/GAIA_G.csv'
filepath_two = '/d/cha1/kconder/PHYS4840_labs/homeworks/figures/vega_SED.csv'

vega_wavelength, vega_flux = np.loadtxt(filepath_two, usecols=(0,1), unpack=True, delimiter=',', skiprows=1)
gaia_col_one, gaia_col_two = np.loadtxt(filepath_one, usecols=(0,1), unpack=True, delimiter=',', skiprows=1)

#plotting out our curves 
fig, ax = plt.subplots(2, figsize=(7,5))

ax[0].plot(vega_wavelength, vega_flux, color='blue')
ax[0].set_xlabel('Wavelength')
ax[0].set_ylabel('Flux')
ax[0].set_xscale('log')
ax[0].set_title('Vega Spectrum')

ax[1].plot(gaia_col_one, gaia_col_two, color='green')
ax[1].set_xlabel('Wavelength')
ax[1].set_title('Gaia')

fig.tight_layout(pad=2.0)
plt.show()



########################################
#
# Previously Defined Integration Methods
#
########################################

vega_trap = trapezoidal(vega_flux, vega_wavelength, 20) 
vega_simp = simpsons(vega_flux, vega_wavelength, 20)
vega_romb = romberg(vega_flux, vega_wavelength, 20)

gaia_trap = trapezoidal(gaia_col_two, gaia_col_one, 20)
gaia_simp = simpsons(gaia_col_two, gaia_col_one, 20)
gaia_romb = romberg(gaia_col_two, gaia_col_one, 20)

print('Initial Trapz Vega =', vega_trap)
print('InitialSimps Vega =', vega_simp)
print('Initial Romb Vega =',  vega_romb)
print('np.trapz Comparison =', np.trapz(vega_flux, vega_wavelength))

print()

print('Initial Trapz Gaia =', gaia_trap)
print('Initial Simps Gaia =', gaia_simp)
print('Initial Romb Gaia =',  gaia_romb)
print('np.trapz Comparison =', np.trapz(gaia_col_two, gaia_col_one))

print()

'''Examining the plot above, we recognize that the tail of the Vega spectrum is going to create errors within 
our integration methods, particularly Romberg. Therefore, we can remove the tail end of this data to improve the 
accuracy of our integration method.'''

new_vega = np.where(vega_wavelength < 20000)

vega_wavelength = vega_wavelength[new_vega]
vega_flux = vega_flux[new_vega]

#plotting out our curves 
fig, ax = plt.subplots()

ax.plot(vega_wavelength, vega_flux, color='blue')
ax.set_xlabel('Wavelength')
ax.set_ylabel('Flux')
ax.set_xscale('log')
ax.set_title('NEW Vega Spectrum')



fig.tight_layout(pad=2.0)
plt.show()

vega_trap_mid = trapezoidal(vega_flux, vega_wavelength, 20) 
vega_simp_mid = simpsons(vega_flux, vega_wavelength, 20)
vega_romb_mid = romberg(vega_flux, vega_wavelength, 20)

gaia_trap = trapezoidal(gaia_col_two, gaia_col_one, 20)
gaia_simp = simpsons(gaia_col_two, gaia_col_one, 20)
gaia_romb = romberg(gaia_col_two, gaia_col_one, 20)

print('Slightly Improved Trapz Vega =', vega_trap_mid)
print('Slightly Improved Simps Vega =', vega_simp_mid)
print('Slightly Improved Romb Vega =',  vega_romb_mid)
print('np.trapz Comparison =', np.trapz(vega_flux, vega_wavelength))

print()

print('Slightly Improved Trapz Gaia =', gaia_trap)
print('Slightly Improved Simps Gaia =', gaia_simp)
print('Slightly Improved Romb Gaia =',  gaia_romb)
print('np.trapz Comparison =', np.trapz(gaia_col_two, gaia_col_one))


'''With the above answers, we are still utilizing the same N value for all of our methods and not changing the numerical 
integration methods from Problem 0. However, when we do this, we find dissimilar values across our integration methods. 
In order to obtain similar values for the area under our cuves, we present two changes: 

1. Changing the Simpson's Rule to account for errors in calculating N=even values via the Simpson's 3/8 Rule

2. Changing the number of bins for the trapezoid and simpson's rule method, as changing the romberg N value does not 
increase the accuracy by much whilst also increasing the run time by such a large amount that it takes several minutes 
to run. 

Below we implement these two key changes: '''


def simpsons_3_8_rule(y_values, x_values, N):
    """
    Approximates the integral using Simpson's 3/8 rule for given y_values at given x_values.

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


    if N % 2 == 0: 

        for k in range(1, N, 2):  # Odd indices (weight 4)
            xk = a + k * h
            yk = np.interp(xk, x_values, y_values)
            integral += 4 * yk

        for k in range(2, N, 2):  # Even indices (weight 2)
            xk = a + k * h
            yk = np.interp(xk, x_values, y_values)
            integral += 2 * yk

        scaled_integral = (h/3) * integral


    if N % 2 != 0: 

        for k in range(1,N,2): # Odd Indices (?) (weight 3)
            xk = a + k * h
            yk = np.interp(xk, x_values, y_values)
            integral += 3 * yk

        for k in range(2,N,2): # Even Indices (?) (weight 2)
            xk = a + k * h  
            yk = np.interp(xk, x_values, y_values)
            integral += 2 * yk 

        scaled_integral = ((3*h)/8) * integral



    return scaled_integral  # Final scaling


vega_trap_fin = trapezoidal(vega_flux, vega_wavelength, 1000) 
vega_simp_fin = simpsons_3_8_rule(vega_flux, vega_wavelength, 1000)
vega_romb_fin = romberg(vega_flux, vega_wavelength, 20)

gaia_trap_fin = trapezoidal(gaia_col_two, gaia_col_one, 1000)
gaia_simp_fin = simpsons_3_8_rule(gaia_col_two, gaia_col_one, 1000)
gaia_romb_fin = romberg(gaia_col_two, gaia_col_one, 20)

print()

print('Improved Trapz Vega =', vega_trap_fin)
print('Improved Simps Vega =', vega_simp_fin)
print('Improved Romb Vega =',  vega_romb_fin)
print('np.trapz Comparison =', np.trapz(vega_flux, vega_wavelength))

print()

print('Improved Trapz Gaia =', gaia_trap_fin)
print('Improved Simps Gaia =', gaia_simp_fin)
print('Improved Romb Gaia =',  gaia_romb_fin)
print('np.trapz Comparison =', np.trapz(gaia_col_two, gaia_col_one))

#####################################################
print()
#####################################################

#Part B

'''

In order to compute the area from the VEGA_SED file we need to follow the following steps: 

1. Import the .csv file using np.loadtxt. We need to make sure that we skip the first row 
in the file, as this first row is the header and will not load properly as a float. We then 
need to make sure that we are only using the wavelength and flux columns, as specified in class. 

2. We can plot the imported data, just to ensure that our spectrum of Vega appears to be, well,
looking like a stellar spectrum. 

3. From here, we can use either the Trapezoid, Simpson's, or Romberg method of integration. The 
factors to consider here when choosing a method are compute time and integral accuracy. In the 
particular case of our Vega spectrum, find that even small differences in method can lead to significant 
changes in our final values (given that the Vega flux values are much smaller than the GAIA values). 

4. Definitions can be made for each of these methods of integration, testing which one is the most 
accurate. In the case of our Vega spectrum, we can use the Romberg method of integration, the likes
of which has the highest accuracy but also the highest integration time. 


'''

######################################################################################
'''PROBLEM TWO: NUMERICAL INTEGRATION WRITING QUESTION'''

"""

A numerical integration problem that could be solved utilizing the methods discussed 
in class thus far would be finding the average number of solar protons measured across a 
period of time (say during a CME). With the necessary data, I would most likely use Romberg’s 
method for integration. Despite the longer computation times necessitated by this method, an 
acceptable level of accuracy can be achieved with a relatively low N value (compared to the N 
values needed to achieve a similar level of accuracy via the Trapezoid or Simpson’s methods). 
Therefore, given the prioritization of a high level of accuracy, Romberg’s rule is one I would 
implement in this case of numerical integration. 

An additional numerical integration problem that I find intriguing (though I’m not sure it applies 
to what we have learned thus far) is that of satellite orbit integration. Specifically, numerical
 integration methods can be used to integrate the second order three dimensional differential equation 
 for the motion of a satellite. According to Eshagh et al. 2005, the method of using one dimensional 
 numerical integration can be used to solve a three dimensional differential equation numerically, 
 particularly via the Runge-Kutta method. 

"""



######################################################################################
'''PROBLEM THREE: MATHEMATICAL EXPRESSIONS IN PYTHON'''

print()
print('Problem 3, Part A Answers:')
#####################################################
#Part A

'''For Part A, I chose to define a generalized function that gives the sum for any given 
function, rather than my function specifically. My function is specified after the fact 
given this general summation definition. To make a function specific definition, we need 
only put the function inside of the original definiton and remove the 'function' variable 
from the definition parameters. '''

########################################
#
# Generalized Summation Function
#
########################################

def general_sum(i,n, x_list, function):

	'''
	Computes the sum of a function, given an upper limit n, 
	and an index of summation i. This function assumes that i 
    is not equal to x, and rather, that they are distinct values. 

	Parameters: 
		n = Upper limit of the sum
		i = Lower limit of the sum
        x_list = List of x values for which you want to find the summation
		function = definition of the function whose sum you wish to compute

	Returns: 
		float: The value of the function sum over the given limits. 
	'''

	summation = 0

	#Looping through the range between the upper and lower limits given our function
	for i in range(i,n+1):
		summation += function(x_list[i])

	return summation


#defining our function for this specific problem 
def initial_function_x(x):
		y = x**2
		
		return y

x_list_example = [0,1,2,3,4,5,6,7,8,9,10]

print('Generalized Summation for y=x**2 Between 1 & 10 =',general_sum(1,10,x_list_example, initial_function_x))



########################################
#
# Problem Specific Summation Function
#
########################################


def specific_sum(n, x_list):

	'''
	Computes the sum of y=x**2, given an upper limit n, 
	and an assumed lower limit of 1. This function assumes that i 
    is not equal to x, and rather, that they are distinct values.

	Parameters: 
		n = Upper limit of the sum
        x_list = List of x values for which you want to find the summation
        function = definition of the function whose sum you wish to compute

	Returns: 
		float: The value of the function sum over the given limits. 
	'''

	#defining our function 
	def initial_function(x):
		y = x**2
		
		return y

	summation = 0

	#Looping through the range between the upper and lower limits given our function
	for i in range(1, n+1):
		summation += initial_function(x_list[i])

	return summation


#defining our function for this specific problem 


print('Function Specific Summation for y=x**2 Between 1 & 10 =', specific_sum(10, x_list_example))

#####################################################
print()
print('Problem Three, Part B Answers:')
#####################################################
#Part B

def average(n, x_list):

	''' 
	Returns the average value of a list of numbers, given 
	a certain index range n. 

	Parameters: 
		n = Upper limit of index range of x_list values
		x_list = list of values we want to find the average of 

	Returns: 
		Average value in the given list of length n
	'''

	summation = 0

	#making sure that we don't index out of range 
	if len(x_list)-1 < n: 
			average = 'Error! Index range longer than list length.'

	else: 

		#Looping through our index range in x_list
		for i in range(1,n+1):
			summation += x_list[i]

		average = summation * (1/n) #turning our sum into an average

	return average


example_list = [0,1,2,3,4,5,6,7,8,9,10]

#printing examples of each of our scenarios 
print('Average of First 11 Items of Range 1-10 =', average(11, example_list))
print('Average the First 5 Items of Range 1-10 =', average(5,example_list)) #HERE WE ASSUME THAT THE FIRST VALUE OF THE SUM IS ALWAYS ZERO AND THEREFORE IRRELEVANT
																			#THEREFORE THE 'FIRST' FIVE IS ACTUALLY THE INDEXES 1-5





#####################################################
print()
print('Problem Three, Part C Answers:')
#####################################################
#Part C

def factorial(n): 
	'''
	Returns the factorial of a specific number n. 

	Parameters:
		n = The number whose factorial you want to 
		return. 

	Return: 
		Factorial value for n
	'''

	#defining conditions where the factorial definition is not true/would fail
	if type(n) == float: 
		fact_final = 'Error! You have entered a non-integer.'

	elif n < 0: 
		fact_final = 'Error! You have entered a negative number.'

	elif n == 0: 
		fact_final = 1  

	elif n == 1: 
		fact_final = 1

	else: 
		fact = 1
		#Looping through all of the numbers less than n up to n-(n-1)
		for i in range(1,n):
			fact *= (n-i)
		fact_final = fact * n #making sure we still multiply times n as well as all the values in range n - (n-1)

	return fact_final

#printing examples of each of these conditions
print('Factorial of 17 = ',  factorial(17))
print('Factorial of 1 = ',   factorial(1))
print('Factorial of 2.0 = ', factorial(2.0))
print('Factorial of 0  = ',  factorial(0))
print('Factorial of -10 = ', factorial(-10))


######################################################################################
'''PROBLEM FOUR: TEXTBOOK EXERCISE 5.20'''

#####################################################
print()
print('Problem Four, Part A Answers:')
#####################################################

#Part A

def functionz(x):

    if x == 0: 
        y = 1

    else: 

        y = (sin(x)**2) / (x**2)

    return y



midpoints = []

def my_step(x1,x2,f1,f2, function): 
    """
    Returns the area under curve, defined by a given function, 
    via the ADJUSTED trapezoid rule. 

    Parameters: 
        x1 = 
        x2 = 
        f1 = 
        f2 = 
        function = 

    Returns: 
        Definite integral value
    """

    h = x2-x1
    I_one = ((f1 + f2)/2) * h
    midpoint = (x2 + x1)/2
    h_one = midpoint-x1
    h_two = x2- midpoint
    midpoints.append(midpoint)

    I_two = ((0.5* h_one) * (f1 + function(midpoint))) + ((0.5* h_two) * (f2 + function(midpoint)))

    if (1/3) * abs(I_two - I_one) < 10e-4: 
        return I_two

    q = function(midpoint)

    return my_step(x1, midpoint, f1, q, function) + my_step(midpoint, x2, q, f2, function)

print('Adjusted Trapezoidal Definite Integral = ', my_step(0,10,functionz(0), functionz(10), functionz))



#####################################################
print()
#####################################################

#Part B


"""
We put the function values f1 and f2 of our x1 and x2 into the function as an argument, rather than having it 
change inside of the definition given a specific function, because we save computation time by providing this 
predetermined value rather than having the function calculate f1 and f2 through every recursion. Depending on 
the number of recursions that are done, this can add a considerable amount of time to the computation. Additionally, 
providing the value as an argument, rather than an in-definition calculation/variable, prevents it from becoming 
lost/mixed up with a different value when recursion occurs.


"""


#####################################################
print()
#####################################################

#Part C

fig, ax = plt.subplots()

#setting x values for my function and finding their corresponding y values
x = np.linspace(0,10,100)

y_scatters = []
for i in midpoints:
    y_scatters.append(functionz(i))

y_plots = []
for i in x: 
    y_plots.append(functionz(i))


#plotting midpoints onto the integrand 
ax.scatter(midpoints, y_scatters, color='red', label='Slice Endpoints')
ax.plot(x, y_plots, color='blue', label='Integrand, y = sin(x)$^2$/x$^2$')
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Adjusted Trapezoid Rule Visualized')

plt.show()


