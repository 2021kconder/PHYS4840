#!usr/local/Anaconda2023/bin/python3.11


#####################################
#
# Class 7: Numerical Limits of Python, Optimization, Timing & GitHub
# Author: <Kaycee Conder> 
# Date: February 11th, 2024
#
#####################################

from math import sqrt 
import numpy as np
import timeit
import numpy as np
import time
import sys
import pandas as pd
#####################################

'''In Class Exercises'''

print('Double Precision Format:')
print(0.1+0.2)
print(repr(0.1+0.2))

print()


x = 1.0
y = 1.0 + (1e-14)*sqrt(2)

answer_1 = 1e14*(y-x)
answer_2 =  sqrt(2)

print('Precision Percentage Difference:')
print('Answer 1 = ', answer_1)
print('Answer 2 = ', answer_2)
print('% Difference= ', (1- (answer_2/answer_1))*100)

print()


setup_code_2 = "import numpy as np; my_array = np.arange(100000)"

#### compare the speed of 
# 		sum([x**2 for x in range(100000)])
#               vs
#       np.sum(my_array**2)
##
## for 100 iterations, then 1000

loop_time_100 = timeit.timeit("[x**2 for x in range(100000)]", setup=setup_code_2, number=100)
numpy_time_100 = timeit.timeit("np.sum(my_array**2)", setup=setup_code_2, number=100)

loop_time_1000 = timeit.timeit("[x**2 for x in range(100000)]", setup=setup_code_2, number=100)
numpy_time_1000 = timeit.timeit("np.sum(my_array**2)", setup=setup_code_2, number=1000)

print("100 Iteration List comprehension time: ","%.5f"%loop_time_100 ," seconds")
print("100 Iteration Numpy Time =","%.5f"%numpy_time_100," seconds")

print()

print("1000 List comprehension time: ","%.5f"%loop_time_1000 ," seconds")
print("1000 Iteration Numpy Time =","%.5f"%numpy_time_1000," seconds")
print()

#####################################

'''Meat of the Class: Timing Exercises'''

#np.loadtxt()

filepath = '/d/cha1/kconder/PHYS4840_labs/NGC6341.dat'

start_numpy = time.perf_counter()

blue, green, red, probability = np.loadtxt(filepath, usecols=(8,14,26,32), unpack=True)

print('len(green):', len(green))

end_numpy = time.perf_counter()

print('Time to run loadtxt version:', end_numpy-start_numpy, 'seconds')
print()

#pandas

start_pandas = time.perf_counter()

df = pd.read_csv(filepath, delim_whitespace=True, comment='#', header=None, skiprows=54)

blue = df.iloc[:,8]
green = df.iloc[:,14]
red = df.iloc[:,26]

blue = blue.to_numpy()
green = green.to_numpy()
red =red.to_numpy()

print('len(green):', len(green))

end_pandas = time.perf_counter()

print('Time to run pandas version:', end_pandas-start_pandas, 'seconds')
print()

#Parsing

start_parse = time.perf_counter()


blue, green, red = [], [], []

with open(filepath, 'r') as file:
    for line in file:
        if line.startswith('#'):
            continue
       
        columns = line.split()
        
        blue.append(float(columns[8]))   
        green.append(float(columns[14])) 
        red.append(float(columns[26]))   

blue = np.array(blue)
green = np.array(green)
red = np.array(red)

print("len(green): ", len(green))

end_parse = time.perf_counter()

print('Time to run parsing version:', end_parse-start_parse, 'seconds')
print()


#THE FASTEST FILE LOADER IS NP.LOADTXT!!!!!