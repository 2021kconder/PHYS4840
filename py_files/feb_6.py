#!usr/local/Anaconda2023/bin/python3.11

import numpy as np 
import sys 
import matplotlib.pyplot as plt 

########################################

'''@author: Kaycee Conder
   Lab: February 6th, 2024'''

   #PERSONAL ACCESS TOKEN 

########################################

#importing the data

filepath = '/d/cha1/kconder/PHYS4840_labs/NGC6341.dat'

#Column  9: F336W calibrated magnitude (BLUE)
#Column 15: F428W calibrated magnitude (GREEN)
#Column 27: F814W calibrated magnitude (RED)

F336W, F438W, F814W = np.loadtxt(filepath, usecols=(8,14,26), unpack=True)


#plotting the data

plt.rcParams['font.family']= 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

fig, ax = plt.subplots()



ax.scatter(F336W-F814W,F336W, color='green')
ax.set_xlabel('Color:B-R')
ax.set_ylabel('Magnitude:B')
ax.set_xscale('log')
ax.set_yscale('log')
#ax.set_xlim(-3,1.5)
ax.invert_yaxis()
plt.show()
