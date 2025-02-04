#! /usr/local/Anaconda2023/bin/python



#####################################
#
# Class 5: Linear and Log + Plotting
# Author: <Kaycee Conder> 
#
#####################################
import numpy as np
import matplotlib.pyplot as plt
import conder_functions_lib as cfl #importing functions library 


#making sure that the font of my plots is Times
plt.rcParams['font.family']= 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']



# Defining plot values 
x = np.linspace(1, 100, 500)  # x values
y = cfl.y(x) 



#initializing subplots
fig, ax = plt.subplots(3, figsize=(10,30))

###PLOT 1: a linear plot of y vs x

ax[0].plot(x,y,color='red', label='y=2x$^{3}$')
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')
ax[0].set_title('Linear X-Y Plot')
ax[0].legend(loc='best')


### PLOT 2: a log-log plot of y vs x

ax[1].plot(x,y,color='green', label='y=2x$^{3}$')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_xlabel('X')
ax[1].set_ylabel('Y')
ax[1].set_title('Logarithmic X-Y Plot')
ax[1].legend(loc='best')

## PLOT 3: a plot of log(x) vs log(y)

ax[2].plot(np.log10(x),np.log10(y),color='purple', label='y=2x$^{3}$')
ax[2].set_xlabel('log(X)')
ax[2].set_ylabel('log(Y)')
ax[2].set_title('Logarithmic X-Y Plot')
ax[2].legend(loc='best')

plt.tight_layout(pad=15)
plt.show()
