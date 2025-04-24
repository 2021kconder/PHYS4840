#!usr/local/Anaconda2023/bin/python3.11

################################################################
#
# PHYS4840 Lab
# Kaycee Conder 
# April 15th, 2025
#
################################################################

'''Textbook Example 9.1 Walkthrough Below via Dr. Joyce Code.'''

#laplace.py 

from numpy import empty,zeros,max
from pylab import imshow,gray,show

# Constants
M = 100         # Grid squares on a side
h = 1           # length between adjacent nodes

V = 1.0         # Voltage at top wall
target = 1e-6   # Target accuracy -- tolerance threshold for solution

# Create arrays to hold potential values
phi = zeros([M+1,M+1],float) ## 2D array 


'''
the following statement is setting a Dirichlet boundary condition on the top edge of the 2D grid
phi is a 2D NumPy array of shape (M+1, M+1) representing the potential at each point on a square grid
The notation phi[0,:] means: “all columns in row 0” — in other words, the entire top row of the grid
phi[0,:] = V sets the potential to V = 1.0 on the entire top boundary.
All other boundaries (bottom, left, and right) are implicitly left at zero 
(since phi was initialized with zeros(...)), meaning those edges are held at 0 volts.
'''
phi[0,:] = V    


phinew = empty([M+1,M+1],float)

# Main loop
delta = 1.0
while delta>target:

    # Calculate new values of the potential
    for i in range(M+1):
        for j in range(M+1):
            ## boundary conditions
            if i==0 or i==M or j==0 or j==M:
                phinew[i,j] = phi[i,j]
            else:
                phinew[i,j] = (phi[i+h,j] + phi[i-h,j] \
                                 + phi[i,j+h] + phi[i,j-h])/4.

    # Calculate maximum difference from old values
    delta = max(abs(phi-phinew))

    phi = phinew  # the new value of phi is set to what we just found for phinew
    phinew = phi  # phinew will be immediately overwritten in the next iteration, so 
                  # we assign it a placeholder value of the correct size until then, 
                  # which might as well be phi


    # shorthand way of doing this is to simply swap the two arrays around
    #   phi,phinew = phinew,phi

# Make a plot
imshow(phi)
gray()
show()

'''Now we want to answer in lab questions via 3D_solver.py'''

import numpy as np
import matplotlib.pyplot as plt

# Constants
N = 30              # Grid size (cube of size N x N x N)
h = 1               # Grid spacing
V = 1.0             # Voltage on the top face (z = 0)
target = 1e-6       # Convergence criterion

# Initialize the potential arrays
phi = np.zeros((N+1, N+1, N+1), dtype=float)
phinew = np.empty_like(phi)

# Apply boundary condition: top face (z = 0) at V, others at 0
phi[:,:,0] = V

# Iterative solution using Gauss-Seidel-like update
delta = 1.0
iteration = 0
while delta > target:
    iteration += 1
    for i in range(1, N):
        for j in range(1, N):
            for k in range(1, N):
                phinew[i,j,k] = (phi[i+h,j,k] + phi[i-h,j,k] +
                                 phi[i,j+h,k] + phi[i,j-h,k] +
                                 phi[i,j,k+h] + phi[i,j,k-h]) / 6.0

    # Preserve boundary conditions
    phinew[:,:,0] = V
    phinew[:,:,N] = 0
    phinew[:,0,:] = 0
    phinew[:,N,:] = 0
    phinew[0,:,:] = 0
    phinew[N,:,:] = 0

    delta = np.max(np.abs(phi - phinew))
    phi, phinew = phinew, phi

    if iteration % 10 == 0:
        print(f"Iteration {iteration}, max delta = {delta:.2e}")

# Visualization: middle slice in z-direction
mid_z = N // 2
plt.figure(figsize=(6,5))
plt.imshow(phi[:,:,mid_z], origin='lower', cmap='inferno')
plt.colorbar(label='Potential $\Phi$')
plt.title(f"Midplane slice at z = {mid_z}")
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()


# ------------------------------
# Question 1, Part A
# ------------------------------

'''There are now six terms in phi_new instead of four because in 3 dimensions, 
we need to consider the adjacent points in not just the +-x and +- y directions, 
but ALSO the +- z direction, therefore a total of 6 terms for each dimension and 
direction.'''

# ------------------------------
# Question 1, Part B
# ------------------------------

'''We change the grid size to be smaller given the increased number given 
the fact that the 3D version takes MUCH longer to perform iterations than 
the 2D version, and therefore to finish running the code before the end 
of the millenia, we need to change the grid size to be much smaller.'''

# ------------------------------
# Question 1, Part C
# ------------------------------

'''When we change the convergence criterion, this directy impacts the time
it takes to run the code/the numer of iterations we perform. Specifcally, 
when we increase the precision, i.e. decrease the convergence criterion, it takes longer
for our iterations to reach this target value. Whereas, if we decrease the precision, 
i.e. increase the convergence criterion, it takes far less iterations to achieve
this target leve.'''

# ------------------------------
# Question 1, Part D
# ------------------------------

'''In the 2D case, the boundary conditions are being preserved via the lines

for j in range(M+1):
            ## boundary conditions
            if i==0 or i==M or j==0 or j==M:
                phinew[i,j] = phi[i,j]
            else:
                phinew[i,j] = (phi[i+h,j] + phi[i-h,j] \
                                 + phi[i,j+h] + phi[i,j-h])/4.

whereas in the 3D case, the boundary conditions are being 
preserved via the lines 

phinew[:,:,0] = V
phinew[:,:,N] = 0
phinew[:,0,:] = 0
phinew[:,N,:] = 0
phinew[0,:,:] = 0
phinew[N,:,:] = 0

where the above lines exist in the first layer of a for loop for range(1,N).'''


'''Our next set of questions deals with the following piece of code.'''

#poisson_using_roll.py

import numpy as np

def laplacian_operator(Phi, dx, dy, dz):
    """
    Compute the Laplacian of a scalar field Phi (i.e., apply the Poisson operator)
    using central finite differences on a 3D uniform grid.

    Parameters:
    - Phi : 3D numpy array of shape (nx, ny, nz)
    - dx, dy, dz : grid spacings in x, y, z directions

    Returns:
    - laplacian : 3D numpy array of the same shape as Phi
    """

    laplacian = (
        (np.roll(Phi, -1, axis=0) - 2*Phi + np.roll(Phi, 1, axis=0)) / dx**2 +
        (np.roll(Phi, -1, axis=1) - 2*Phi + np.roll(Phi, 1, axis=1)) / dy**2 +
        (np.roll(Phi, -1, axis=2) - 2*Phi + np.roll(Phi, 1, axis=2)) / dz**2
    )

    return laplacian


# ------------------------------
# Question 2, Part A
# ------------------------------

''' np.roll() takes two key arguments:

x: input array
shift: this determines the number of values that 
will be shifted, in original order, to the beginning of the 
new array. 

The np.roll() takes a certain number of elements at the end of 
an array and 'rolls' them, in the original order in which
they exist in the input array, and moves them to the front 
of the original array.'''

