#!/usr/bin/python3.8
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


#####################################
#
#April 8th Lab Work
#Author: <Kaycee Conder> 
#Course: Spring 2025 PHYS4840
#
#####################################

"""
Simple Fourier Series Demonstration
----------------------------------
This script demonstrates Fourier series approximation with a fixed number of terms,
generating four key visualizations:
1. Series plot with terms set
2. PSD with only terms set
3. Convergence and error up to the terms set
4. Animation showing only terms set

PHYS 4840 - Mathematical and Computational Methods II
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import *
import fourier_series as fs
import time

def do_fourier(TERMS, wave):

    start_time = time.time()
    # Create x values for plotting
    x_range = (-2*np.pi, 2*np.pi)
    num_points = 10000
    x = np.linspace(x_range[0], x_range[1], num_points)
    y_exact = wave(x)
    



    # Compute Fourier series coefficients
    a0, an, bn = fs.compute_coefficients(wave, TERMS)
    
    # Calculate the Fourier approximation
    y_approx = fs.fourier_series_approximation(x, a0, an, bn)
    
    # Calculate partial approximations
    partial_approx = fs.compute_partial_approximations(x, a0, an, bn)
    
    end_time = time.time()

    total_time = end_time - start_time 

    # 1. Plot the series with TERMS set
    '''plt.figure(figsize=(10, 6))
    plt.plot(x, y_exact, 'k-', label='Exact')
    plt.plot(x, y_approx, 'r-', label=f'Fourier ({TERMS} terms)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()#plt.savefig('fourier_approximation.png')
    plt.close()'''
    




    # 2. Plot the power spectral density (PSD) / coefficient spectrum
    #plt.figure(figsize=(10, 6))
    
    # Compute magnitude of coefficients
    n_values = np.arange(1, TERMS + 1)
    
    # Plot coefficient magnitudes
    '''plt.stem(n_values, an, 'g-', markerfmt='g^', label='an', basefmt=" ", linefmt='g--')
    plt.stem(n_values, bn, 'r-', markerfmt='rs', label='bn', basefmt=" ", linefmt='r--')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel('Harmonic (n)')
    plt.ylabel('Coefficient Magnitude')
    plt.yscale('log')
    plt.show()#plt.savefig('coefficient_spectrum.png')
    plt.close()'''
    




    # 3. Convergence and error analysis
    # Calculate error for each partial approximation
    errors = []
    term_counts = range(1, TERMS + 1)
    
    for i, approx in enumerate(partial_approx):
        error = np.sqrt(np.mean((y_exact - approx)**2))
        errors.append(error)
    


    '''plt.figure(figsize=(10, 6))
    plt.plot(term_counts, errors, 'bo-')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Terms')
    plt.ylabel('RMS Error')
    plt.title('Numer of Terms vs RMS Error')
    plt.legend()
    plt.show()#plt.savefig('convergence_rate_linear.png')
    plt.close()'''
    
    # Log-log plot to better visualize error scaling
    '''plt.figure(figsize=(10, 6))
    plt.loglog(term_counts, errors, 'bo-')
    plt.grid(True, alpha=0.3, which='both')
    plt.xlabel('Number of Terms')
    plt.ylabel('RMS Error')
    plt.show()#plt.savefig('convergence_rate_log.png')
    plt.close()'''
    




    # 4. Create an animation showing how the approximation improves with terms
    '''fig, ax = plt.subplots(figsize=(10, 6))
    
    exact_line, = ax.plot(x, y_exact, 'k-', label='Exact')
    approx_line, = ax.plot([], [], 'r-', label='Fourier Approximation')
    
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set axis limits
    margin = 0.1
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(x_range)
    
    # Text to display current number of terms
    terms_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        """Initialize animation"""
        approx_line.set_data([], [])
        terms_text.set_text('')
        return approx_line, terms_text
    
    def update(frame):
        """Update animation for each frame"""
        n_terms = frame + 1
        # Use pre-computed partial approximation
        y_approx = partial_approx[n_terms - 1]
        
        approx_line.set_data(x, y_approx)
        terms_text.set_text(f'Terms: {n_terms}')
        return approx_line, terms_text
    
    ani = FuncAnimation(fig, update, frames=TERMS,
                       init_func=init, blit=True, interval=200)
    
    plt.show()
    ani.save('fourier_animation.gif', writer='pillow', fps=5)
    plt.close()'''
    
    return total_time, errors, term_count


def ecg_like_signal(x):

    def r(val, variation=0.1):
        return val * np.random.uniform(1 - variation, 1 + variation)

    # Normalize x to [0, 2pi]
    x_norm = x % (2 * np.pi)
    # P-wave with randomized amplitude, center, and width
    p_wave = r(0.25, 0.2) * np.exp(-((x_norm - r(0.7 * np.pi, 0.05))**2) / (r(0.1 * np.pi, 0.05)**2))
    # QRS complex: one positive peak and two negative deflections
    qrs1 = r(1.0, 0.2) * np.exp(-((x_norm - r(np.pi, 0.05))**2) / (r(0.05 * np.pi, 0.05)**2))
    qrs2 = r(-0.3, 0.2) * np.exp(-((x_norm - r(0.9 * np.pi, 0.05))**2) / (r(0.04 * np.pi, 0.05)**2))
    qrs3 = r(-0.2, 0.2) * np.exp(-((x_norm - r(1.1 * np.pi, 0.05))**2) / (r(0.04 * np.pi, 0.05)**2))
    # T-wave with random parameters
    t_wave = r(0.5, 0.2) * np.exp(-((x_norm - r(1.4 * np.pi, 0.05))**2) / (r(0.1 * np.pi, 0.05)**2))
    
    return p_wave + qrs1 + qrs2 + qrs3 + t_wave

def my_signal(x):
    return half_rectified_sine(np.sin(x) * np.cos(np.sin(10 * x)))



TERMS = 1000
wave = ecg_like_signal


term_count = []
error_mean = []
times = []

for i in range(0,100):
    error_mean.append((np.mean(do_fourier(i, wave)[1])))
    times.append(do_fourier(i,wave)[0])

term_count = np.linspace(0,100, 100)
fig, ax = plt.subplots()

ax.plot(term_count, times)
ax.set_xlabel('Term Count')
ax.set_ylabel('Computation Time')
ax.set_title('Computation time vs Term Count for FT')
ax.legend()

plt.show()



fig, ax = plt.subplots()

ax.plot(times, error_mean)
ax.set_xlabel('Computation Time')
ax.set_ylabel('Mean RMS Error')
ax.set_title('Computation Time vs Meam RMS Error for FT')
ax.legend()

plt.show()