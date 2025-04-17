#!usr/local/Anaconda2023/bin/python3.11

import numpy as np 
import matplotlib.pyplot as plt
import random 
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.io import wavfile 


################################################################
#
# PHYS 4840 HOMEWORK SIX 
# <Kaycee Conder> 
# Spring 2025 PHYS4840
#
################################################################


# ---------------------------------
# PROBLEM ZERO: Double Pendelum
# ---------------------------------

'''

For this problem, the 'april_3_upated.py' file was run three 
separate times in order to yield three separate text files and .gifs
depicting double pendelum motion with perturbed initial conditions
(CHAOS!)The following code is copied below. 

Run #1 Initial Conditions: 

Theta1 = 1.5714053027367674
Theta2 = 1.5549393377502192
Omega1 = -0.0005715965360789388
Omega2 = 0.07453779416084813

Run #2 Initial Conditions: 

Theta1 = 1.5764319894442065
Theta2 = 1.5617220135269898
Omega1 = 0.0267172828762198
Omega2 = -0.0194895234542056

Run #3 Initial Conditions: 

Theta1 = 1.573007458351615
Theta2 = 1.5786344328219657
Omega1 = 0.027632702206151494
Omega2 = 0.022237759686197386

'''

#####################################
import numpy as np
from math import *
import random

# Constants
g = -9.81  # Gravity (m/s^2)
l = 0.4   # Length of pendulum arms (m)
m = 1.0   # Mass of pendulums (kg)

# Set random seed for reproducibility if needed (remove this for full randomness)
random.seed()  # Seed from system time, removes fixed randomness

# Randomly perturb initial conditions to simulate chaos
theta1 = np.radians(90 + random.uniform(-1, 1))  # Slight random perturbation
theta2 = np.radians(90 + random.uniform(-1, 1))  # Slight random perturbation
omega1 = random.uniform(-0.1, 0.1)  # Small random initial velocity
omega2 = random.uniform(-0.1, 0.1)  # Small random initial velocity
# State vector r = [theta1, theta2, omega1, omega2]

print('Theta1 =', theta1)
print('Theta2 =', theta2)
print('Omega1 =', omega1)
print('Omega2 =', omega2)

r0 = np.array([theta1, theta2, omega1, omega2])  

# Time parameters
dt = 0.01  # Time step
t_max = 10.0  # Simulation duration: sets number of TIME STEPS
t = np.arange(0, t_max, dt)

# Equations of motion for the double pendulum
def equations(r):
    ## assign the four variables we need to evolve to ONE vector r 
    ## that holds them all
    theta1, theta2, omega1, omega2 = r
    delta_theta = theta2 - theta1

    # Define the four equations for the system
    ftheta1 = omega1
    ftheta2 = omega2

    ## HINT: the expressions for fomega1, fomega2 are quite long,
    ## so create smaller expressions to hold the denominators
    denom1 = (2 * m * l ** 2)
    denom2 = ...

    fomega1 = (((-omega1**2)*sin( (2*theta1) - (2*theta2))) + (2*(omega2**2)*sin(theta1-theta2)) + ((g/l)*(sin(theta1-(2*theta2))+(3*sin(theta1)))))/(3-cos((2*theta1)-(2*theta2)))

    fomega2 = (((4*(omega1**2))*sin(theta1 - theta2)) + ((omega2**2)*sin((2*theta1)-(2*theta2))) + (2*(g/l)*(sin((2*theta1)-(sin(theta2))))))/(3-cos((2*theta1)-(2*theta2)))

    return np.array([ftheta1, ftheta2, fomega1, fomega2], float)

# Runge-Kutta 4th order method
def rk4_step(r, dt):
    k1 = dt * equations(r)
    k2 = dt * equations(r+0.5*k1)
    k3 = dt * equations(r+0.5*k2)
    k4 = dt * equations(r+k3)
    return r + (k1 + 2 * k2 + 2 * k3 + k4) / 6

## this is a CLEVER way to hold all of your data in one object
## R is a vector of lenght t (time steps) that will hold the evolution
## of all FOUR of your variables
## r0 is a VECTOR initialized to r0 = [0,0,0,0]
R = np.zeros((len(t), 4))
R[0] = r0

# Integrate equations and save data
## remember: numerical integration --> for loop
for i in range(1, len(t)):
    R[i] = rk4_step(R[i - 1], dt)

# Extract angles and angular velocities
theta1_vals, theta2_vals, omega1_vals, omega2_vals = R.T

# Convert to Cartesian coordinates for visualization
x1 = l * np.sin(theta1_vals)
y1 = -l * np.cos(theta1_vals)
x2 = x1 + l * np.sin(theta2_vals)
y2 = y1 - l * np.cos(theta2_vals)

# Save data
#np.savetxt("/d/cha1/kconder/PHYS4840_labs/homeworks/perturbed_pendulum_THREE_final.txt", np.column_stack([t, x1, y1, x2, y2]),
           #header="time x1 y1 x2 y2", comments="")

#########################################
# GENERATING THE VIDEO!!!!!
#########################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter

# Load the data
data = np.loadtxt("/d/cha1/kconder/PHYS4840_labs/homeworks/perturbed_pendulum_THREE_final.txt", skiprows=1)
t, x1, y1, x2, y2 = data.T

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.set_xlabel("X position (m)")
ax.set_ylabel("Y position (m)")
ax.set_title("Double Pendulum Simulation")

# Plot the pivot point (fixed at the origin)
pivot, = ax.plot([], [], 'ko', label="Pivot")

# Create lines for the pendulum arms
line1, = ax.plot([], [], color='green', label="Mass 1 Path")
line2, = ax.plot([], [], color='purple', label="Mass 2 Path")

# Create markers for the masses
mass1, = ax.plot([], [], marker='*', color='green', label="Mass 1", markersize=8)
mass2, = ax.plot([], [], marker='*', color='purple', label="Mass 2", markersize=8)

ax.legend()

# Initial conditions for the animation
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    mass1.set_data([], [])
    mass2.set_data([], [])
    return line1, line2, mass1, mass2

# Update function for the animation
def update(frame):
    # Get the current positions of the masses
    x1_pos = x1[frame]
    y1_pos = y1[frame]
    x2_pos = x2[frame]
    y2_pos = y2[frame]
    
    # Update the data for the lines
    line1.set_data([0, x1_pos], [0, y1_pos])  # Line from pivot to mass 1
    line2.set_data([x1_pos, x2_pos], [y1_pos, y2_pos])  # Line from mass 1 to mass 2

    # Update the positions of the masses
    mass1.set_data(x1_pos, y1_pos)
    mass2.set_data(x2_pos, y2_pos)
    
    return line1, line2, mass1, mass2

# Set up the animation
# Adjust interval and fps
interval_ms = 10  # 200 ms between frames
fps = 1000 // interval_ms  # Ensure the fps matches the interval

ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=interval_ms)

# Save the animation as a video (MP4 file)
writergif=PillowWriter(fps=30)

ani.save('/d/cha1/kconder/PHYS4840_labs/homeworks/perturbed_pendulum_THREE_final.gif', writer=writergif)

plt.show()

'''Now we want to plot the data from these runs on the same graph.'''

run_one = np.loadtxt("/d/cha1/kconder/PHYS4840_labs/homeworks/perturbed_pendulum_ONE_final.txt", skiprows=1)
t_one, x1_one, y1_one, x2_one, y2_one = run_one.T

run_two = np.loadtxt("/d/cha1/kconder/PHYS4840_labs/homeworks/perturbed_pendulum_TWO_final.txt", skiprows=1)
t_two, x1_two, y1_two, x2_two, y2_two = run_two.T

run_three = np.loadtxt("/d/cha1/kconder/PHYS4840_labs/homeworks/perturbed_pendulum_THREE_final.txt", skiprows=1)
t_three, x1_three, y1_three, x2_three, y2_three = run_three.T

# Plot
fig, ax = plt.subplots(3, figsize=(6,10))

ax[0].plot(x1_one, y1_one, marker='.', label="Mass 1 Run 1 (Path)")
ax[0].plot(x2_one, y2_one, marker='.', label="Mass 2 Run 1 (Path)", color="red")
ax[0].scatter([0], [0], color="black", marker="o", label="Pivot")  # Fixed pivot point
ax[1].plot(x1_two, y1_two, marker='.', label="Mass 1 Run 2 (Path)", color='orange')
ax[1].plot(x2_two, y2_two, marker='.', label="Mass 2 Run 2 (Path)", color="cyan")
ax[2].plot(x1_three, y1_three, marker='.', label="Mass 1 Run 3 (Path)", color='green')
ax[2].plot(x2_three, y2_three, marker='.', label="Mass 2 Run 3 (Path)", color="purple")


# Mark initial positions
ax[0].scatter([x1_one[0]], [y1_one[0]], color="blue", marker="+", s=100, label="Mass 1 Run 1(Start)", zorder=3)
ax[0].scatter([x2_one[0]], [y2_one[0]], color="red", marker="+", s=100, label="Mass 2 Run 1(Start)", zorder=3)
ax[1].scatter([x1_two[0]], [y1_two[0]], color="orange", marker="+", s=100, label="Mass 1 Run 2(Start)", zorder=3)
ax[1].scatter([x2_two[0]], [y2_two[0]], color="cyan", marker="+", s=100, label="Mass 2 Run 2(Start)", zorder=3)
ax[2].scatter([x1_three[0]], [y1_three[0]], color="green", marker="+", s=100, label="Mass 1 Run 3(Start)", zorder=3)
ax[2].scatter([x2_three[0]], [y2_three[0]], color="purple", marker="+", s=100, label="Mass 2 Run 3(Start)", zorder=3)

# Formatting
ax[0].set_xlabel("X position (m)")
ax[0].set_ylabel("Y position (m)")
ax[0].set_title("Run 1 Double Pendulum Motion")
ax[0].legend()
ax[0].axis("equal")
ax[0].grid()

ax[1].set_xlabel("X position (m)")
ax[1].set_ylabel("Y position (m)")
ax[1].set_title("Run 2 Double Pendulum Motion")
ax[1].legend()
ax[1].axis("equal")
ax[1].grid()

ax[2].set_xlabel("X position (m)")
ax[2].set_ylabel("Y position (m)")
ax[2].set_title("Run 3 Double Pendulum Motion")
ax[2].legend()
ax[2].axis("equal")
ax[2].grid()

# Save figure
plt.savefig('/d/cha1/kconder/PHYS4840_labs/homeworks/motions.png')
plt.show()
plt.close()



# ---------------------------------
# PROBLEM ONE: Complete Labs
# ---------------------------------

'''
These homework portions are completed
in the py files april_8.py and april_10.py 
in the py_files repository of my github. 

They are however, attached below (commented out).
'''

##################
#
#April 8th Lab Work
#Author: <Kaycee Conder> 
#Course: Spring 2025 PHYS4840
#
###################

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

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from math import *
# import fourier_series as fs
# import time

# def do_fourier(TERMS, wave):

#     start_time = time.time()
#     # Create x values for plotting
#     x_range = (-2*np.pi, 2*np.pi)
#     num_points = 10000
#     x = np.linspace(x_range[0], x_range[1], num_points)
#     y_exact = wave(x)
    



#     # Compute Fourier series coefficients
#     a0, an, bn = fs.compute_coefficients(wave, TERMS)
    
#     # Calculate the Fourier approximation
#     y_approx = fs.fourier_series_approximation(x, a0, an, bn)
    
#     # Calculate partial approximations
#     partial_approx = fs.compute_partial_approximations(x, a0, an, bn)
    
#     end_time = time.time()

#     total_time = end_time - start_time 

#     # 1. Plot the series with TERMS set
#     '''plt.figure(figsize=(10, 6))
#     plt.plot(x, y_exact, 'k-', label='Exact')
#     plt.plot(x, y_approx, 'r-', label=f'Fourier ({TERMS} terms)')
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.xlabel('x')
#     plt.ylabel('f(x)')
#     plt.show()#plt.savefig('fourier_approximation.png')
#     plt.close()'''
    




#     # 2. Plot the power spectral density (PSD) / coefficient spectrum
#     #plt.figure(figsize=(10, 6))
    
#     # Compute magnitude of coefficients
#     n_values = np.arange(1, TERMS + 1)
    
#     # Plot coefficient magnitudes
#     '''plt.stem(n_values, an, 'g-', markerfmt='g^', label='an', basefmt=" ", linefmt='g--')
#     plt.stem(n_values, bn, 'r-', markerfmt='rs', label='bn', basefmt=" ", linefmt='r--')
    
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.xlabel('Harmonic (n)')
#     plt.ylabel('Coefficient Magnitude')
#     plt.yscale('log')
#     plt.show()#plt.savefig('coefficient_spectrum.png')
#     plt.close()'''
    




#     # 3. Convergence and error analysis
#     # Calculate error for each partial approximation
#     errors = []
#     term_counts = range(1, TERMS + 1)
    
#     for i, approx in enumerate(partial_approx):
#         error = np.sqrt(np.mean((y_exact - approx)**2))
#         errors.append(error)
    


#     '''plt.figure(figsize=(10, 6))
#     plt.plot(term_counts, errors, 'bo-')
#     plt.grid(True, alpha=0.3)
#     plt.xlabel('Number of Terms')
#     plt.ylabel('RMS Error')
#     plt.title('Numer of Terms vs RMS Error')
#     plt.legend()
#     plt.show()#plt.savefig('convergence_rate_linear.png')
#     plt.close()'''
    
#     # Log-log plot to better visualize error scaling
#     '''plt.figure(figsize=(10, 6))
#     plt.loglog(term_counts, errors, 'bo-')
#     plt.grid(True, alpha=0.3, which='both')
#     plt.xlabel('Number of Terms')
#     plt.ylabel('RMS Error')
#     plt.show()#plt.savefig('convergence_rate_log.png')
#     plt.close()'''
    




#     # 4. Create an animation showing how the approximation improves with terms
#     '''fig, ax = plt.subplots(figsize=(10, 6))
    
#     exact_line, = ax.plot(x, y_exact, 'k-', label='Exact')
#     approx_line, = ax.plot([], [], 'r-', label='Fourier Approximation')
    
#     ax.set_xlabel('x')
#     ax.set_ylabel('f(x)')
#     ax.grid(True, alpha=0.3)
#     ax.legend()
    
#     # Set axis limits
#     margin = 0.1
#     ax.set_ylim(-1.5, 1.5)
#     ax.set_xlim(x_range)
    
#     # Text to display current number of terms
#     terms_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
#     def init():
#         """Initialize animation"""
#         approx_line.set_data([], [])
#         terms_text.set_text('')
#         return approx_line, terms_text
    
#     def update(frame):
#         """Update animation for each frame"""
#         n_terms = frame + 1
#         # Use pre-computed partial approximation
#         y_approx = partial_approx[n_terms - 1]
        
#         approx_line.set_data(x, y_approx)
#         terms_text.set_text(f'Terms: {n_terms}')
#         return approx_line, terms_text
    
#     ani = FuncAnimation(fig, update, frames=TERMS,
#                        init_func=init, blit=True, interval=200)
    
#     plt.show()
#     ani.save('fourier_animation.gif', writer='pillow', fps=5)
#     plt.close()'''
    
#     return total_time, errors, term_count


# def ecg_like_signal(x):

#     def r(val, variation=0.1):
#         return val * np.random.uniform(1 - variation, 1 + variation)

#     # Normalize x to [0, 2pi]
#     x_norm = x % (2 * np.pi)
#     # P-wave with randomized amplitude, center, and width
#     p_wave = r(0.25, 0.2) * np.exp(-((x_norm - r(0.7 * np.pi, 0.05))**2) / (r(0.1 * np.pi, 0.05)**2))
#     # QRS complex: one positive peak and two negative deflections
#     qrs1 = r(1.0, 0.2) * np.exp(-((x_norm - r(np.pi, 0.05))**2) / (r(0.05 * np.pi, 0.05)**2))
#     qrs2 = r(-0.3, 0.2) * np.exp(-((x_norm - r(0.9 * np.pi, 0.05))**2) / (r(0.04 * np.pi, 0.05)**2))
#     qrs3 = r(-0.2, 0.2) * np.exp(-((x_norm - r(1.1 * np.pi, 0.05))**2) / (r(0.04 * np.pi, 0.05)**2))
#     # T-wave with random parameters
#     t_wave = r(0.5, 0.2) * np.exp(-((x_norm - r(1.4 * np.pi, 0.05))**2) / (r(0.1 * np.pi, 0.05)**2))
    
#     return p_wave + qrs1 + qrs2 + qrs3 + t_wave

# def my_signal(x):
#     return half_rectified_sine(np.sin(x) * np.cos(np.sin(10 * x)))



# TERMS = 1000
# wave = ecg_like_signal


# term_count = []
# error_mean = []
# times = []

# for i in range(0,100):
#     error_mean.append((np.mean(do_fourier(i, wave)[1])))
#     times.append(do_fourier(i,wave)[0])

# term_count = np.linspace(0,100, 100)
# fig, ax = plt.subplots()

# ax.plot(term_count, times)
# ax.set_xlabel('Term Count')
# ax.set_ylabel('Computation Time')
# ax.set_title('Computation time vs Term Count for FT')
# ax.legend()

# plt.show()



# fig, ax = plt.subplots()

# ax.plot(times, error_mean)
# ax.set_xlabel('Computation Time')
# ax.set_ylabel('Mean RMS Error')
# ax.set_title('Computation Time vs Meam RMS Error for FT')
# ax.legend()

# plt.show()


##################
#
#April 10th Lab Work
#Author: <Kaycee Conder> 
#Course: Spring 2025 PHYS4840
#
##################
"""
Simplified Fourier Transform Demonstration
-----------------------------------------
This script demonstrates basic Fourier Transform applications:
1. Analysis of a sinusoid with noise
2. Simple audio processing with note identification

PHYS 4840 - Mathematical and Computational Methods II
"""

# import numpy as np
# import matplotlib.pyplot as plt
# import fourier_transform as ft
# from scipy.io import wavfile


# def half_rectified_sine(x):
#     """Half-rectified sine wave: max(0, sin(x))"""
#     return np.maximum(0, np.sin(x))

# #signal = half_rectified_sine(np.sin(x) * np.cos(np.sin(10 * x)))

# ################################


# def create_frequency_grid(signal_length, sample_rate):
#     """
#     Create a frequency grid for the given signal length and sample rate.
#     """
#     return np.linspace(0, sample_rate/2, signal_length//2)


# def find_peaks(spectrum, frequencies, threshold=0.1, min_distance=50):
#     """
#     Find peaks in the frequency spectrum with better isolation.
#     Simple but robust implementation that finds truly significant peaks.
#     """
#     # Find local maxima in isolated regions
#     peak_indices = []
#     max_val = np.max(spectrum)
#     min_val = threshold * max_val
    
#     # First find candidate peaks
#     candidates = []
#     for i in range(1, len(spectrum)-1):
#         # Must be a local maximum and above threshold
#         if (spectrum[i] > spectrum[i-1] and 
#             spectrum[i] > spectrum[i+1] and 
#             spectrum[i] > min_val):
#             candidates.append(i)
    
#     # Sort candidates by magnitude
#     candidates.sort(key=lambda idx: spectrum[idx], reverse=True)
    
#     # Take peaks in order of magnitude, ensuring minimum distance between them
#     for candidate in candidates:
#         # Check if this peak is far enough from all existing peaks
#         isolated = True
#         for peak in peak_indices:
#             dist = abs(candidate - peak)
#             if dist < min_distance:
#                 isolated = False
#                 break
        
#         if isolated:
#             peak_indices.append(candidate)
    
#     # Sort peak indices by frequency
#     peak_indices.sort()
    
#     # Extract the frequencies and magnitudes
#     peak_freqs = frequencies[peak_indices]
#     peak_mags = spectrum[peak_indices]
    
#     return peak_freqs, peak_mags


# def identify_note(frequency):
#     """
#     Identify musical note from frequency.
#     """
#     # Define A4 = 440 Hz
#     A4 = 440.0
    
#     # Define note names
#     note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
#     # Calculate semitones from A4
#     if frequency <= 0:
#         return "Unknown"
    
#     semitones = 12 * np.log2(frequency / A4)
#     semitones_rounded = round(semitones)
    
#     # Calculate octave and note index
#     octave = 4 + (semitones_rounded + 9) // 12
#     note_idx = (semitones_rounded + 9) % 12
    
#     # Calculate cents (how far from the exact note)
#     cents = 100 * (semitones - semitones_rounded)
    
#     return f"{note_names[note_idx]}{octave} ({cents:+.0f} cents)"


# '''def demo_noisy_sinusoid():

#     # Create a signal
#     fs = 1000  # Sampling frequency (Hz)
#     duration = 1.0  # Signal duration (seconds)
#     t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
#     # Signal with two frequencies
#     f1 = 0.0#Frequency 1
#     f2 = 1.0#and 2
#     A1 = -2.0#Amplitude 1
#     A2 = 2.0 #and 2
#     signal = np.sin(2*np.pi*f1*t) + 0.5 * np.sin(2*np.pi *f2*t)
    
#     # Add some noise
#     noisy_signal = signal + 0.2 * np.random.randn(len(t))
    
#     # Compute FT using our module

#     # Create frequency grid
    
#     # Find peaks in the spectrum
#     peak_freqs, peak_mags = find_peaks(magnitudes, freqs)
    
#     # Plot time domain signal and frequency spectrum
#     plt.figure(figsize=(10, 8))
    
#     # Time domain
#     plt.subplot(2, 1, 1)
#     plt.plot(t, noisy_signal)
#     plt.grid(True, alpha=0.3)
#     plt.title('Noisy Sinusoid (Time Domain)')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
    
#     # Frequency domain
#     plt.subplot(2, 1, 2)
#     plt.plot(freqs, magnitudes)
#     plt.grid(True, alpha=0.3)
#     plt.title('Frequency Spectrum')
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Magnitude')
#     plt.xlim(0, fs/2)  # Nyquist frequency
    
#     # Mark the true frequencies
#     plt.axvline(f1, color='r', linestyle='--', label=f'{f1} Hz')
#     plt.axvline(f2, color='g', linestyle='--', label=f'{f2} Hz')
    
#     # Mark detected peaks
#     for i, (freq, mag) in enumerate(zip(peak_freqs, peak_mags)):
#         plt.plot(freq, mag, 'ro', markersize=8)
#         plt.text(freq, mag*1.1, f"{freq:.1f} Hz", ha='center')
    
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('sinusoid_analysis.png')
#     plt.show()'''


# def demo_audio_processing():
    
#     # Load the audio file
#     fs, audio_data = wavfile.read('marty_robins.wav')
    
#     # Convert to mono if stereo and normalize
#     if len(audio_data.shape) > 1:
#         audio_data = np.mean(audio_data, axis=1)
#     audio_data = audio_data / np.max(np.abs(audio_data))
    
#     # Process the ENTIRE audio
#     audio_segment = audio_data
#     n_samples = len(audio_segment)
#     duration = n_samples / fs
    
#     # Create time axis for the full audio
#     t = np.linspace(0, duration, n_samples)
    
#     print(f"Starting FT with {n_samples} samples...")
#     # Compute FFT for all samples
#     X = np.fft.fft(audio_segment)
    
#     # Get only the first half of the spectrum (positive frequencies)
#     half_n = len(X) // 2
#     magnitudes = np.abs(X[:half_n]) / len(audio_segment)
    
#     # Create frequency grid limited to human hearing range (20 Hz - 20 kHz)
#     # But still maintain the correct number of points
#     human_hearing_max = min(20000, fs/2)  # Either 20 kHz or Nyquist frequency, whichever is lower
#     freqs = np.linspace(0, human_hearing_max, half_n)
    
#     # Find peaks in the spectrum
#     peak_freqs, peak_mags = find_peaks(magnitudes, freqs, threshold=0.2)
    
#     # Identify musical notes from peaks
#     notes = []
#     for freq in peak_freqs:
#         notes.append(identify_note(freq))
    
#     # Plot the audio and its spectrum
#     plt.figure(figsize=(12, 8))
    
#     # Time domain - downsample for display only
#     plt.subplot(2, 1, 1)
#     max_plot_points = 10000
#     if len(t) > max_plot_points:
#         plot_step = len(t) // max_plot_points
#         plt.plot(t[::plot_step], audio_segment[::plot_step])
#     else:
#         plt.plot(t, audio_segment)
#     plt.grid(True, alpha=0.3)
#     plt.title('Complete Audio Signal (Time Domain)')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
    
#     # Frequency domain - focus on 0-1000 Hz range
#     plt.subplot(2, 1, 2)
#     # Find the index corresponding to 1000 Hz for plotting
#     idx_1000hz = int(1000 * len(freqs) / (fs/2))
#     plt.plot(freqs[:idx_1000hz], magnitudes[:idx_1000hz])
#     plt.grid(True, alpha=0.3)
#     plt.title('Frequency Spectrum with Identified Notes')
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Magnitude')
    
#     # Mark detected peaks and notes
#     for i, (freq, mag, note) in enumerate(zip(peak_freqs, peak_mags, notes)):
#         if freq <= 1000:  # Only annotate peaks below 1000 Hz
#             plt.plot(freq, mag, 'ro', markersize=8)
#             plt.text(freq, mag*1.1, f"{note}", ha='center')
    
#     plt.tight_layout()
#     plt.savefig('audio_analysis.png')
#     plt.show()
    
#     # Define a simple bandpass filter around the main frequency component
#     if len(peak_freqs) > 0:
#         main_freq = peak_freqs[0]
#         filter_width = 50  # Hz
        
#         print(f"Applying bandpass filter around {main_freq:.1f} Hz")
        
#         # Create a filtered spectrum - by copying the original spectrum
#         X_filtered = np.zeros_like(X, dtype=complex)
        
#         # Apply bandpass filter in frequency domain
#         for i in range(half_n):
#             freq = freqs[i]
#             if abs(freq - main_freq) < filter_width:
#                 X_filtered[i] = X[i]
#                 # Also set the corresponding negative frequency
#                 if i > 0:  # Skip DC component
#                     X_filtered[len(X)-i] = X[len(X)-i]
        


#         filtered_signal = np.fft.ifft(X_filtered)
        
#         # Plot the original and filtered signals
#         plt.figure(figsize=(12, 8))
        
#         # Time domain comparison - downsample for display only
#         plt.subplot(2, 1, 1)
#         if len(t) > max_plot_points:
#             plot_step = len(t) // max_plot_points
#             plt.plot(t[::plot_step], audio_segment[::plot_step], alpha=0.7, label='Original')
#             plt.plot(t[::plot_step], np.real(filtered_signal[::plot_step]), label='Filtered')
#         else:
#             plt.plot(t, audio_segment, alpha=0.7, label='Original')
#             plt.plot(t, np.real(filtered_signal), label='Filtered')
#         plt.grid(True, alpha=0.3)
#         plt.title(f'Original vs Filtered Signal (Bandpass around {main_freq:.1f} Hz)')
#         plt.xlabel('Time (s)')
#         plt.ylabel('Amplitude')
#         plt.legend()
        
#         # Frequency domain comparison - focus on 0-1000 Hz range
#         plt.subplot(2, 1, 2)
#         filtered_mags = np.abs(X_filtered[:half_n]) / len(filtered_signal)
#         plt.plot(freqs[:idx_1000hz], magnitudes[:idx_1000hz], alpha=0.7, label='Original Spectrum')
#         plt.plot(freqs[:idx_1000hz], filtered_mags[:idx_1000hz], label='Filtered Spectrum')
#         plt.axvline(main_freq, color='r', linestyle='--', 
#                    label=f'Main Frequency: {main_freq:.1f} Hz ({identify_note(main_freq)})')
#         plt.grid(True, alpha=0.3)
#         plt.title('Original vs Filtered Spectrum')
#         plt.xlabel('Frequency (Hz)')
#         plt.ylabel('Magnitude')
#         plt.legend()
        
#         plt.tight_layout()
#         plt.savefig('audio_filtering.png')
#         plt.show()
        
#         # Save the filtered audio
#         wavfile.write('marty_robins_filtered.wav', fs, np.real(filtered_signal).astype(np.float32))
#         print(f"Filtered audio saved as 'marty_robins_filtered.wav' (kept frequencies around {main_freq:.1f} Hz)")


# def main():
    
#     # Demo 1: Noisy Sinusoid Analysis
#     #demo_noisy_sinusoid()
    
#     # Demo 2: Audio Analysis with Note Identification
#     demo_audio_processing()


# if __name__ == "__main__":
#     main()




# ---------------------------------
# PROBLEM TWO: Book 8.15
# ---------------------------------

'''PART A:
-------------

Answer is derived in the written version of the homework.'''


'''PART B:
------------- 

The necessary equations have already been defined for 
this problem in PROBLEM 0. We simply need to alter the 
constants for the purposes of this problem AND solve
for the energy.'''

# Redefining constants
g_three = g #gravity [m/s^2], same as Problem 0
l_three = 40e-2 #length of arms [m]
m_three = m #bob mass [kg], same as Problem 0


# Setting initial conditions
theta1_three = np.radians(90) #same as problem 0
theta2_three = np.radians(90) #same as problem 0
omega1_three = 0.0 #same as problem 0
omega2_three = 0.0 #same as problem 0

# State vector r = [theta1, theta2, omega1, omega2]
r0_three = np.array([theta1_three, theta2_three, omega1_three, omega2_three])  

# Time parameters
dt_three = 0.01  # Time step
t_max_three = 100.0  # Simulation duration: sets number of TIME STEPS
t_three = np.arange(0, t_max_three, dt_three)

## this is a CLEVER way to hold all of your data in one object
## R is a vector of lenght t (time steps) that will hold the evolution
## of all FOUR of your variables
## r0 is a VECTOR initialized to r0 = [0,0,0,0]
R_three = np.zeros((len(t_three), 4))
R_three[0] = r0_three

# Integrate equations and save data
## remember: numerical integration --> for loop
for i in range(1, len(t_three)):
    R_three[i] = rk4_step(R_three[i - 1], dt_three)

# Extract angles and angular velocities
theta1_vals_three, theta2_vals_three, omega1_vals_three, omega2_vals_three = R_three.T

# Defining the equation for our total energy 
def total_energy_dp(theta_one, theta_two, omega_one, omega_two, m, l):
    g = -9.81
    kinetic_energy = m*(l**2)*( (omega_one**2) + ((omega_two**2)/2) + omega_one*omega_two*cos(theta_one-theta_two))
    potential_energy = -m*g*l* ( (2*cos(theta_one)) + cos(theta_two))

    total_energy = kinetic_energy + potential_energy

    return total_energy

energy_range = []

for i in range(len(theta1_vals_three)):
    energy_range.append(total_energy_dp(theta1_vals_three[i], theta2_vals_three[i], omega1_vals_three[i], omega2_vals_three[i], m_three, l_three))


fig, ax = plt.subplots()

ax.plot(t_three, energy_range, color='purple')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Energy [J]')
ax.set_title('Double Pendulum: Total Energy as a Function of Time')
ax.legend()

plt.show()


'''Note that in the above plot, there seems to be some violation of the conservation of energy
in the small time ranges, converging to a value of 0 (i.e. energy is not a function of time).
I am not sure why this is the case, but overall the trend of the graph demonstrates a lack of 
dependence on time in the upper limit.'''



'''PART C:
------------- 

The necessary equations have already been defined for Problem 0, we simply now need 
to utilize them here for this specific case.'''

#Convert to Cartesian coordinates for visualization
x1_three = l_three * np.sin(theta1_vals_three)
y1_three = -l_three * np.cos(theta1_vals_three)
x2_three = x1_three + l_three * np.sin(theta2_vals_three)
y2_three = y1_three - l_three * np.cos(theta2_vals_three)

# Save data
np.savetxt("double_pendulum_data_problem_three.txt", np.column_stack([t, x1, y1, x2, y2]),
           header="time x1 y1 x2 y2", comments="")
# Load the data
data_3_ani = np.loadtxt("double_pendulum_data_problem_three.txt", skiprows=1)
t_3_ani, x1_3_ani, y1_3_ani, x2_3_ani, y2_3_ani = data_3_ani.T

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.set_xlabel("X position (m)")
ax.set_ylabel("Y position (m)")
ax.set_title("Double Pendulum Simulation Problem 8.15")

# Plot the pivot point (fixed at the origin)
pivot, = ax.plot([], [], 'ko', label="Pivot")

# Create lines for the pendulum arms
line1_3_ani, = ax.plot([], [], 'b-', label="Mass 1 Path")
line2_3_ani, = ax.plot([], [], 'r-', label="Mass 2 Path")

# Create markers for the masses
mass1_3_ani, = ax.plot([], [], 'b*', label="Mass 1", markersize=8)
mass2_3_ani, = ax.plot([], [], 'r*', label="Mass 2", markersize=8)

ax.legend()

# Initial conditions for the animation
def init():
    line1_3_ani.set_data([], [])
    line2_3_ani.set_data([], [])
    mass1_3_ani.set_data([], [])
    mass2_3_ani.set_data([], [])
    return line1_3_ani, line2_3_ani, mass1_3_ani, mass2_3_ani

# Update function for the animation
def update(frame):
    # Get the current positions of the masses
    x1_pos_3_ani = x1_3_ani[frame]
    y1_pos_3_ani = y1_3_ani[frame]
    x2_pos_3_ani = x2_3_ani[frame]
    y2_pos_3_ani = y2_3_ani[frame]
    
    # Update the data for the lines
    line1_3_ani.set_data([0, x1_pos_3_ani], [0, y1_pos_3_ani])  # Line from pivot to mass 1
    line2_3_ani.set_data([x1_pos_3_ani, x2_pos_3_ani], [y1_pos_3_ani, y2_pos_3_ani])  # Line from mass 1 to mass 2

    # Update the positions of the masses
    mass1_3_ani.set_data(x1_pos_3_ani, y1_pos_3_ani)
    mass2_3_ani.set_data(x2_pos_3_ani, y2_pos_3_ani)
    
    return line1_3_ani, line2_3_ani, mass1_3_ani, mass2_3_ani

# Set up the animation
# Adjust interval and fps
interval_ms = 10  # 200 ms between frames
fps = 1000 // interval_ms  # Ensure the fps matches the interval

ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=interval_ms)

# Save the animation as a video (MP4 file)
writergif=PillowWriter(fps=30)

ani.save('double_pendulum_simulation_problem_three.gif', writer=writergif)

plt.show()




# ---------------------------------
# PROBLEM THREE: Audio Compression
# ---------------------------------

'''Write an algorithm that compresses audio.'''

def compress_audio_fft(audio, keep_ratio, fs):
    """
    Compress audio by keeping only the top `keep_ratio` frequency components (by magnitude).
    
    Parameters:
        audio (np.ndarray): Time-domain audio signal
        keep_ratio (float): Fraction of strongest frequencies to keep (0 < keep_ratio <= 1)
        
    Returns:
        compressed_audio (np.ndarray): Reconstructed audio from compressed frequency domain
        X_compressed (np.ndarray): The compressed spectrum (mostly zero)
    """
    N = len(audio)
    X = np.fft.fft(audio)
    magnitudes = np.abs(X)
    
    # Determine how many frequencies to keep
    new_N = int(keep_ratio * N)#these are the N values we want to keep


    # Get indices of highest frequencies by magnitude
    N_indices = np.argsort(magnitudes)[-new_N:]

    # Create a compressed version of the spectrum
    compressed_X = X[N_indices]
    

    # Inverse FFT to get time-domain signal
    compressed_audio = np.fft.ifft(compressed_X)

    wavfile.write('/d/cha1/kconder/PHYS4840_labs/homeworks/filtered_audio.wav', fs, compressed_audio.astype(np.float32) )
    
    return compressed_audio, compressed_X

fs, audio_data = wavfile.read('/d/cha1/kconder/PHYS4840_labs/py_files/audio.wav')

compress_audio_fft(audio_data, 0.1, fs)

# ---------------------------------
# PROBLEM FOUR: Writing Prompt
# ---------------------------------

'''Q: If a scientists is analyzing astronomical 
images, why should they ensure that the images are NOT 
jpegs?'''


'''A: If a scientist is analyzing astronomical images, they should ensure that the images are NOT jpegs because they are missing a 
significant amount of information necessary for astronomical analysis such as photometry or machine learning. More specifically, when 
creating a .jpeg, the image is broken up into 8x8 pixel blocks, each of which is treated independently. A 2D discrete cosine transformation 
is applied to each of these blocks, turning 64 spatial pixel values into 64 frequency coefficients. Note that this transformation for the 
simple fact that this process is simple and fast. During said transformation, high frequency coefficient terms are very greatly reduced or 
deleted, compressing the image in the process. As a result, pixel information that is crucial to many common astronomical processes is 
deleted when creating a .jpeg. Fits files should be used instead of .jpegs for astronomical work. '''





