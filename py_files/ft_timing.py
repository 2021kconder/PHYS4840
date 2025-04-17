#!/usr/bin/env python3
"""
Compare Fourier Transform Implementations
PHYS 4840 - Minimal benchmarking
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import fourier_transform as ft

sizes = [2,4,8,16, 32, 64, 128, 256, 512, 1024]
def compare_speeds():
    sizes = [2,4,8,16, 32, 64, 128, 256, 512, 1024]
    x_2 = np.linspace(0,2,2)
    x_4 = np.linspace(0,4,4)
    x_8 = np.linspace(0, 8,8)
    x_16 = np.linspace(0,16,16)
    x_32 = np.linspace(0,32,32)
    x_64 = np.linspace(0,64,64)
    x_128 = np.linspace(0,128,128)
    x_256 = np.linspace(0,256,256)
    x_515 = np.linspace(0,512,512)
    x_1024 = np.linspace(0,1024,1024)

    x_sets = [x_2, x_4, x_8, x_16, x_32, x_64, x_128, x_256, x_515, x_1024]

    times_dft = []
    times_radix2 = []
    times_bluestein = []
    times_zeropad = []
    times_fft_numpy = []

    #for N in sizes:
    #   x = np.random.rand(N)
    x_dft = []
    x_bluestein = []
    x_zeropad = []
    x_numpy = []
    x_radix2 = []

    #DFT Timing Tests
    for i in x_sets:  
        start_dft_time = time.time()
        x_dft = ft.dft(i)
        end_dft_time = time.time()

        total_dft_time = end_dft_time - start_dft_time
        times_dft.append(total_dft_time)

    #Radix2 Timing Tests
    for i in x_sets: 
        start_radix2_time = time.time()
        x_radix2 = ft.fft_radix2(i)
        end_radix2_time = time.time()

        total_radix2_time = end_radix2_time - start_radix2_time
        times_radix2.append(total_radix2_time)

    #Bluestein Timing Tests
    for i in x_sets:
        start_bluestein_time = time.time()
        x_bluestein = ft.fft_bluestein(i)
        end_bluestein_time = time.time()

        total_bluestein_time = end_bluestein_time - start_bluestein_time
        times_bluestein.append(total_bluestein_time)

    #Zeropad Timing Tests 
    for i in x_sets: 
        start_zeropad_time = time.time()
        x_zeropad = ft.fft_zeropad(i)
        end_zeropad_time = time.time()

        total_zeropad_time = end_zeropad_time - start_zeropad_time
        times_zeropad.append(total_zeropad_time)

    #Numpy Timing Tests
    for i in x_sets: 
        start_fft_time = time.time()
        x_numpy = np.fft.fft(i)
        end_fft_time = time.time()

        total_fft_time = end_fft_time - start_fft_time
        times_fft_numpy.append(total_fft_time)

    return sizes, times_dft, times_radix2, times_bluestein, times_zeropad, times_fft_numpy


fig, ax = plt.subplots()

ax.plot(compare_speeds()[0], compare_speeds()[1], label='DFT', color='red')
ax.plot(compare_speeds()[0], compare_speeds()[2], label='Radix2', color='orange')
ax.plot(compare_speeds()[0], compare_speeds()[3], label='Bluestein', color='green')
ax.plot(compare_speeds()[0], compare_speeds()[4], label='Zeropad', color='blue')
ax.plot(compare_speeds()[0], compare_speeds()[5], label='Numpy', color='purple')
ax.legend()
ax.set_xlabel('log(Sizes)')
ax.set_ylabel('log(Times)')
ax.set_title('Fourier Transfrom Types vs Computation Time')
ax.set_xscale('log')
ax.set_yscale('log')


plt.show()