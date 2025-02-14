#!usr/local/Anaconda2023/bin/python3.11

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import conder_functions_lib as cfl

######################################################################################

#HOMEWORK TWO <Kaycee Conder> Spring 2025 PHYS4840

######################################################################################
'''PROBLEM ZERO: FINISHING LAB FROM FEBRUARY 6TH'''

#importing the data

filepath = '/d/cha1/kconder/PHYS4840_labs/NGC6341.dat'

#Column  9: F336W calibrated magnitude (BLUE)
#Column 15: F428W calibrated magnitude (GREEN)
#Column 27: F814W calibrated magnitude (RED)

F336W, F438W, F814W, prob = np.loadtxt(filepath, usecols=(8,14,26,32), unpack=True)



'''quality_cut = np.where( (F814W   > -99.) &\
                   (F438W  > -99)  &\
                   (F336W > -99)  &\
                   (prob != -1))'''


color = F336W-F814W
mag   = F336W

probability = prob

#plotting the data

plt.rcParams['font.family']= 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

fig, ax = plt.subplots(figsize=(8,16))

#ax.scatter(np.log10(final_color),final_blue, color='firebrick', s=2, alpha=0.2)
#scatter = ax.plot(color, mag, "k.", markersize = 4, alpha = 0.2)
scatter=ax.scatter(color,mag, cmap='BrBG', c=(prob),s=2)

fig.colorbar(scatter, orientation='vertical', label='Membership Probability')
ax.set_xlabel('Color: B-R')
ax.set_ylabel('Magnitude:B')
ax.set_title('HST Data for the Globular Cluster NGC6341')
ax.set_xlim(-2,5)
ax.set_ylim(14, 25)
ax.invert_yaxis()

ax.set_facecolor('slategray')
plt.show()



######################################################################################
'''PROBLEM ONE: LOADING DATA, ISOCHRONE MODEL'''

#For the plot below, we will be creating a color-magnitude diagram (B-V) vs V_mag for our isochrone model and the ngc6341 data. 

#######################
#
# The below work is Dr. Joyce's from 
# GitHub that she gave us permission 
# to use in this homework assignment. 
#
#######################

plt.rcParams['font.family']= 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

load_file = '/d/cha1/kconder/PHYS4840_labs/homeworks/MIST_v1.2_feh_m1.75_afe_p0.0_vvcrit0.4_HST_WFPC2.iso.cmd'

log10_isochrone_age_yr, F606, F814,\
logL, logTeff, phase= np.loadtxt(load_file, usecols=(1,14,18,6,4,22), unpack=True, skiprows=14)

age_Gyr_1e9 = (10.0**log10_isochrone_age_yr)/1e9
age_Gyr = age_Gyr_1e9

age_selection = np.where((age_Gyr > 12) & (age_Gyr <= 13.8)) 

color_selected = F606[age_selection]-F814[age_selection]
magnitude_selected = F606[age_selection]

Teff = 10.0**logTeff
Teff_for_desired_ages =  Teff[age_selection]
logL_for_desired_ages =  logL[age_selection]

phases_for_desired_age = phase[age_selection]
desired_phases = np.where(phases_for_desired_age <= 3)

cleaned_color = color_selected[desired_phases]
cleaned_magnitude = magnitude_selected[desired_phases]
cleaned_Teff = Teff_for_desired_ages[desired_phases]
cleaned_logL = logL_for_desired_ages[desired_phases]

filename = '/d/cha1/kconder/PHYS4840_labs/homeworks/NGC6341.dat'
blue, green, red, probability = np.loadtxt(filename, usecols=(8, 14, 26, 32), unpack=True)

magnitude = blue
color     = blue - red

quality_cut = np.where( (red   > -99.) &\
					    (blue  > -99)  &\
					    (green > -99)  &\
					    (probability != -1))
 
print("quality_cut: ", quality_cut )


def format_axes(ax):
    ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.5)  # Larger major ticks
    ax.tick_params(axis='both', which='minor', labelsize=12, length=3, width=1)    # Minor ticks
    ax.minorticks_on()  # Enable minor ticks


##########################
#
# KAYCEE CONDER WORK BELOW 
# Now we want to find the shift between our isochrone
# and our HST data given the distance modulus and an 
# assumed distance of 8.63 kpc
#
###########################

ngc6341_distance = 8.63e3 #parsecs
cmd_shift = cfl.distance_modulus(ngc6341_distance)
print(cmd_shift)


fig, ax = plt.subplots()

ax.scatter(color[quality_cut], magnitude[quality_cut]-cmd_shift, s=4,alpha=0.2, color='purple', label='HST Data')
ax.plot(cleaned_color, cleaned_magnitude, color='green', markersize=2, linestyle='-', label='Isochrone')
ax.invert_yaxis()
ax.set_xlabel('Color: B-R', fontsize=18)
ax.set_ylabel('Magnitude: B', fontsize=18)
ax.set_title('Globular Cluster NGC6341 CMD')
ax.legend()
ax.plot()
plt.savefig("isochrone_vs_hst_CMD.png", dpi=300)


plt.show()

######################################################################################
'''PROBLEM TWO: LOG VS LINEAR GRID COMBOS'''

x = np.arange(-100,101,1)
print(x)

y = x**4

fig, ax = plt.subplots(1,3, figsize=(20,15))

#linear-linear plot
ax[0].plot(x,y,color='green', lw=5, label='y=x$^{4}$')
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')
ax[0].set_title('XY Linear Plot')

#setting our graph scale lines (linear on linear graph)
for i in x:
	ax[0].axvline(x=i, color='black', alpha=0.1)

for j in y: 
	ax[0].axhline(y=j, color='black', alpha=0.1)

ax[0].legend(loc='lower right')


#linear data, log graph scale plot

ax[1].plot(x,y,color='red', lw=5, label='y=x$^{4}$')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_xlabel('X')
ax[1].set_ylabel('Y')
ax[1].set_title('XY Log-Log Plot')
ax[1].set_ylim((10**1,10**10))


#setting our graph scale lines (linear on log graph)
for i in x:
	ax[1].axvline(x=i, color='black', alpha=0.1)

for j in y: 
	ax[1].axhline(y=j, color='black', alpha=0.1)



ax[1].legend(loc='lower right')

#log data, linear graph scale plot ASK DR JOYCE ABOUT HOW WE WOULD WANT TO REPRESENT THIS? BC LIKE IT'S LOGGED DATA

ax[2].plot(np.log10(x), np.log10(y), color='orange', lw=5, label='log(y)=log(x$^{4}$)')
ax[2].set_xlabel('log(X)')
ax[2].set_ylabel('log(Y)')
ax[2].set_title('Linear XY Plot')

#setting our graph scale lines (log on linear graph)

ax[2].grid(color='lightgray', linestyle='--', linewidth=0.5)


ax[2].legend(loc='lower right')

fig.tight_layout(pad=10.0)
plt.show()

######################################################################################
'''PROBLEM THREE: BOOK 3.1, SUNSPOTS (YAY!)'''

plt.rcParams['font.family']= 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

#Col 1 = Months since January 1749
#Col 2 = Number of Sunspots during that given month 


#Loading in the data
filepath_three = '/d/cha1/kconder/PHYS4840_labs/homeworks/sunspots.txt'
month_num, sunspots = np.loadtxt(filepath_three, usecols=(0,1), unpack=True)


#Plotting our sunspots per month 

fig, ax = plt.subplots(3,figsize=(30,15))


'''PART A'''

ax[0].scatter(month_num/12 + 1749, sunspots, color='darkgreen') #putting our x axis in terms of the year (easier to visualize time scales)
ax[0].set_xlabel('Year')
ax[0].set_ylabel('Number of Sunspots')
ax[0].text(1801,120,'Dalton Minimum', fontsize='10') #labeling important period for funsies lol 
ax[0].axvspan(1797,1827, color='gold', alpha=0.5)
ax[0].set_title('Sunspot Count per Month Since 1749')

'''PART B'''

ax[1].scatter(month_num[:1001]/12 + 1749, sunspots[:1001], color='seagreen')
ax[1].set_xlabel('Year')
ax[1].set_ylabel('Number of Sunspots')
ax[1].text(1805, 120,'Dalton Minimum', fontsize='20') #labeling important period since it's really highlighted by this data criterion 
ax[1].axvspan(1797,1827, color='gold', alpha=0.5)
ax[1].set_title('Sunspot Count From ~ 1749-1833')


'''PART C'''

#finding my running average
running_aves = []
for i in range(len(sunspots)):
	running_aves.append((sum(sunspots[i-5:i+6]))/10)



ax[2].scatter(month_num/12 + 1749, sunspots, color='darkgreen') #putting our x axis in terms of the year (easier to visualize time scales)
ax[2].plot(month_num/12 + 1749, running_aves, color='cyan', label='Running Average')
ax[2].set_xlabel('Year')
ax[2].set_ylabel('Number of Sunspots')
ax[2].text(1801,120,'Dalton Minimum', fontsize='10') #labeling important period for funsies lol 
ax[2].axvspan(1797,1827, color='gold', alpha=0.5)
ax[2].set_title('Running Average of Sunspot Count Since 1749')
ax[2].legend()

fig.tight_layout(pad=10.0)
plt.show()


######################################################################################
'''PROBLEM FOUR: GITHUB, 1'''

#PART A: The following set of commands will guarantee me a clean push to my GitHub assuming that my code is 'ahead' fo  my main branch

'''When the local machine code is ahead of the main branch, we need to make sure that we are using git pull!'''

# > git init
# > git remote -v
# > git pull origin main --allow-unrelated-histories
# > git branch --set-upstream-to=origin/main main

# The following two lines are where we switch it up: we want to make sure
# that we are pulling changes, and we can do so by clarifying rebasing as a means
# of pulling. We then pull. 

# > git config pull.rebase true 
# > git pull
# > git add 'myfile.py'
# > git commit -m "Commit Message"
# > git push origin 
# > ENTER USERNAME
# > ENTER PASSWORD (IN THIS CASE PASSKEY)

#PART B

'''After navigating to the repository with your git location...'''

# > rm -rf .git
# > rm -rf .gitignore


######################################################################################
'''PROBLEM FIVE: GITHUB, 2'''


'''All code for corresponding labs and homeworks has successfully been uploaded to GitHub,
primarily utilizing the code from Part 4a to successfully do so.'''

