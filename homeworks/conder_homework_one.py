#!usr/local/Anaconda2023/bin/python3.11

from math import sqrt

import numpy as np
######################################################################################

#HOMEWORK ONE <Kaycee Conder> Spring 2025 PHYS4840

######################################################################################
'''PROBLEM ONE: FLOW OF CONTROL'''

#Done in Google Drawings. Attached Separately. 

######################################################################################
'''PROBLEM TWO: SANDWITCH OPTIMIZATION'''



#least spare change in only quarters and dollars 
#contains at least one half sandwitch but no more than one half of any given sandwitch



def maximize_sammies(max_cash):

	ham_p = 3.65
	apple_brie_p = 4.25
	pbj_p = 3.00
	turkey_p = 3.35


	sammie_change = []
	sammie_combos = []

	#nested loop like the cookie problem advanced cookir problem
	for full_ham in range(int(max_cash//ham_p)+1):
		for full_apple in range(int(max_cash//apple_brie_p)+1):
			for full_pbj in range(int(max_cash//pbj_p)+1):
				for full_turkey in range(int(max_cash//turkey_p)+1):
					#making sure that we have no more than 2 half sandwitches of each kind (either 0 or 1)
					for half_ham in range(0,2):
						for half_apple in range(0,2): 	
							for half_pbj in range(0,2):
								for half_turkey in range(0,2):
									#finding the total cost for each combination 
									cost = ( 
										(full_ham*ham_p) +
										(half_ham*0.6*ham_p) +
										(full_apple*apple_brie_p) +
										(half_apple*0.6*apple_brie_p) +
										(full_pbj*pbj_p) +
										(half_pbj*0.6*pbj_p)+
										(full_turkey*turkey_p) +
										(half_turkey*0.6*turkey_p)
										)	

									#making sure we stay in budget
									if cost <= max_cash: 	
										total_half = (half_ham+half_apple+half_pbj+half_turkey)
										#making sure that we have at least one half sandwitch of any kind
										if total_half>=1: 
											#making sure we can get the correct change
											if (max_cash-cost)%0.25 ==0: #making sure we can get the correct change
												change = max_cash - cost 
												sammie_change.append(change)
												sammie_combos.append((full_ham, half_ham, full_apple, half_apple, full_pbj, half_pbj, full_turkey, half_turkey))


	change_array = np.array(sammie_change)
	combos_array = np.array(sammie_combos)

	smallest_change_index = change_array.argmin()
	bestest_sammies_order = combos_array[smallest_change_index]

	return (
		'Change =',change_array.min(), 'Full Ham #',bestest_sammies_order[0], 
		'Half Ham #',bestest_sammies_order[1], 'Full Apple #',bestest_sammies_order[2], 
		'Half Apple #',bestest_sammies_order[3], 'Full PBJ #',bestest_sammies_order[4], 
		'Half PBJ #',bestest_sammies_order[5], 'Full Turkey #',bestest_sammies_order[6], 
		'Half Turkey #',bestest_sammies_order[7]
		)

						

	
print('Problem #2: Sammiez!', maximize_sammies(21.50))
print()




######################################################################################
'''PROBLEM THREE: PRIME NUMBERS'''

def primes(lower_value,upper_value):
	primelist = []
	
	for j in range(lower_value,upper_value+1):
		for i in range(2, int(j**0.5)+1):
			if j % i == 0:
				break
		else:
			primelist.append(j)
					

	return primelist

lower_value = 2
upper_value = 10000
print('Problem #3: Prime Numbers between', lower_value, 'and', upper_value, '=',(primes(lower_value,upper_value)))
print()

######################################################################################
'''PROBLEM FOUR: RECURSION'''

#Part A: Catalan Numbers 

def catalan(n):
	if n==0:
		cat_num = 1

	elif n>0: 
		cat_num = ((4*n)-2)/(n+1) *catalan(n-1)

	else:
		cat_num='Error!'

	return cat_num


print('Problem #4: Fibonacci Sequence',catalan(0), catalan(1), catalan(2), catalan(3), catalan(4), catalan(5), catalan(6), catalan(7), catalan(8), catalan(9))
print('Problem #4: Catalan 100',catalan(100))



#Catalan_100 = 8.965199470901317e+56
#The above function successfully returns the Catalan numbers, or in other words, the values of the Fibonacci sequenc. In order: 1, 1, 2, 5, 14, 42, 132, 429, 1430, etc. 


#Part B: Greatest Common Divisor

'''Writing a function that defines our greatest common denomiator'''

def gcd(m,n):

	m_factors = []
	n_factors = []
	common_factors = []

	for i in range(1,m+1):
		if m%i == 0: 
			m_factors.append(i)

	for j in range(1,n+1):
		if n%j == 0: 
			n_factors.append(j)

	common_factors = set(m_factors).intersection(set(n_factors))
	final_answer = max(common_factors)
	
	return final_answer

'''Applying the GCD function to our recursion'''


def euclid(m,n):

	if n==0:
		res = m

	elif n> 0:
		res = gcd(m,n)

	return res

print('Problem #4: GCD 108 & 192 =',euclid(108,192))

#The greatest common divisor of 108 and 192 is 12

######################################################################################
'''PROBLEM FIVE: FOUNDING COMPUTING FIGURE'''

#Done. Attached separately. 






