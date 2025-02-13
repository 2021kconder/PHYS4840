import numpy as np

def my_function(vector): 
	a = vector[0]
	b = vector[1]
	c = vector[2]

	return np.linalg.norm(vector)


def single_cookie(price): 
    sugar_price     = 2.65
    chocolate_price = 3.20
    snicker_price   = 3.45
    smores_price    = 3.70
    
    price_list = [sugar_price,chocolate_price,snicker_price,smores_price]
    cookie_name_list = ['Sugar Cookie', 'Chocolate Cookie', 'Snickerdoodle Cookie', 'Smores Cookie']
    
    max_cookie_num = []
    lowest_change  = []
    
    for i in price_list: 
        max_cookie_num.append(price//i)
        lowest_change.append(price%i)
        
        lowest_change_value = min(lowest_change)
        lowest_change_index = lowest_change.index(lowest_change_value)
        lowest_change_cookie = cookie_name_list[lowest_change_index]
        lowest_change_c_num = max_cookie_num[lowest_change_index]
    
        
    return ('Leftover Change =',round(lowest_change_value,3), 'Cookie Type =',lowest_change_cookie, 'Number of Cookies=',lowest_change_c_num)

def square_root(value):
	if value >=0: 
		sqrt = value**(1/2)
	else:
		sqrt='Error! Value is a negative number.'

	return sqrt


def floor(value):
	if type(value)==float: 
		floor_value = int(value)

	elif type(value)==int:
		floor_value = 'Input is already an Integer!'

	elif type(value)==complex:
		floor_value= 'Error! Input is complex!'

	return floor_value

def y(x):
 	y = 2.0*x**3.0
 	return y


def distance_modulus(distance):
	mu = 5*np.log10(distance/10)
	return mu 
	



