# Ben Weber 2021-01-27
# Code to determine least-squares fit for observed data, assuming 1/C_eff has a linear trend

import math as m
import numpy as np

FILE_NAME_FEM = "tstep_max.csv"
FILE_NAME_DG = "tstep_dg_max.csv"
DEGREE_MAX = 18

# Use the Lagrange-GL data for continuous Lagrange and DG, this is best behaved data
# Function to read results into three different arrays for performing regression on
def file_read():

	lagr_results = np.zeros( (18, 2) )
	herm_results = np.zeros( (7, 2) )
	dg_results = np.zeros( (18, 2) )
	
	file_cont = open( FILE_NAME_FEM, "r" )
	file_cont.readline()
	
	for i in range(0, DEGREE_MAX):
	
		str_curr = file_cont.readline()
		
		deg_str = ""
		gl_str = ""
		herm_str = ""
		
		comma_count = 0
		
		for ch in str_curr:
		
			if ch == ",":
				comma_count += 1
				continue	
			elif ch == "\n":
				break
				
			if comma_count == 0:
				deg_str = deg_str + ch
			elif comma_count == 2:
				gl_str = gl_str + ch
			elif comma_count == 3:
				herm_str = herm_str + ch
				
		degree = int(deg_str)
		
		if degree - 1 != i:
			print("Error: polynomial degree read from file " + FILE_NAME_FEM + " incorrect.\n")
			file_cont.close()
			exit(1)
			
		gl_val = 1.0 / float(gl_str)
		lagr_results[i,:] = np.array( (degree, gl_val) )
		
		if (degree % 2 == 1) and (degree < 14):
			herm_val = 1.0 / float(herm_str)
			herm_results[int(0.5 * i),:] = np.array( (degree, herm_val) )
			
	file_cont.close()
	
	file_dg = open(FILE_NAME_DG, "r")
	file_dg.readline()
	
	for i in range(0, DEGREE_MAX):
	
		str_curr = file_dg.readline()
		
		deg_str = ""
		dg_str = ""
		comma_count = 0
		
		for ch in str_curr:
		
			if ch == ",":
				comma_count += 1
				continue
			elif ch == "\n":
				break
				
			if comma_count == 0:
				deg_str = deg_str + ch
			elif comma_count == 2:
				dg_str = dg_str + ch
				
		degree = int(deg_str)
		
		if degree - 1 != i:
			print("Error: polynomial degree in file " + FILE_NAME_DG + " incorrect.\n")
			file_dg.close()
			exit(1)
			
		dg_val = 1.0 / float(dg_str)
		dg_results[i,:] = np.array( (degree, dg_val) )
		
	file_dg.close()
	
	return lagr_results, herm_results, dg_results
	

def least_squares_fit( array ):
	"""Takes an array of shape (N,2) and finds a straight line of best fit"""
	
	size = np.shape(array)
	
	if (size[1] != 2) or (len(size) != 2):
		print("Error: array passed to regression function has incorrect shape.")
		exit(1)
		
	N = size[0]
	
	sum_x = 0
	sum_xx = 0
	sum_y = 0
	sum_xy = 0
	
	for i in range(0, N):
	
		x = array[i,0]
		y = array[i,1]
		
		sum_x += x
		sum_xx += x * x
		sum_y += y
		sum_xy += x * y
		
	A = np.zeros( (2,2) )
	b = np.zeros( (2) )
	
	A[0,0] = sum_xx
	A[1,0] = A[0,1] = sum_x
	A[1,1] = N
	
	b[0] = sum_xy
	b[1] = sum_y
	
	return np.linalg.inv(A) @ b
	
	
if __name__ == "__main__":

	print("\nReading data from CSR files...")
	lagr_data, herm_data, dg_data = file_read()
	
	print("Calculating least squares")
	lagr_fit = least_squares_fit(lagr_data)
	herm_fit = least_squares_fit(herm_data)
	dg_fit = least_squares_fit(dg_data)
	
	print("\nData as follows:\n")
	
	print("Lagrange:")
	print(lagr_data)
	
	print("\nHermite:")
	print(herm_data)
	
	print("\nDG:")
	print(dg_data)
	
	print("\nPrinting lines of best fit:\n")
	
	print("\nLagrange fit: 1/C_eff = " + str(round(lagr_fit[0], 4)) + "p + " + str(round(lagr_fit[1], 4))) 
	print("Hermite fit: 1/C_eff = " + str(round(herm_fit[0], 4)) + "p + " + str(round(herm_fit[1], 4))) 
	print("DG fit: 1/C_eff = " + str(round(dg_fit[0], 4)) + "p + " + str(round(dg_fit[1], 4)) + "\n")

