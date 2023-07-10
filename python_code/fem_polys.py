# Ben Weber, <completion date here>

# Module of classes and functions based on the polynomial module, with the
# aim of producing the standard basis functions in 1D for Lagrangian FEM and
# Hermite FEM. These can then be expanded to higher dimensions using hypercubic
# elements via tensor products.

# FOR LAGRANGIAN FEM:

# Classes and functions that generate GL points on the interval [-1, 1], and
# the Lagrangian polynomials given by these points. GL points are the points
# -1, 1, and the zeros of P'_m(x), where P_m is the mth Legendre polynomial

# FOR HERMITE FEM:

# Classes and functions that generate Hermite polynomials of degree
# (2q+1) on the interval [0,1], for use with the Hermite finite element method
# (HFEM)

# Two types of polynomial: type f and type g. For type f,
# d^(i)f_j(x=0) = diracdelta_ij, and d^(i)f_j(x=1) = 0. For type g,
# d^(i)g_(j)(x=0) = 0 and d^(i)g_(j)(x=1) = delta_ij. Note that indices i, j
# here run from 0 to q inclusive. NOTE: refer to f as left_type in code, and
# g as right_type, indices from 0 to q

# Type f can be generated from type g via f_i(1-x) = (-1)^i g_i(x). Type g can
# be generated by noting that g_i(x) = x^(q+1) h_i(x), where h_i is a degree q
# polynomial. The coefficients of h_i(x) = h_i0 + h_i1 x + h_i2 x^2 + ...
# + h_iq x^q satisfy MH = I, where I is the identity matrix, H is the matrix of
# coefficients such that H_ij = h_ij, and M_ij = (q+j+1)!/(q+j-i+1)!

#Import required modules

import math as m
import numpy as np
import matplotlib.pyplot as plt

#Import custom modules

import polynomial as p

#Define classes

class lagrange_even(p.polynomial):

	#Define Lagrange polynomials on the interval [-1,1] (as for the case 
	#with GL points) using evenly spaced points

	@staticmethod
	def __points(nn):
		"""Return a set of nn+1 points evenly distributed over [-1,1]"""

		if nn < 1:
			nnx = 1
		else:
			nnx = m.floor(nn)

		step = 2.0/nnx

		return list(np.arange(-1.0, 1.0 + step, step))

	def __init__(self, q, r, var = 0):
		"""Return a normalised Lagrange polynomial based on a set of q+1 points, centred on point r"""

		if q == 0:

			self.coeffs = np.ones((1))
			self.degrees = np.zeros((1))
			self.dim = 1
			self.var = [var]

		else:

			q_new = m.floor(q)
			r_new = m.floor(max(0, min(r, q_new + 0.9)))

			zeros = lagrange_even.__points(q_new)
			centre = zeros.pop(r)

			l_poly = p.poly_from_roots(zeros)
			norm = l_poly.evaluate(centre)

			self.coeffs = l_poly.coeffs * (1.0/norm)
			self.degrees = l_poly.degrees
			self.dim = l_poly.dim
			self.var = [var]



class lagrange(p.polynomial):

	#Add functions to generate GL points, making them private to avoid
	#overcomplicating end usage

	@staticmethod
	def __leg_coeffs(n):
		"""Return a numpy array of coefficients for use in calculating legendre polynomials"""

		low_lim_coeff = m.floor((n + 1)/2)
		coeffs = np.ones( (n + 1 - low_lim_coeff) )

		#Multiply each element to rescale it to the correct value
		kc = low_lim_coeff

		while kc <= n:

			coeffs[kc - low_lim_coeff] *= m.factorial(2 * kc)
			coeffs[kc - low_lim_coeff] /= m.factorial(kc)
			coeffs[kc - low_lim_coeff] /= m.factorial(n - kc)
			coeffs[kc - low_lim_coeff] /= m.factorial((2 * kc) - n)
			coeffs[kc - low_lim_coeff] /= 2 ** n

			if (n - kc)%2 == 1:
				coeffs[kc - low_lim_coeff] *= -1

			kc += 1

		return coeffs

	@staticmethod
	def __leg_poly(n, x):
		"""Return the result of the Legendre polynomial of order n at x"""

		low_lim = m.floor((n + 1)/2)
		coeffs = lagrange.__leg_coeffs(n)

		if n%2 == 0:
			x_cond = 1
		else:
			x_cond = x

		k = n
		result = 0

		while k >= low_lim:

			result *= m.pow(x, 2)
			result += x_cond * coeffs[-1 - n + k]
			k -= 1

		return result

	@staticmethod
	def __leg_dev_1(n, x):
		"""Return the result of the first derivative of a Legendre polynomial of order n at x"""

		if n == 0:

			result = 0

		else:

			low_lim_1 = 1 + m.floor(n/2)
			coeffs = lagrange.__leg_coeffs(n)

			if (n - 1)%2 == 0:
				x_cond = 1
			else:
				x_cond = x

			k = n
			result = 0

			while k >= low_lim_1:

				result *= m.pow(x, 2)
				result += x_cond * ((2 * k) - n) * coeffs[-1 - n + k]
				k -= 1

		return result

	@staticmethod
	def __leg_dev_2(n, x):
		"""Return the result of the second derivative of a Legendre polynomial of order n at x"""

		if n < 2:

			result = 0

		else:

			low_lim_2 = m.floor((n + 3)/2)
			coeffs = lagrange.__leg_coeffs(n)

			if n%2 == 0:
				x_cond = 1
			else:
				x_cond = x

			k = n
			result = 0

			while k >= low_lim_2:

				result *= m.pow(x, 2)
				result += x_cond * ((2 * k) - n) * ((2 * k) - n - 1) * coeffs[-1 -n + k]
				k -= 1

		return result

	@staticmethod
	def start_points(n, points, alpha = 0.3, dev_tol = m.pow(10,-2)):
		"""Create a list of starting values for NR iterations, based on list of previous GL points and using relative shift rate alpha"""

		#There are n-1 GL points excluding -1,1. If n-1 is even, half will lie in
		#the interval (-1,0), if not then 0 will also be a GL point and half of the
		#remaining will lie in this interval. The number of starting points should
		#be the same as the number of GL points in (-1,0)
		lgth = m.floor((n - 1)/2)
		points_new = np.zeros(lgth)

		#As n increases, expect relevant GL points to shift in the direction of -1,
		#making space for new points to form around x=0
		mm = 0

		while mm < len(points):

			#Approximate next GL points as lying in intervals between old GL points
			if mm == 0:
				a = -1
			else:
				a = points[mm - 1]

			b = points[mm]

			#GL points with larger intervals to left will move faster
			points_new[mm] = b - alpha * (b - a)

			#Ensure point is not in a region with 0 or small derivative
			while m.fabs(lagrange.__leg_dev_2(n, points_new[mm])) < dev_tol:

				points_new[mm] -= 0.1 * alpha * (b - a)

			mm += 1

		#If n is odd, then there will be a GL point between 0 and the last non-zero
		#GL point. Otherwise there will be no GL point in this region
		if n%2 == 1:

			if n == 3:
				a = -1
			else:
				a = points[-1]

			b = 0

			points_new[-1] = b - alpha * (b - a)

		return points_new

	@staticmethod
	def __NR_for_GL(n, start, tol = m.pow(10, -6)):
		"""Perform NR iterations to find GL points, based on starting points 'start' """

		#Loop over each starting point, continuing until it converges to a fixed
		#point with a tolerance given by tol
		p = len(start)
		mm = 0
		roots = np.zeros(p)
		fail_ind = 0

		while mm < p:

			#print("m = " + str(m))

			conv_ind = 0
			x_r = start[mm]

			#rep_count = no. NR iterations from a given starting point
			#maj_count = no. starting points attempted previously
			rep_count = 0
			maj_count = 0

			if mm == 0:
				low_lim = -1

			else:
				low_lim = start[mm - 1]

			while conv_ind == 0:

				rep_count += 1
				#print("rep_count = " + str(rep_count))

				#Check the derivative is non-zero
				while m.fabs(lagrange.__leg_dev_2(n, x_r)) <= m.pow(10,-2):

					x_r -= 0.1 * (start[mm] - low_lim)

				#NR iteration
				x_r_new = x_r - (lagrange.__leg_dev_1(n, x_r)/lagrange.__leg_dev_2(n, x_r))
				e = m.fabs(x_r_new - x_r)
				#print(e)

				#Check for a converged solution  
				if e < tol:

					roots[mm] = x_r_new
					conv_ind = 1

				#Check for a repeated failure to remain in the interval
				elif maj_count > 9:

					conv_ind = 2
					root[mm] = start[mm]

				#Change starting point to avoid exit of interval
				elif x_r_new < low_lim:

					if mm != p - 1:
						x_r += 0.2 * (start[mm + 1] - start[mm])
					else:
						x_r -= 0.2 * start[mm]
 
					maj_count += 1
					rep_count = 0

				#In case of a non-convergent sequence
				elif rep_count > pow(10, 3):

					if maj_count < 4:

						if mm == 0:
							x_r -= 0.1 * (1 + start[mm])
						else:
							x_r -= 0.1 * (start[mm] - start[mm - 1])

						maj_count += 1
						rep_count = 0

					else:
						#Record that root failed to converge
						conv_ind = 2 
						roots[mm] = start[mm]

				else:
					x_r = x_r_new

			if conv_ind == 2:

				fail_ind = 1

			mm += 1

		return roots, fail_ind
                             
	@staticmethod
	def GL_points(q):
		"""Return GL points corresponding to order q polynomials; alltsaa -1,1 and the roots of P'_q"""

		q_new = m.floor(m.fabs(q))

		if q_new > 40:

			print("Requested q is too high!")
			q_new = 40

		#variables to keep track of which set of GL points are being calculated
		legendre_count = 3

		#Initialise arrays to hold GL points. Only two sets of GL points are
		#needed at a time, the previous and current set
		GL_points_old = - 2 * np.ones( m.floor((q_new + 2)/2) )

		#Return GL points {-1} for n=1, and {-1,0} for n=2, omitting strictly
		#positive GL points
		if q_new == 0:

			return [-1]

		elif q_new == 1:

			return [-1,1]

		elif q_new == 2:

			return [-1,0,1]

		else:        

			GL_points_old[0] = -1
			GL_points_old[1] = 0

			l = 2

			#start recurrence of finding points for n based on n-1 at n=3

			while legendre_count <= q_new:

				#Derive a list of start-points from the previous entries in GL_points
				are_there_points = 1
				points_count = 1
				input_points = []

				while are_there_points == 1:

					x_curr = GL_points_old[points_count]

					if x_curr == -2:

						are_there_points = 0

					elif x_curr == 0:

						are_there_points = 0

					else:

						input_points.append(x_curr)
						points_count += 1

				#Use GL points in interval (-1,0) for n-1 to find starting points for
				#iteration
				s_points = lagrange.start_points(legendre_count, input_points)

				#Perform NR to find approximate values for GL points
				new_GL, flag = lagrange.__NR_for_GL(legendre_count, s_points)

				#Commands for how to proceed if the NR fails to converge
				if flag == 1:

					print("ERROR: NR failed to converge for one or more points at Legendre polynomial:", end = " ")
					print(legendre_count)

					break

				#Check for duplicate points (indicates poor choice of starting point)

				#To do so, note that the second derivative of the Legendre polynomial
				#must be of alternating signs at each of the new GL when taken in order. 
				#If two consecutive GL points (excluding -1) are found to have the same
				#sign, then it follows that one must be a duplicate
				l = len(new_GL)

				attempt_count = 0
				corr_ind = 0
				fail_ind = 0

				while corr_ind == 0:

					test_sign = (legendre_count - 1)%2
					test_sign = (2 * test_sign) - 1

					test_res = np.zeros(l)

					check_count = 0
             
					while check_count < l:

						x = new_GL[check_count]
     
						sign = lagrange.__leg_dev_2(legendre_count, x)
						sign /= m.fabs(sign)

						if sign != test_sign:

							#Duplicate detected!
							if check_count == 0:

								#1st point in list is actually 2nd or later GL root
								test_res[0] = 1

							elif m.fabs(x - new_GL[check_count - 1]) < m.pow(10, -6):

								test_res[check_count] = -1

							else:

								test_res[check_count] = 1

						check_count += 1
						test_sign *= -1

					if all(test_res == np.zeros(l)):

						corr_ind = 1

					elif attempt_count > 4:

						fail_ind = 1
						corr_ind = 1
         
					else:

						#If needed, Update starting points for better performance

						#The values in test_res indicate in which direction the GL root
						#has been erroneously deflected. Since this is due to the gradient
						#at some point being too low, resulting in an overshoot, increment
						#the starting point in the direction of the deflection. If this
						#cannot be achieved, set fail_ind to 1 to terminate the iterations.

						corr_count = 0

						while corr_count < l:

							if test_res[corr_count] == 0:

								corr_count += 1

							else:

								s_points[corr_count] -= 0.1 * (s_points[corr_count] - s_points[int(corr_count + test_res[corr_count])])
								corr_count += 1

						#Repeat iteration with updated starting points                 
						new_GL, flag = lagrange.__NR_for_GL(legendre_count, s_points)

						if flag == 1:

							fail_ind = 1
							corr_ind = 1

						else:

							attempt_count += 1

					#Repeat until the required number of unique GL points in the region
					#[floor((n-1)/2) ] is obtained

				if fail_ind == 1:

					#Corrective iterations failed to find the required roots
					print("ERROR: Unable to find roots for GL points at Legendre degree:", end = " ")
					print(legendre_count)

					break                   

				#Add GL point -1 (and 0 if required) to the list
				#new_GL.insert(0, -1)
				new_GL = np.insert(new_GL, 0, [-1])
                
				if legendre_count%2 == 0:
					new_GL = np.insert(new_GL, l + 1, 0)

				l = len(new_GL)

				#Record GL points, noted intervals and interval positions
				record_count = 0

				while record_count < l:

					GL_points_old[record_count] = new_GL[record_count]
					record_count += 1

				#Continue until predetermined maximum value of n (legendre_max), or until
				#finding required number of unique GL points becomes unfeasible (if so,
				#print error message stating this and the value of n it occurred for)
				legendre_count += 1

			GL_points = GL_points_old[:l]

			GL_points_out = np.zeros(q_new + 1)

			for GL_ind in range(0, l):

				GL_points_out[GL_ind] = GL_points[GL_ind]
				GL_points_out[q_new - GL_ind] = - GL_points[GL_ind]

			return list(GL_points_out)


	def __init__(self, q, r, var = 0, zeros_in = []):
		"""Generate the rth Lagrange polynomial of order q, for a qth order FEM method"""

		if q == 0:

			self.coeffs = np.ones((1))
			self.degrees = np.zeros((1))
			self.dim = 1
			self.var = [var]

		else:

			if len(zeros_in) == 0:
				zeros = lagrange.GL_points(q)
			else:
				zeros = zeros_in[:]

			no_zeros = len(zeros)

			r_new = max([0,min([r,no_zeros - 1])])
			r_new = int(r_new)
        
			centre = zeros.pop(r)

			l_poly = p.poly_from_roots(zeros)

			norm = l_poly.evaluate(centre)

			self.coeffs = l_poly.coeffs * (1.0/norm)
			self.degrees = l_poly.degrees
			self.dim = l_poly.dim
			self.var = [var]



def inner_product(p_1, p_2, fem_type = "lagrange"):
	"""Takes the inner product of p_1 and p_2 over the reference hypercube related to the FEM type in use"""

	if (fem_type == "hermite") or (fem_type == "hermite_dg"):

		a = 0
		b = 1

	else:

		a = -1
		b = 1

	try:

		p_int = p.poly_mult(p_1, p_2)

		dims = max(max(p_1.var), max(p_2.var))

		for var in range(0,dims + 1):

			p_int = p_int.integ(var)
        
			p_int_a = p_int.evaluate([a], [var])
			p_int_b = p_int.evaluate([b], [var])

			try:
				p_int_a.coeffs *= -1
				p_int = p.poly_add(p_int_a, p_int_b)

			except:
				p_int = p_int_b - p_int_a

		return p_int

	except:
		return (float(p_1) * float(p_2)) * (b - a)



class hermite(p.polynomial):

	@staticmethod
	def __building_poly(cf, index_1, index_2, var):
		"""Return the polynomial x^(index_1) (x - 1)^(index_2)"""

		arr_1 = np.zeros(index_1 + 1)
		arr_1[-1] = cf

		p_1 = p.polynomial(arr_1, [var])

		p_2 = p.polynomial([-1,1],[var])

		for count in range(0, index_2):

			p_1 = p.poly_mult(p_1, p_2)

		return p_1

	@staticmethod
	def hermite_gen_mat(q):
		"""Return a matrix used to determine the coefficients of the building polynomials in an expansion of a hermite polynomial"""

		q_new = m.floor(abs(q))

		matrix = np.zeros( (q_new + 1, q_new + 1) )

	        #Entry k,j should be the kth derivative of building polynomial q+1,j (x^(q+1) (x - 1)^j at x = 1
		#NOTE: This matrix is used to derive the coefficients of g, from which those of f can be derived

		#Try preconditioning this matrix to allow resulting rth Hermite polynomials to have derivative r!
		#at end point, rather than 1 (divide column r by r!, r goes from 0 to q

		for j in range(0, q_new + 1):
			for k in range(j, q_new + 1):

#				matrix[j,k] = np.float64( m.factorial(k) * m.factorial(q_new + 1) ) / np.float64( m.factorial(k - j) * m.factorial(q_new + j - k + 1) )
				matrix[j,k] = np.float64( m.factorial(q_new + 1) ) / np.float64( m.factorial(k - j) * m.factorial(q_new + j - k + 1) )

		#Use Gaussian elimination to form the inverse of matrix, matrix is upper triangular

		mat_out = np.zeros( (q_new + 1, q_new + 1) )

		for j in range(0, q_new + 1):

			col = np.float64(1) #/ matrix[ q_new - j, q_new - j ]

			for i in range(1, q_new - j + 1):

				row = matrix[ q_new - i - j, q_new - i - j + 1:q_new - j + 1]

				if i == 1:
					res = - col * row
				else:
					res = - row @ col

				#res /= matrix[ q_new - i - j, q_new - i - j ]

				if i == 1:
					col = np.array( [[res], [col]] )
				else:
					temp = np.zeros( ( i + 1, 1) )
					temp[0,0] = res[0]
					temp[1:,0] = col[:,0]

					col = temp[:,:]

			if j != q_new:
				mat_out[ :(q_new - j + 1), q_new - j] = col[:,0]
			else:
				mat_out[0, 0] = col

#		return np.linalg.inv(matrix)
		return mat_out


	def __init__(self, q, r, side = 0, var = 0, mat_herm_init = np.zeros( (1,1) ) ):
		"""Return the degree q Hermite polynomial on [0,1] f, with f^(r)(side) = 1"""
		
		if not mat_herm_init.any():
			coeff_mat = hermite.hermite_gen_mat(q)
		else:
			#Problem is here, the loaded, pre-calculated value of mat_herm_init is different to one calculated on the spot
			#Seems to occur at random, involves entries flickering in sign
			coeff_mat = mat_herm_init[:][:]
        
		r_new = int(max([0, min([q, r])]))

		coeffs = np.zeros( (1, q + 1) )
		coeffs[0,:] = coeff_mat[r_new,:]
        
		if not bool(side):
			for j in range(0, q + 1):

				coeffs[0,j] *= (1 - 2 * ((j + r_new + q + 1)%2))

		p_out = p.polynomial([0],[var])

		#Note that coeff_mat is upper triangular. This means the first r_new coefficients
		#will be 0

		for index in range(r_new, q + 1):
			if bool(side):

				p_out = p.poly_add(p_out, hermite.__building_poly(coeffs[0,index], q + 1, index, var))

			else:

				p_out = p.poly_add(p_out, hermite.__building_poly(coeffs[0,index], index, q + 1, var))


		self.coeffs = p_out.coeffs
		self.degrees = p_out.degrees
		self.dim = p_out.dim
		self.var = p_out.var
        
        
# Define useful functions using the above classes

def array_check( input_array, q, fem_type = "lagrange" ):
	"""Tests whether an input array is valid for a given finite element method, returning a correct array if not"""

	if (fem_type == "hermite") or (fem_type == "hermite_dg"):
		#Should be a numpy array of size (q+1)*(q+1)

		try:
			input_array = np.array( input_array )

			test_shape = np.shape( input_array )
			corr_shape = ( q + 1, q + 1 )

			if (test_shape[0] == corr_shape[0]) and (test_shape[1] == corr_shape[1]):
				check_ind = True
			else:
				check_ind = False

		except:

			check_ind = False

		if check_ind:
			out_array = np.zeros( corr_shape )
			out_array[:,:] = input_array[:,:]

			return out_array
		else:
			print("ERROR: input array is not valid for Hermite FEM. Returning default Hermite matrix")
			return hermite.hermite_gen_mat(q)

	elif (fem_type == "lagrange") or (fem_type == "lagrange_dg"):
		#Should be a list of zeros for a polynomial, of type int or float

		if type(input_array) is not list:
			check_ind = False
			print("TYPE 0")
		else:
			check_ind = True

			for test_el in input_array:

				if (type(test_el) is int) or (type(test_el) is float) or (type(test_el) is np.float64):
					continue
				else:
					check_ind = False
					print(type(test_el))
					break

		if check_ind:
			return input_array[:]
		elif q == 0:
			return None
		else:
			print("ERROR: input array is not valid for Lagrange FEM. Returning GL points")
			return lagrange.GL_points(q)

	else:
		return None


def cell_funct(q = 1, indices = [0], fem_type = "lagrange_even", input_array = None):
	"""Returns the basis function referred to by indices for an FEM method of order on the reference cell"""

	if len(indices) == 0:

		indices = [0]

	for i in range(0, len(indices)):

		if (fem_type == "hermite") or (fem_type == "hermite_dg"):
			#q is the maximum derivative that is matched at the boundaries.
			#Note that q=0 is equivalent to the Lagrangian method with q=1
			indices[i] = max(0, min(2 * q + 1, indices[i]))

		else:
			#q is the degree of the polynomials generated
			indices[i] = max(0, min(q, indices[i]))


	poly_out = p.polynomial([1],[0])
    
	if (fem_type == "hermite") or (fem_type == "hermite_dg"):

		#Output a tensor product of Hermite polynomials, given in each variable by indices[var]
		herm_mat = array_check( input_array, q, "hermite" )

		for i in range(0, len(indices)):

			r = indices[i] % (q + 1)
			s = (indices[i] - r)//(q + 1)

			poly_out = p.poly_mult(poly_out, hermite(q, r, s, i, herm_mat[:,:]))


	elif (fem_type == "lagrange") or (fem_type == "lagrange_dg"):

		#Output a tensor product of Lagrangian polynomials
		lag_pts = array_check( input_array, q, "lagrange" )

		for i in range(0, len(indices)):

			j = indices[i]

			poly_out = p.poly_mult(poly_out, lagrange(q, j, i, lag_pts[:]))

	else:

		#Output a tensor product of Lagrangian polynomials
		for i in range(0, len(indices)):

			j = indices[i]

			poly_out = p.poly_mult(poly_out, lagrange_even(q, j, i))


	return poly_out

    




def inner_product_funct(f, g, fem_type = "lagrange", no_points = 21):
	"""Approximates the inner product of general functions f and g, over the interval specified by fem_type. f, g are assumed to be integrable"""

	if (fem_type == "hermite") or (fem_type == "hermite_dg"):
		a = 0
		b = 1
	else:
		a = -1
		b = 1

	h = (b - a)/(no_points - 1.0)
	points = np.linspace(a, b, no_points)

	#Use trapezoidal quadrature rule
	result = - 0.5 * h * f(a) * g(a)
	result -= 0.5 * h * f(b) * g(b)

	for x in points:

		result += h * f(x) * g(x)

	return result


def mass_1(q, fem_type = "lagrange_even", input_array = None ):
	"""Defines the diagonal block for the mass matrix of 1D Lagrange FEM"""
	#Interpolation works as well for H-DG as L-DG methods, check for error in Laplace matrices
	if fem_type == "hermite":

		herm_mat = array_check( input_array, q, "hermite" )
		mat_out = np.zeros( (q + 1, q + 1) )

		for i in range(0, q + 1):

			p_i = hermite( q, i, 0, 0, herm_mat[:,:] )
			q_i = hermite( q, i, 1, 0, herm_mat[:,:] )

			for j in range(i, q + 1):

				p_j = hermite( q, j, 0, 0, herm_mat[:,:] )
				mat_out[i,j] = inner_product(p_i, p_j, fem_type = "hermite")

				p_j = hermite( q, j, 1, 0, herm_mat[:,:] )
				mat_out[i,j] += inner_product(q_i, p_j, fem_type = "hermite")

				mat_out[j,i] = mat_out[i,j]


	elif fem_type == "lagrange":

		lag_pts = array_check( input_array, q, "lagrange" )
		mat_out = np.zeros( (q, q) )

		for i in range(0,q):

			p_i = lagrange( q, i, 0, lag_pts[:] )

			mat_out[i,i] = inner_product( p_i, p_i )

			for j in range(i + 1, q):

				p_j = lagrange( q, j, 0, lag_pts[:] )

				mat_out[i,j] = inner_product( p_i, p_j )
				mat_out[j,i] = mat_out[i,j]

		p_i = lagrange( q, q, 0, lag_pts[:] )
		mat_out[0,0] += inner_product( p_i, p_i )


	elif fem_type == "hermite_dg":

		herm_mat = array_check( input_array, q, "hermite" )
		mat_out = np.zeros( (2 * (q + 1), 2 * ( q + 1 )) )

		for index_1 in range(0, 2 * (q + 1)):

			end_ind_1 = int( index_1 > q )
			i = index_1 - end_ind_1 * (q + 1)

			p_i = hermite( q, i, end_ind_1, 0, herm_mat )

			mat_out[index_1,index_1] = inner_product( p_i, p_i, fem_type = "hermite_dg" )

			for index_2 in range(index_1 + 1, 2 * (q + 1)):

				end_ind_2 = int( index_2 > q )
				j = index_2 - end_ind_2 * (q + 1)

				p_j = hermite( q, j, end_ind_2, 0, herm_mat )

				mat_out[index_1, index_2] = inner_product( p_i, p_j, fem_type = "hermite_dg" )
				mat_out[index_2, index_1] = mat_out[index_1, index_2]


	elif fem_type == "lagrange_dg":

		lag_pts = array_check( input_array, q, "lagrange" )
		mat_out = np.zeros( (q + 1, q + 1) )

		for i in range(0, q + 1):

			p_i = lagrange( q, i, 0, lag_pts[:] )

			mat_out[i,i] = inner_product( p_i, p_i, fem_type = "lagrange_dg" )

			for j in range(i + 1, q + 1):

				p_j = lagrange( q, j, 0, lag_pts[:] )

				mat_out[i,j] = inner_product( p_i, p_j, fem_type = "lagrange_dg" )
				mat_out[j,i] = mat_out[i,j]

	elif fem_type == "lagrange_even_dg":

		mat_out = np.zeros( (q + 1, q + 1) )

		for i in range(0, q + 1):

			p_i = lagrange_even( q, i )

			mat_out[i,i] = inner_product( p_i, p_i, fem_type = "lagrange_even_dg" )

			for j in range(i + 1, q + 1):

				p_j = lagrange_even( q, j )

				mat_out[i,j] = inner_product( p_i, p_j, fem_type = "lagrange_even_dg" )
				mat_out[j,i] = mat_out[i,j]

	else:

		mat_out = np.zeros( (q, q) )

		for i in range(0,q):

			p_i = lagrange_even( q, i )

			mat_out[i,i] = inner_product( p_i, p_i )

			for j in range(i + 1, q):

				p_j = lagrange_even( q, j )

				mat_out[i,j] = inner_product( p_i, p_j )
				mat_out[j,i] = mat_out[i,j]


		p_i = lagrange_even( q, q )
		mat_out[0,0] += inner_product( p_i, p_i )

	return mat_out


def mass_2(q, fem_type = "lagrange_even", input_array = None ):
	"""Define the upper diagonal block of a mass matrix for a given FEM type"""

	if fem_type == "hermite":

		herm_mat = array_check( input_array, q, "hermite" )
		mat_out = np.zeros( (q + 1, q + 1) )

		for i in range(0, q + 1):

			f_i = hermite( q, i, 0, 0, herm_mat[:,:] )

			for j in range(0, q + 1):

				g_j = hermite( q, j, 1, 0, herm_mat[:,:] )
				mat_out[i,j] = inner_product(f_i, g_j, fem_type = "hermite")



	elif fem_type == "lagrange":

		lag_pts = array_check( input_array, q, "lagrange" )

		mat_out = np.zeros( (q, q) )
		p_p = lagrange( q, q, 0, lag_pts[:] )

		for i in range(0, q):

			p_i = lagrange( q, i, 0, lag_pts[:] )
			mat_out[i,0] = inner_product( p_i, p_p )

	elif fem_type == "hermite_dg":

		mat_out = np.zeros( ( 2 * (q + 1), 2 * (q + 1) ) )


	elif (fem_type == "lagrange_dg") or (fem_type == "lagrange_even_dg"):

		mat_out = np.zeros( (q + 1, q + 1) )

	else:

		mat_out = np.zeros( (q, q) )
		p_p = lagrange_even( q, q )

		for i in range(0, q):

			p_i = lagrange_even( q, i )
			mat_out[i,0] = inner_product( p_i, p_p )

	return mat_out


def laplace_1(q, fem_type = "lagrange_even", input_array = (None), beta = 16.0 ):
	"""Define the diagonal block of the FEM matrix representing the Laplacian"""

	if fem_type == "hermite":

		herm_mat = array_check( input_array, q, "hermite" )
		mat_out = np.zeros( (q + 1, q + 1) )

		for i in range(0, q + 1):

			p_i = hermite( q, i, 0, 0, herm_mat[:,:] ).diff()
			q_i = hermite( q, i, 1, 0, herm_mat[:,:] ).diff()

			for j in range(i, q + 1):

				p_j = hermite( q, j, 0, 0, herm_mat[:,:] ).diff()
				mat_out[i,j] = - inner_product( p_i, p_j, fem_type = "hermite" )

				p_j = hermite( q, j, 1, 0, herm_mat[:,:] ).diff()
				mat_out[i,j] -= inner_product( q_i, p_j, fem_type = "hermite" )

				mat_out[j,i] = mat_out[i,j]


	elif fem_type == "lagrange":

		lag_pts = array_check( input_array, q, "lagrange" )
		mat_out = np.zeros( (q, q) )

		for i in range(0, q):

			p_i = lagrange( q, i, 0, lag_pts[:] ).diff()

			for j in range(i, q):

				p_j = lagrange( q, j, 0, lag_pts[:] ).diff()
				mat_out[i,j] = - inner_product( p_i, p_j )

				mat_out[j,i] = mat_out[i,j]

		p_i = lagrange( q, q, 0, lag_pts[:] ).diff()
		mat_out[0,0] -= inner_product( p_i, p_i )


	elif fem_type == "hermite_dg":

		herm_mat = array_check( input_array, q, "hermite" )
		mat_out = np.zeros( ( 2 * (q + 1), 2 * (q + 1) ) )

		mat_out[0,0] -= beta
		mat_out[q + 1, q + 1] -= beta

#				endpt_1 = float(p_i.evaluate(0)) #Only non-zero if i = 1, end_ind_1 = 0 => index_1 = 1, then equal to 1
#				endpt_2 = float(p_i.evaluate(1)) #Only non-zero if i = 1, end_ind_1 = 1 => index_1 = q + 2, then equal 1

		if q != 0:

			mat_out[0, 1] += 0.5 # average of endpt_1 and 0 when index_1=1
			mat_out[1, 0] += 0.5 # average of endpt_1 and 0 when above

			mat_out[q + 2, q + 1] -= 0.5 # average of endpt_2 and 0
			mat_out[q + 1, q + 2] -= 0.5 # average of endpt_2 and 0

		else:

			#endpt_1 = endpt_2 = -1 if index_1 = 0, endpt_1 = endpt_2 = 1 if index_1 = 1, average gradients
			mat_out[0, 0] -= 1
			mat_out[0, 1] += 1
			mat_out[1, 0] += 1
			mat_out[1, 1] -= 1

		for index_1 in range(0, 2 * (q + 1)):

			end_ind_1 = int( index_1 > q )
			i = index_1 - end_ind_1 * (q + 1)

			p_i = hermite( q, i, end_ind_1, 0, herm_mat ).diff()

			mat_out[index_1, index_1] -= inner_product( p_i, p_i, fem_type = "hermite" )

			for index_2 in range(index_1 + 1, 2 * (q + 1)):

				end_ind_2 = int( index_2 > q )
				j = index_2 - end_ind_2 * (q + 1)

				p_j = hermite( q, j, end_ind_2, 0, herm_mat ).diff()
						
				mat_out[index_1, index_2] -= inner_product( p_i, p_j, fem_type = "hermite" )
				mat_out[index_2, index_1] = mat_out[index_1, index_2]


	elif fem_type == "lagrange_dg":

		lag_pts = array_check( input_array, q, "lagrange" )
		mat_out = np.zeros( (q + 1, q + 1) )

		mat_out[0,0] -= 0.5 * beta
		mat_out[q,q] -= 0.5 * beta

		for i in range(0, q + 1):

			p_i = lagrange( q, i, 0, lag_pts[:] ).diff()

			mat_out[i,i] -= inner_product( p_i, p_i, fem_type = "lagrange_dg" )

			endpt_1 = float(p_i.evaluate(-1))
			endpt_2 = float(p_i.evaluate(1))

			mat_out[0,i] += 0.5 * endpt_1
			mat_out[i,0] += 0.5 * endpt_1

#			mat_out[i,q] += 0.5 * endpt_2
#			mat_out[q,i] += 0.5 * endpt_2
			mat_out[i,q] -= 0.5 * endpt_2
			mat_out[q,i] -= 0.5 * endpt_2

			for j in range(i + 1, q + 1):

				p_j = lagrange( q, j, 0, lag_pts[:] ).diff()

				mat_out[i,j] -= inner_product( p_i, p_j, fem_type = "lagrange_dg" )
				mat_out[j,i] = mat_out[i,j]


	elif fem_type == "lagrange_even_dg":

		mat_out = np.zeros( (q + 1, q + 1) )

		mat_out[0,0] -= 0.5 * beta
		mat_out[q,q] -= 0.5 * beta

		for i in range(0, q + 1):

			p_i = lagrange_even( q, i ).diff()

			mat_out[i,i] -= inner_product( p_i, p_i, fem_type = "lagrange_even_dg" )

			endpt_1 = float(p_i.evaluate(-1))
			endpt_2 = float(p_i.evaluate(1))

			mat_out[0,i] += 0.5 * endpt_1	#Because average gradients
			mat_out[i,0] += 0.5 * endpt_1

			mat_out[i,q] -= 0.5 * endpt_2
			mat_out[q,i] -= 0.5 * endpt_2

			for j in range(i + 1, q + 1):

				p_j = lagrange_even( q, j ).diff()

				mat_out[i,j] -= inner_product( p_i, p_j, fem_type = "lagrange_even_dg" )
				mat_out[j,i] = mat_out[i,j]


	else:

		mat_out = np.zeros( (q, q) )

		for i in range(0, q):

			p_i = lagrange_even( q, i ).diff()

			for j in range(i, q):

				p_j = lagrange_even( q, j ).diff()
				mat_out[i,j] = - inner_product( p_i, p_j )

				mat_out[j,i] = mat_out[i,j]

		p_i = lagrange_even( q, q ).diff()
		mat_out[0,0] -= inner_product( p_i, p_i )

	return mat_out


def laplace_2(q, fem_type = "lagrange_even", input_array = (None), beta = 16.0 ):
	"""Return the upper diagonal block matrix from the FEM matrix representing the Laplacian"""

	if fem_type == "hermite":

		herm_mat = array_check( input_array, q, "hermite" )
		mat_out = np.zeros( (q + 1, q + 1) )

		for i in range(0, q + 1):

			g_i = hermite( q, i, 1, 0, herm_mat[:,:] ).diff()

			for j in range(0, q + 1):

				f_j = hermite( q, j, 0, 0, herm_mat[:,:] ).diff()
				mat_out[j,i] = - inner_product( f_j, g_i, fem_type = "hermite" )


	elif fem_type == "lagrange":

		lag_pts = array_check( input_array, q, "lagrange" )

		mat_out = np.zeros( (q, q) )
		p_p = lagrange( q, q, 0, lag_pts[:] ).diff()

		for i in range(0, q):

			p_i = lagrange( q, i, 0, lag_pts[:] ).diff()
			mat_out[i,0] = - inner_product( p_i, p_p )


	elif fem_type == "hermite_dg":	#AVERAGE GRADIENT!!!

		mat_out = np.zeros( ( 2 * (q + 1), 2 * (q + 1) ) )

		if q != 0:

			mat_out[q + 2, 0] += 0.5
			mat_out[q + 1, 1] -= 0.5

		else:

			mat_out[0, 0] -= 0.5
			mat_out[1, 0] += 1
			mat_out[1, 1] -= 0.5

		mat_out[q + 1, 0] += beta


	elif fem_type == "lagrange_dg":

		lag_pts = array_check( input_array, q, "lagrange" )
		mat_out = np.zeros( (q + 1, q + 1) )

		for ip in range(0, q + 1):

			p_curr = lagrange( q, ip, 0, lag_pts[:] ).diff()

			endpt_1 = float(p_curr.evaluate(-1))
			endpt_2 = float(p_curr.evaluate(1))

#			mat_out[ip,0] -= 0.5 * endpt_2
#			mat_out[q,ip] += 0.5 * endpt_1
			mat_out[ip,0] += 0.5 * endpt_2
			mat_out[q,ip] -= 0.5 * endpt_1

		mat_out[q,0] += 0.5 * beta


	elif fem_type == "lagrange_even_dg":

		mat_out = np.zeros( (q + 1, q + 1) )

		for ip in range(0, q + 1):

			p_curr = lagrange_even( q, ip ).diff()

			endpt_1 = float(p_curr.evaluate(-1))
			endpt_2 = float(p_curr.evaluate(1))

			mat_out[ip,0] += 0.5 * endpt_2
			mat_out[q,ip] -= 0.5 * endpt_1

		mat_out[q,0] += 0.5 * beta


	else:

		mat_out = np.zeros( (q, q) )
		p_p = lagrange_even( q, q ).diff()

		for i in range(0, q):

			p_i = lagrange_even( q, i ).diff()
			mat_out[i,0] = - inner_product( p_i, p_p )

	return mat_out
    

#MAIN:

if __name__ == "__main__":

    #This is intended as a module, not a program.
    print( "ERROR: fem_polys.py is a module not a stand-alone program." )

    pp = 1
    b = 2 * pp * (pp + 1)
    array_1 = array_check( pp, "lagrange", [] )

    print( mass_1( pp, "lagrange_dg", array_1 ) )
    print( mass_1( pp, "lagrange", array_1 ) )
    print( laplace_1( pp, "lagrange_dg", array_1, b ) )
    print( laplace_2( pp, "lagrange_dg", array_1, b ) )
#END 
