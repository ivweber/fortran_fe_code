# Ivy Weber
# Set up initial conditions for the 2D and 3D solver, then save to a .npy file

# Use a gaussian bell-shape within a certain radius as initial conditions, set
# initial time derivative to 0

#Import required modules

import math as m
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lu

from mpl_toolkits import mplot3d
import random as rn
import matplotlib.pyplot as plt

#Import any custom modules

import polynomial as p
import fem_polys as fem

#Define functions and classes

import solver_const as sol

INITIAL_RADIUS = 1.0

quad_pts = sol.quad_pts
if sol.syst_dims == 2:
	dims = (sol.M, sol.N)
else:
	dims = (sol.M, sol.N, sol.O)

def exact_1(x):

	r_2 = x**2

	if r_2 < INITIAL_RADIUS:
		return m.exp( -10 * r_2 )
	else:
		return 0


def x_cell(x, n, dim_ind, fem_type = "lagrange"):

	if (fem_type == "hermite") or (fem_type == "hermite_dg"):
		a = 0
	else:
		a = -1
	b = 1

	x_h = (x - a) / (b - a)
	x_h = (x_h + n) * sol.HERM_WGT[dim_ind] + sol.MINS[dim_ind]

	return x_h

# Modify this function to use the faster tensor product method of 3d
def fem_interp_2d(fem_type, input_array = []):
	"""Return a vector of FEM coefficients representing an approximation of funct in the specified FEM space"""

	if (fem_type == "hermite") or (fem_type == "hermite_dg"):
		sz_1 = int( (sol.p - 1)/2 )
		wgt_2 = sol.HERM_WGT
	else:
		sz_1 = sol.p
		wgt_2 = sol.LAGR_WGT
		
	if (fem_type == "lagrange_even") or (fem_type == "lagrange"):
		sz_2 = sz_1
	elif fem_type == "hermite_dg":
		sz_2 = 2 * sz_1 + 2
	else:
		sz_2 = sz_1 + 1

	array = fem.array_check( input_array, sz_1, fem_type )

	#Interpolate the system with M X = (f, phi), where phi is the basis function corresponding to a given row
	#of M
	b = np.zeros( (max(dims) * sz_2, 2) )

	if fem_type == "hermite":

		for ip in range(0, sz_2):

			pq_1 = fem.hermite(sz_1, ip, 0, 0, array)
			pq_2 = fem.hermite(sz_1, ip, 1, 0, array)

			for nm in range(0, dims[0]):

				xcell_next = (nm + 1) % dims[0]

				funct = lambda x: exact_1(x_cell( x, nm, 0, fem_type) / sol.init_stretch[0] )

				ind_here = ip + nm * sz_2
				ind_next = ip + xcell_next * sz_2

				p_curr = lambda x: pq_1.evaluate(x)
				b[ind_here, 0] += fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

				p_curr = lambda x: pq_2.evaluate(x)
				b[ind_next, 0] += fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

			for nn in range(0, dims[1]):

				ycell_next = (nn + 1) % dims[1]

				funct = lambda y: exact_1(x_cell( y, nn, 1, fem_type) / sol.init_stretch[1] )

				ind_here = ip + nn * sz_2
				ind_next = ip + ycell_next * sz_2

				p_curr = lambda y: pq_1.evaluate(y)
				b[ind_here, 1] += fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

				p_curr = lambda y: pq_2.evaluate(y)
				b[ind_next, 1] += fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

	elif fem_type == "lagrange":

		pq_2 = fem.lagrange(sz_1, sz_2, 0, array[:])

		for nm in range(0, dims[0]):

			xcell_next = (nm + 1) % dims[0]

			funct = lambda x: exact_1(x_cell( x, nm, 0, fem_type) / sol.init_stretch[0] )
			p_curr = lambda x: pq_2.evaluate(x)

			ind_next = xcell_next * sz_2
			b[ind_next, 0] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

			for ip in range(0, sz_2):

				pq_1 = fem.lagrange(sz_1, ip, 0, array[:])
				p_curr = lambda x: pq_1.evaluate(x)

				ind_here = ip + nm * sz_2
				b[ind_here, 0] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

		for nn in range(0, dims[1]):

			ycell_next = (nn + 1) % dims[1]

			funct = lambda y: exact_1(x_cell( y, nn, 1, fem_type) / sol.init_stretch[1] )
			p_curr = lambda y: pq_2.evaluate(y)

			ind_next = ycell_next * sz_2
			b[ind_next, 1] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

			for jp in range(0, sz_2):

				pq_1 = fem.lagrange(sz_1, jp, 0, array[:])
				p_curr = lambda y: pq_1.evaluate(y)

				ind_here = jp + nn * sz_2
				b[ind_here, 1] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

	elif fem_type == "lagrange_even":

		pq_2 = fem.lagrange_even(sz_1, sz_2, 0)

		for nm in range(0, dims[0]):

			xcell_next = (nm + 1) % dims[0]

			funct = lambda x: exact_1(x_cell( x, nm, 0, fem_type) / sol.init_stretch[0] )
			p_curr = lambda x: pq_2.evaluate(x)

			ind_next = xcell_next * sz_2
			b[ind_next, 0] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

			for ip in range(0, sz_2):

				pq_1 = fem.lagrange_even(sz_1, ip, 0)
				p_curr = lambda x: pq_1.evaluate(x)

				ind_here = ip + nm * sz_2
				b[ind_here, 0] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

		for nn in range(0, dims[1]):

			ycell_next = (nn + 1) % dims[1]

			funct = lambda y: exact_1(x_cell( y, nn, 1, fem_type) / sol.init_stretch[1] )
			p_curr = lambda y: pq_2.evaluate(y)

			ind_next = ycell_next * sz_2
			b[ind_next, 1] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

			for jp in range(0, sz_2):

				pq_1 = fem.lagrange_even(sz_1, jp, 0)
				p_curr = lambda y: pq_1.evaluate(y)

				ind_here = jp + nn * sz_2
				b[ind_here, 1] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

	elif fem_type == "hermite_dg":

		half = int(0.5 * sz_2)

		for ip in range(0, half):

			pq_1 = fem.hermite(sz_1, ip, 0, 0, array)
			pq_2 = fem.hermite(sz_1, ip, 1, 0, array)

			for nm in range(0, dims[0]):

				funct = lambda x: exact_1(x_cell( x, nm, 0, fem_type) / sol.init_stretch[0] )

				ind_here = ip + nm * sz_2
				ind_next = ip + half + nm * sz_2

				p_curr = lambda x: pq_1.evaluate(x)
				b[ind_here, 0] = fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

				p_curr = lambda x: pq_2.evaluate(x)
				b[ind_next, 0] = fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

			for nn in range(0, dims[1]):

				funct = lambda y: exact_1(x_cell( y, nn, 1, fem_type) / sol.init_stretch[1] )

				ind_here = ip + nn * sz_2
				ind_next = ip + half + nn * sz_2

				p_curr = lambda y: pq_1.evaluate(y)
				b[ind_here, 1] = fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

				p_curr = lambda y: pq_2.evaluate(y)
				b[ind_next, 1] = fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

	elif fem_type == "lagrange_dg":

		for ip in range(0, sz_2):

			pq_1 = fem.lagrange(sz_1, ip, 0, array[:])
			p_curr = lambda x: pq_1.evaluate(x)

			for nm in range(0, dims[0]):

				funct = lambda x: exact_1(x_cell( x, nm, 0, fem_type) / sol.init_stretch[0] )

				ind_here = ip + nm * sz_2
				b[ind_here, 0] = fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

			for nn in range(0, dims[1]):

				funct = lambda y: exact_1(x_cell( y, nn, 1, fem_type) / sol.init_stretch[1] )

				ind_here = ip + nn * sz_2
				b[ind_here, 1] = fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

	elif fem_type == "lagrange_even_dg":

		for ip in range(0, sz_2):

			pq_1 = fem.lagrange_even(sz_1, ip, 0)
			p_curr = lambda x: pq_1.evaluate(x)

			for nm in range(0, dims[0]):

				funct = lambda x: exact_1(x_cell( x, nm, 0, fem_type) / sol.init_stretch[0] )

				ind_here = ip + nm * sz_2
				b[ind_here, 0] = fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

			for nn in range(0, dims[1]):

				funct = lambda y: exact_1(x_cell( y, nn, 1, fem_type) / sol.init_stretch[1] )

				ind_here = ip + nn * sz_2
				b[ind_here, 1] = fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)


	b[:,0] = wgt_2[0] * b[:,0]
	b[:,1] = wgt_2[1] * b[:,1]

	return b






def fem_interp_3d(fem_type, input_array = []):
	"""Return a 3 vectors of FEM coefficients in 3 dimensions representing an approximation of funct in the specified FEM space"""

	if (fem_type == "hermite") or (fem_type == "hermite_dg"):
		sz_1 = int( (sol.p - 1)/2 )		
		wgt_3 = sol.HERM_WGT
	else:
		sz_1 = sol.p
		wgt_3 = sol.LAGR_WGT

	if (fem_type == "lagrange") or (fem_type == "lagrange_even"):
		sz_2 = sz_1
	elif fem_type == "hermite_dg":
		sz_2 = 2 * sz_1 + 2
	else:
		sz_2 = sz_1 + 1

	array = fem.array_check( input_array, sz_1, fem_type )

	#Interpolate the system with M X = (f, phi), where phi is the basis function corresponding to a given row
	#of M
	b = np.zeros( ( max(dims) * sz_2, 3 ) )

	if fem_type == "hermite":

		for ip in range(0, sz_2):

			pq_1 = fem.hermite(sz_1, ip, 0, 0, array)
			pq_2 = fem.hermite(sz_1, ip, 1, 0, array)

			for nm in range(0, dims[0]):

				xcell_next = (nm + 1) % dims[0]

				funct = lambda x: exact_1(x_cell( x, nm, 0, fem_type) / sol.init_stretch[0] )

				ind_here = ip + nm * sz_2
				ind_next = ip + xcell_next * sz_2

				p_curr = lambda x: pq_1.evaluate(x)
				b[ind_here, 0] += fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

				p_curr = lambda x: pq_2.evaluate(x)
				b[ind_next, 0] += fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

			for nn in range(0, dims[1]):

				ycell_next = (nn + 1) % dims[1]

				funct = lambda y: exact_1(x_cell( y, nn, 1, fem_type) / sol.init_stretch[1] )

				ind_here = ip + nn * sz_2
				ind_next = ip + ycell_next * sz_2

				p_curr = lambda y: pq_1.evaluate(y)
				b[ind_here, 1] += fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

				p_curr = lambda y: pq_2.evaluate(y)
				b[ind_next, 1] += fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

			for no in range(0, dims[2]):

				zcell_next = (no + 1) % dims[2]

				funct = lambda z: exact_1(x_cell( z, no, 2, fem_type) / sol.init_stretch[2] )

				ind_here = ip + no * sz_2
				ind_next = ip + zcell_next * sz_2

				p_curr = lambda z: pq_1.evaluate(z)
				b[ind_here, 2] += fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

				p_curr = lambda z: pq_2.evaluate(z)
				b[ind_next, 2] += fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

	elif fem_type == "lagrange":

		pq_2 = fem.lagrange(sz_1, sz_2, 0, array[:])

		for nm in range(0, dims[0]):

			xcell_next = (nm + 1) % dims[0]

			funct = lambda x: exact_1(x_cell( x, nm, 0, fem_type) / sol.init_stretch[0] )
			p_curr = lambda x: pq_2.evaluate(x)

			ind_next = xcell_next * sz_2
			b[ind_next, 0] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

			for ip in range(0, sz_2):

				pq_1 = fem.lagrange(sz_1, ip, 0, array[:])
				p_curr = lambda x: pq_1.evaluate(x)

				ind_here = ip + nm * sz_2
				b[ind_here, 0] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

		for nn in range(0, dims[1]):

			ycell_next = (nn + 1) % dims[1]

			funct = lambda y: exact_1(x_cell( y, nn, 1, fem_type) / sol.init_stretch[1] )
			p_curr = lambda y: pq_2.evaluate(y)

			ind_next = ycell_next * sz_2
			b[ind_next, 1] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

			for jp in range(0, sz_2):

				pq_1 = fem.lagrange(sz_1, jp, 0, array[:])
				p_curr = lambda y: pq_1.evaluate(y)

				ind_here = jp + nn * sz_2
				b[ind_here, 1] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

		for no in range(0, dims[2]):

			zcell_next = (no + 1) % dims[2]

			funct = lambda z: exact_1(x_cell( z, no, 2, fem_type) / sol.init_stretch[2] )
			p_curr = lambda z: pq_2.evaluate(z)

			ind_next = zcell_next * sz_2 
			b[ind_next, 2] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

			for kp in range(0, sz_2):

				pq_1 = fem.lagrange(sz_1, kp, 0, array[:])
				p_curr = lambda z: pq_1.evaluate(z)

				ind_here = kp + no * sz_2
				b[ind_here, 2] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

	elif fem_type == "lagrange_even":

		pq_2 = fem.lagrange_even(sz_1, sz_2, 0)

		for nm in range(0, dims[0]):

			xcell_next = (nm + 1) % dims[0]

			funct = lambda x: exact_1(x_cell( x, nm, 0, fem_type) / sol.init_stretch[0] )
			p_curr = lambda x: pq_2.evaluate(x)

			ind_next = xcell_next * sz_2
			b[ind_next, 0] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

			for ip in range(0, sz_2):

				pq_1 = fem.lagrange_even(sz_1, ip, 0)
				p_curr = lambda x: pq_1.evaluate(x)

				ind_here = ip + nm * sz_2
				b[ind_here, 0] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

		for nn in range(0, dims[1]):

			ycell_next = (nn + 1) % dims[1]

			funct = lambda y: exact_1(x_cell( y, nn, 1, fem_type) / sol.init_stretch[1] )
			p_curr = lambda y: pq_2.evaluate(y)

			ind_next = ycell_next * sz_2
			b[ind_next, 1] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

			for jp in range(0, sz_2):

				pq_1 = fem.lagrange_even(sz_1, jp, 0)
				p_curr = lambda y: pq_1.evaluate(y)

				ind_here = jp + nn * sz_2
				b[ind_here, 1] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

		for no in range(0, dims[2]):

			zcell_next = (no + 1) % dims[2]

			funct = lambda z: exact_1(x_cell( z, no, 2, fem_type) / sol.init_stretch[2] )
			p_curr = lambda z: pq_2.evaluate(z)

			ind_next = zcell_next * sz_2 
			b[ind_next, 2] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

			for kp in range(0, sz_2):

				pq_1 = fem.lagrange_even(sz_1, kp, 0)
				p_curr = lambda z: pq_1.evaluate(z)

				ind_here = kp + no * sz_2
				b[ind_here, 2] += fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

	elif fem_type == "hermite_dg":

		half = int(0.5 * sz_2)

		for ip in range(0, half):

			pq_1 = fem.hermite(sz_1, ip, 0, 0, array)
			pq_2 = fem.hermite(sz_1, ip, 1, 0, array)

			for nm in range(0, dims[0]):

				funct = lambda x: exact_1(x_cell( x, nm, 0, fem_type) / sol.init_stretch[0] )

				ind_here = ip + nm * sz_2
				ind_next = ip + half + nm * sz_2

				p_curr = lambda x: pq_1.evaluate(x)
				b[ind_here, 0] = fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

				p_curr = lambda x: pq_2.evaluate(x)
				b[ind_next, 0] = fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

			for nn in range(0, dims[1]):

				funct = lambda y: exact_1(x_cell( y, nn, 1, fem_type) / sol.init_stretch[1] )

				ind_here = ip + nn * sz_2
				ind_next = ip + half + nn * sz_2

				p_curr = lambda y: pq_1.evaluate(y)
				b[ind_here, 1] = fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

				p_curr = lambda y: pq_2.evaluate(y)
				b[ind_next, 1] = fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

			for no in range(0, dims[2]):

				funct = lambda z: exact_1(x_cell( z, no, 2, fem_type) / sol.init_stretch[2] )

				ind_here = ip + no * sz_2
				ind_next = ip + half + no * sz_2

				p_curr = lambda z: pq_1.evaluate(z)
				b[ind_here, 2] = fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

				p_curr = lambda z: pq_2.evaluate(z)
				b[ind_next, 2] = fem.inner_product_funct( funct, p_curr, fem_type, quad_pts )

	elif fem_type == "lagrange_dg":

		for ip in range(0, sz_2):

			pq_1 = fem.lagrange(sz_1, ip, 0, array[:])
			p_curr = lambda x: pq_1.evaluate(x)

			for nm in range(0, dims[0]):

				funct = lambda x: exact_1(x_cell( x, nm, 0, fem_type) / sol.init_stretch[0] )

				ind_here = ip + nm * sz_2
				b[ind_here, 0] = fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

			for nn in range(0, dims[1]):

				funct = lambda y: exact_1(x_cell( y, nn, 1, fem_type) / sol.init_stretch[1] )

				ind_here = ip + nn * sz_2
				b[ind_here, 1] = fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

			for no in range(0, dims[2]):

				funct = lambda z: exact_1(x_cell( z, no, 2, fem_type) / sol.init_stretch[2] )

				ind_here = ip + no * sz_2
				b[ind_here, 2] = fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

	elif fem_type == "lagrange_even_dg":

		for ip in range(0, sz_2):

			pq_1 = fem.lagrange_even(sz_1, ip, 0)
			p_curr = lambda x: pq_1.evaluate(x)

			for nm in range(0, dims[0]):

				funct = lambda x: exact_1(x_cell( x, nm, 0, fem_type) / sol.init_stretch[0] )

				ind_here = ip + nm * sz_2
				b[ind_here, 0] = fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

			for nn in range(0, dims[1]):

				funct = lambda y: exact_1(x_cell( y, nn, 1, fem_type) / sol.init_stretch[1] )

				ind_here = ip + nn * sz_2
				b[ind_here, 1] = fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)

			for no in range(0, dims[2]):

				funct = lambda z: exact_1(x_cell( z, no, 2, fem_type) / sol.init_stretch[2] )

				ind_here = ip + no * sz_2
				b[ind_here, 2] = fem.inner_product_funct(funct, p_curr, fem_type, quad_pts)


	b[:,0] = wgt_3[0] * b[:,0]
	b[:,1] = wgt_3[1] * b[:,1]
	b[:,2] = wgt_3[2] * b[:,2]

	return b






#MAIN:

if __name__ == "__main__":

	print("""\nEnter 0 to for Lagrange with evenly spaced nodes, 
1 for Lagrange with Gauss-Lobatto nodes,
2 for Hermite, 3 for L-even-DG, 4 for L-GL-DG, 
5 for Hermite-DG or any combination thereof:""", end = " ")
	choice = str(input())
	print("\n")

	solvers = []

	for ic in range(0, 6):
		if str(ic) in choice:
			solvers.append( sol.OPTIONS[ic] )

	lsolv = len(solvers)

	print("Beginning calculation of initial condition vectors:", end = "\n\n")

	for si in range(0, lsolv):

		s = solvers[si]
		print("Solver: " + s, end = "\n\n")

		print("Initialising solver variables: ", end = "")

		if s == "hermite":
			q_1 = int((sol.p - 1)/2)
			input_array = fem.hermite.hermite_gen_mat(q_1)
		elif s == "lagrange_dg":
			q_1 = sol.p
			input_array = fem.lagrange.GL_points(q_1)
		elif s == "hermite_dg":
			q_1 = int((sol.p - 1)/2)
			input_array = fem.hermite.hermite_gen_mat(q_1)
		elif s == "lagrange":
			q_1 = sol.p
			input_array = fem.lagrange.GL_points(q_1)
		else:
			input_array = None


		print("Done.\nCalculating initial conditions: ", end = "")

		#Initialise the system with b_0 = M U_0 = (u_0, phi), where phi is the basis function corresponding to a given row
		#of M

		if sol.syst_dims == 2:
			b_0 = fem_interp_2d( s, input_array )
			if not sol.zero_flag:
				b_t_0 = fem_interp_2d( s, input_array )

		elif sol.syst_dims == 3:
			b_0 = fem_interp_3d( s, input_array )
			if not sol.zero_flag:
				b_t_0 = fem_interp_3d( s, input_array )


		print("Done.\nExporting to save file: ", end = "")


		name_1 = s + "_" + str(sol.syst_dims) + "_p" + str(sol.p) + ".npy"
		np.save( name_1, b_0 )

		if not sol.zero_flag:
			name_2 = s + "_t_" + str(sol.syst_dims) + "_p" + str(sol.p) + ".npy"
			np.save( name_2, b_t_0 )

		print("Done.\n")

	print("All integrations complete. Program terminating.\n")



		
