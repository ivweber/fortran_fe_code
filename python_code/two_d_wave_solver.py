# Ivy Weber
# Solve a test wave equation in 2D up to T = 0.5 to test stability of methods

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

grid_res = sol.ax_len
dims = sol.DIMS

def sol_to_plot( sol_vec, fem_type, input_array = [] ):
	"""Return a 2D array of values representing the real values of the solution at various points in the 2D domain"""

	if (fem_type == "hermite") or (fem_type == "hermite_dg"):
		sz_1 = int( (sol.p - 1)/2 )
	else:
		sz_1 = sol.p

	if (fem_type == "lagrange") or (fem_type == "lagrange_even"):
		sz_2 = sz_1
	elif fem_type == "hermite_dg":
		sz_2 = 2 * sz_1 + 2
	else:
		sz_2 = sz_1 + 1

	array = fem.array_check( input_array, sz_1, fem_type )

	plot_axis = np.linspace(0, 1, grid_res)

	x_axis = sol.x_min + (sol.x_max - sol.x_min) * plot_axis
	y_axis = sol.y_min + (sol.y_max - sol.y_min) * plot_axis

	X, Y = np.meshgrid( x_axis, y_axis )
	Z = np.zeros( ( grid_res, grid_res) )

	x_here = 0
	y_here = 0
			
	for ix in range(0, grid_res):
		x_now = plot_axis[ix]
		for iy in range(0, grid_res):
			y_now = plot_axis[iy]

			#cells seems to be flipped (0 and 1 the wrong way round)

			cell = [ m.floor(dims[0] * x_now), m.floor(dims[1] * y_now) ]

			cell[0] = min(cell[0], dims[0] - 1)
			cell[1] = min(cell[1], dims[1] - 1)

			x_here = dims[0] * x_now - cell[0]
			y_here = dims[1] * y_now - cell[1]

			xcell_next = (cell[0]+1) % dims[0]
			ycell_next = (cell[1]+1) % dims[1]

			if fem_type == "hermite":

				vals_here = np.zeros( (sz_2, 2) )
				vals_next = np.zeros( (sz_2, 2) )

				for pp in range(0, sz_2):

					p_here = fem.hermite(sz_1, pp, 0, 0, array)
					p_next = fem.hermite(sz_1, pp, 1, 0, array)

					vals_here[pp, 0] = p_here.evaluate(x_here)
					vals_here[pp, 1] = p_here.evaluate(y_here)

					vals_next[pp, 0] = p_next.evaluate(x_here)
					vals_next[pp, 1] = p_next.evaluate(y_here)

				for ip in range(0, sz_2):
					for jp in range(0, sz_2):

						local_index = ip + sz_2 * (cell[0] + dims[0] * (jp + sz_2 * cell[1]) )
						local_index_next1 = ip + sz_2 * ( xcell_next + dims[0] * (jp + sz_2 * cell[1]) )
						local_index_next2 = ip + sz_2 * (cell[0] + dims[0] * (jp + sz_2 * ycell_next ) )
						local_index_next12 = ip + sz_2 * ( xcell_next + dims[0] * (jp + sz_2 * ycell_next ) )

						Z[ix, iy] += sol_vec[local_index, 0] * vals_here[ip,0] * vals_here[jp,1]
						Z[ix, iy] += sol_vec[local_index_next1, 0] * vals_next[ip,0] * vals_here[jp,1]
						Z[ix, iy] += sol_vec[local_index_next2, 0] * vals_here[ip,0] * vals_next[jp,1]
						Z[ix, iy] += sol_vec[local_index_next12, 0] * vals_next[ip,0] * vals_next[jp,1]

			elif fem_type == "lagrange":

				x_here = 2 * x_here - 1
				y_here = 2 * y_here - 1

				vals_here = np.zeros( (sz_2, 2) )
				vals_next = 2 * [0]

				p_next = fem.lagrange(sz_1, sz_2, 0, array[:])

				vals_next[0] = p_next.evaluate(x_here)
				vals_next[1] = p_next.evaluate(y_here)

				local_index_next12 = sz_2 * ( xcell_next + dims[0] * sz_2 * ycell_next )
				Z[ix, iy] += sol_vec[local_index_next12, 0] * vals_next[0] * vals_next[1]

				for pp in range(0, sz_2):

					p_here = fem.lagrange(sz_1, pp, 0, array[:])

					vals_here[pp, 0] = p_here.evaluate(x_here)
					vals_here[pp, 1] = p_here.evaluate(y_here)

				for ip in range(0, sz_2):

					local_index_next2 = ip + sz_2 * (cell[0] + dims[0] * sz_2 * ycell_next )
					Z[ix, iy] += sol_vec[local_index_next2, 0] * vals_here[ip, 0] * vals_next[1]

					for jp in range(0, sz_2):

						local_index = ip + sz_2 * (cell[0] + dims[0] * (jp + sz_2 * cell[1]) )
						Z[ix, iy] += sol_vec[local_index, 0] * vals_here[ip, 0] * vals_here[jp, 1]

						if ip == 0:

							local_index_next1 = sz_2 * ( xcell_next + dims[0] * (jp + sz_2 * cell[1]) )
							Z[ix, iy] += sol_vec[local_index_next1, 0] * vals_next[0] * vals_here[jp, 1]

			elif fem_type == "lagrange_even":

				x_here = 2 * x_here - 1
				y_here = 2 * y_here - 1

				vals_here = np.zeros( (sz_2, 2) )
				vals_next = 2 * [0]

				p_next = fem.lagrange_even(sz_1, sz_2, 0)

				vals_next[0] = p_next.evaluate(x_here)
				vals_next[1] = p_next.evaluate(y_here)

				local_index_next12 = sz_2 * ( xcell_next + dims[0] * sz_2 * ycell_next )
				Z[ix, iy] += sol_vec[local_index_next12, 0] * vals_next[0] * vals_next[1]

				for pp in range(0, sz_2):

					p_here = fem.lagrange_even(sz_1, pp, 0)

					vals_here[pp, 0] = p_here.evaluate(x_here)
					vals_here[pp, 1] = p_here.evaluate(y_here)

				for ip in range(0, sz_2):

					local_index_next2 = ip + sz_2 * (cell[0] + dims[0] * sz_2 * ycell_next )
					Z[ix, iy] += sol_vec[local_index_next2, 0] * vals_here[ip, 0] * vals_next[1]

					for jp in range(0, sz_2):

						local_index = ip + sz_2 * (cell[0] + dims[0] * (jp + sz_2 * cell[1]) )
						Z[ix, iy] += sol_vec[local_index, 0] * vals_here[ip, 0] * vals_here[jp, 1]

						if ip == 0:

							local_index_next1 = sz_2 * ( xcell_next + dims[0] * (jp + sz_2 * cell[1]) )
							Z[ix, iy] += sol_vec[local_index_next1, 0] * vals_next[0] * vals_here[jp, 1]

			elif fem_type == "hermite_dg":

				half = int(0.5 * sz_2)

				vals_here = np.zeros( (half, 2) )
				vals_next = np.zeros( (half, 2) )

				for pp in range(0, half):

					p_here = fem.hermite(sz_1, pp, 0, 0, array)
					p_next = fem.hermite(sz_1, pp, 1, 0, array)

					vals_here[pp, 0] = p_here.evaluate(x_here)
					vals_here[pp, 1] = p_here.evaluate(y_here)

					vals_next[pp, 0] = p_next.evaluate(x_here)
					vals_next[pp, 1] = p_next.evaluate(y_here)

				for ip in range(0, half):
					for jp in range(0, half):

						local_index = ip + sz_2 * (cell[0] + dims[0] * (jp + sz_2 * cell[1]) )
						local_index_next1 = ip + half + sz_2 * (cell[0] + dims[0] * (jp + sz_2 * cell[1]) )
						local_index_next2 = ip + sz_2 * (cell[0] + dims[0] * (jp + half + sz_2 * cell[1]) )
						local_index_next12 = ip + half + sz_2 * (cell[0] + dims[0] * (jp + half + sz_2 * cell[1]) )

						Z[ix, iy] += sol_vec[local_index, 0] * vals_here[ip,0] * vals_here[jp,1]
						Z[ix, iy] += sol_vec[local_index_next1, 0] * vals_next[ip,0] * vals_here[jp,1]
						Z[ix, iy] += sol_vec[local_index_next2, 0] * vals_here[ip,0] * vals_next[jp,1]
						Z[ix, iy] += sol_vec[local_index_next12, 0] * vals_next[ip,0] * vals_next[jp,1]

			elif fem_type == "lagrange_dg":

				x_here = 2 * x_here - 1
				y_here = 2 * y_here - 1

				vals_here = np.zeros( (sz_2, 2) )

				for pp in range(0, sz_2):

					p_here = fem.lagrange(sz_1, pp, 0, array[:])

					vals_here[pp, 0] = p_here.evaluate(x_here)
					vals_here[pp, 1] = p_here.evaluate(y_here)

				for ip in range(0, sz_2):
					for jp in range(0, sz_2):

						local_index = ip + sz_2 * (cell[0] + dims[0] * (jp + sz_2 * cell[1]) )
						Z[ix, iy] += sol_vec[local_index, 0] * vals_here[ip, 0] * vals_here[jp, 1]

			elif fem_type == "lagrange_even_dg":

				x_here = 2 * x_here - 1
				y_here = 2 * y_here - 1

				vals_here = np.zeros( (sz_2, 2) )

				for pp in range(0, sz_2):

					p_here = fem.lagrange_even(sz_1, pp, 0)

					vals_here[pp, 0] = p_here.evaluate(x_here)
					vals_here[pp, 1] = p_here.evaluate(y_here)

				for ip in range(0, sz_2):
					for jp in range(0, sz_2):

						local_index = ip + sz_2 * (cell[0] + dims[0] * (jp + sz_2 * cell[1]) )
						Z[ix, iy] += sol_vec[local_index, 0] * vals_here[ip, 0] * vals_here[jp, 1]


	return X, Y, Z



#MAIN:

if __name__ == "__main__":

	if sol.syst_dims != 2:
		print("ERROR: Solver incompatible with dimensionality chosen!")
	else:

#		ds = 1.0 / m.sqrt( m.pow(sol.dx,-2) + m.pow(sol.dy,-2) )
		ds = sol.dx

		print(\n"p = " + str(sol.p))
		print("\nC_eff = " + str(sol.c_eff_1) )

		c_eff_2 = sol.c_eff / m.sqrt(2)

		print("C_eff for 2D system = " + str(c_eff_2) )

		cfl_lag = c_eff_2 / sol.p
		cfl_herm = 2 * c_eff_2 / (sol.p + 1)
		cfl_dg = c_eff_2 / (sol.p + 1)

		dt_lag = cfl_lag * ds
		dt_herm = cfl_herm * ds
		dt_dg = cfl_dg * ds

		beta = 2 * sol.p * (sol.p + 1)

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

		for si in range(0, lsolv):

			s = solvers[si]
			print("Solver: " + s, end = "\n\n")

			discont = (s == "lagrange_even_dg") or (s == "lagrange_dg") or (s == "hermite_dg")

			print("Initialising solver variables: ", end = "")

			if (s == "hermite") or (s == "hermite_dg"):
				q_1 = int((sol.p - 1)/2)
				q_2 = q_1 + 1

				input_array = fem.hermite.hermite_gen_mat(q_1)

				w_1 = sol.HERM_WGT
				w_2 = dims

				if s == "hermite_dg":
					offset = 2 * q_2
				else:
					offset = q_2

			else:
				q_1 = sol.p
				if (s == "lagrange_even" or s == "lagrange"):
					q_2 = sol.p
				else:
					q_2 = sol.p + 1

				if (s == "lagrange") or (s == "lagrange_dg"):
					input_array = fem.lagrange.GL_points(q_1)
				else:
					input_array = None

				w_1 = sol.LAGR_WGT
				w_2 = (2.0 * sol.M, 2.0 * sol.N)

				offset = q_2

			if discont:
				dt = dt_dg
			elif s == "hermite":
				dt = dt_herm
			else:
				dt = dt_lag

			print("Done.\nGenerating FEM matrices: ", end = "")

			M_small_1 = fem.mass_1(q_1, s, input_array)
			if not discont:
				M_small_2 = fem.mass_2(q_1, s, input_array)

			A_small_1 = fem.laplace_1(q_1, s, input_array, beta)
			A_small_2 = fem.laplace_2(q_1, s, input_array, beta)

			for dim_ind in range(0,2):

				M_1 = w_1[dim_ind] * M_small_1

				if discont:
					MI = sp.identity(dims[dim_ind], format = 'csr')
					MM = sp.kron(MI, M_1)
				else:
					M_2 = w_1[dim_ind] * M_small_2
					MM = sp.lil_matrix( (dims[dim_ind] * offset, dims[dim_ind] * offset) )

				A_1 = w_2[dim_ind] * A_small_1
				A_2 = w_2[dim_ind] * A_small_2

				AM = sp.lil_matrix( (dims[dim_ind] * offset, dims[dim_ind] * offset) )

				if discont:
					for n in range(0, dims[dim_ind]):

						start_1 = n * offset
						start_2 = ((n + 1) % dims[dim_ind]) * offset

						#Add blocks to the sparse matrices
						for i in range(0, offset):

							AM[start_1 + i, start_1: start_1 + offset] = list(A_1[i,:])

							for j in range(0, offset):

								AM[start_1 + i, start_2 + j] += A_2[i,j]
								AM[start_2 + i, start_1 + j] += A_2[j,i]

					MM = MM.tocsc()

					M_inv = sp.kron(MI, np.linalg.inv(M_1))
					M_inv = M_inv.todense()

				else:
					for n in range(0, dims[dim_ind]):

						start_1 = n * offset
						start_2 = ((n + 1) % dims[dim_ind]) * offset

						#Add blocks to the sparse matrices
						for i in range(0, offset):

							MM[start_1 + i, start_1: start_1 + offset] = list(M_1[i,:])
							AM[start_1 + i, start_1: start_1 + offset] = list(A_1[i,:])

							for j in range(0, offset):

								MM[start_1 + i, start_2 + j] += M_2[i,j]
								MM[start_2 + i, start_1 + j] += M_2[j,i]

								AM[start_1 + i, start_2 + j] += A_2[i,j]
								AM[start_2 + i, start_1 + j] += A_2[j,i]

					MM = MM.tocsc()

					MM_2 = MM.todense()
					M_inv = np.linalg.inv(MM_2)

				if dim_ind == 0:
					B = M_inv @ AM
					MB = lu.splu(MM)

				else:
					C = M_inv @ AM
					MC = lu.splu(MM)

			I_m = sp.eye( sol.M * offset )
			I_n = sp.eye( sol.N * offset )

			#Usage of kronecker products gives the ordering of first x-element, then x-cell, then y-element, then y-cell
			B = sp.kron( I_n, B ) + sp.kron( C, I_m )

			print("Done.\nLoading initial conditions: ", end = "")

			#Initialise the system with M U_0 = (u_0, phi), where phi is the basis function corresponding to a given row
			#of M

			b_0 = np.load( s + "_2_p" + str(sol.p) + ".npy" )

			U_0_x = MB.solve(b_0[:,0])
			U_0_y = MC.solve(b_0[:,1])

			U_0 = sp.kron(U_0_y, U_0_x)
			U_0 = U_0.todense()

			if not sol.zero_flag:
				b_t_0 = np.load( s + "_t_2_p" + str(sol.p) + ".npy" )

				U_t_0_x = MB.solve(b_t_0[:,0])
				U_t_0_y = MC.solve(b_t_0[:,1])

				U_t_0 = sp.kron(U_t_0_y, U_t_0_x)
				U_t_0 = U_t_0.todense()

			else:
				U_t_0 = np.zeros( np.shape( U_0 ) )

			U_0 = np.transpose(U_0)
			U_t_0 = np.transpose(U_t_0)

			print("Done.\nProceeding to time integration: ", end = "")

			t = 0

			#Use the time-stepping method U_(n+1) = 2 U_n - U_(n-1) +dt^2 inv(M) A U_(n)
				#Initialize with symplectic Euler
#			U_1 = U_0 + dt * U_t_0 + 0.5 * (dt ** 2) * (B @ U_0)
			U_t_0 = U_t_0 + dt * (B.dot(U_0))
			U_1 = U_0 + dt * U_t_0 

			t = dt

			U_prev = U_0
			U_curr = U_1
			count = 1

			while t < sol.T:

				Y = B.dot(U_curr)
				Y *= dt**2

				Y = Y + 2 * U_curr - U_prev

				U_prev = U_curr[:]
				U_curr = Y[:]
				t += dt
				count += 1

			print("Done.\nGenerating plot data: ", end = "")

			#Plot the 2D FEM solution

			X, Y, U_calc_0 = sol_to_plot( U_0, s, input_array )
			X, Y, U_calc = sol_to_plot( U_curr, s, input_array )

			plt.figure(1 + 2 * si)

			ax = plt.axes( projection = "3d" )
			ax.plot_surface( X, Y, U_calc, cmap = 'spring', edgecolor='none')

			ax.legend = [ "FEM solution using " + s + " elements of order " + str(sol.p) + " and C_eff = " + str(sol.c_eff) ]
			
			plt.figure(2 + 2 * si)

			ax = plt.axes( projection = "3d" )
			ax.plot_surface( X, Y, U_calc_0, cmap = 'spring', edgecolor='none')

			ax.legend = [  "FEM solution using " + s + " elements of order " + str(sol.p) + " at time t = 0" ]

			print("Done.\n")

		print("Plotting solutions: ", end = "")
		plt.show()

		print("Done.\n\nProgram terminating.\n")

#END 
