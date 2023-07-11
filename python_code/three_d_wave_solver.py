# Ivy Weber
# Solve a test wave equation in 3D up to T = 0.5 to test stability of methods

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


def index_global( funct_coords, cell_coords, grid_dims, sz ):
	"""Return an index for a given basis function when entire system is written as a 1D vector"""

	return funct_coords[0] + sz * ( cell_coords[0] + grid_dims[0] * ( funct_coords[1] + sz * (cell_coords[1] + grid_dims[1] * (funct_coords[2] + sz * cell_coords[2] ) ) ) )

grid_res = sol.ax_len
dims = sol.DIMS
use_inits = sol.initial_conds_present

def sol_to_plot( sol_vec, fem_type, input_array = [] ):
	"""Return a 3D array of values representing the real values of the solution at various points in the 2D domain"""

	if (fem_type == "hermite") or (fem_type == "hermite_dg"):
		sz_1 = int( (sol.p - 1)/2 )
	else:
		sz_1 = sol.p

	if (fem_type == "lagrange_even") or (fem_type == "lagrange"):
		sz_2 = sz_1
	elif fem_type == "hermite_dg":
		sz_2 = 2 * sz_1 + 2
	else:		
		sz_2 = sz_1 + 1

	array = fem.array_check( input_array, sz_1, fem_type )

	plot_axis = np.linspace(0, 1, grid_res)

	x_axis = sol.x_min + (sol.x_max - sol.x_min) * plot_axis
	y_axis = sol.y_min + (sol.y_max - sol.y_min) * plot_axis
	z_axis = sol.z_min + (sol.z_max - sol.z_min) * plot_axis

	W = np.zeros( ( grid_res, grid_res, grid_res) )

	x_here = 0
	y_here = 0
	z_here = 0

	for ix in range(0, grid_res):
		x_now = plot_axis[ix]
		for iy in range(0, grid_res):
			y_now = plot_axis[iy]
			for iz in range(0, grid_res):
				z_now = plot_axis[iz]

				#cells seems to be flipped (0 and 1 the wrong way round)

				cell = [ m.floor(dims[0] * x_now), m.floor(dims[1] * y_now), m.floor(dims[2] * z_now) ]

				cell[0] = min(cell[0], dims[0] - 1)
				cell[1] = min(cell[1], dims[1] - 1)
				cell[2] = min(cell[2], dims[2] - 1)

				x_here = dims[0] * x_now - cell[0]
				y_here = dims[1] * y_now - cell[1]
				z_here = dims[2] * z_now - cell[2]

				xcell_next = (cell[0]+1) % dims[0]
				ycell_next = (cell[1]+1) % dims[1]
				zcell_next = (cell[2]+1) % dims[2]

				if fem_type == "hermite":

					vals_here = np.zeros( (sz_2, 3) )
					vals_next = np.zeros( (sz_2, 3) )

					for pp in range(0, sz_2):

						p_here = fem.hermite(sz_1, pp, 0, 0, array)
						p_next = fem.hermite(sz_1, pp, 1, 0, array)

						vals_here[pp, 0] = p_here.evaluate(x_here)
						vals_here[pp, 1] = p_here.evaluate(y_here)
						vals_here[pp, 2] = p_here.evaluate(z_here)

						vals_next[pp, 0] = p_next.evaluate(x_here)
						vals_next[pp, 1] = p_next.evaluate(y_here)
						vals_next[pp, 2] = p_next.evaluate(z_here)

					for ip in range(0, sz_2):
						for jp in range(0, sz_2):
							for kp in range(0, sz_2):

								local_index = index_global( (ip, jp, kp), cell, dims, sz_2 )
								local_index_next1 = index_global( (ip, jp, kp), (xcell_next, cell[1], cell[2]), dims, sz_2 )
								local_index_next2 = index_global( (ip, jp, kp), (cell[0], ycell_next, cell[2]), dims, sz_2 )
								local_index_next12 = index_global( (ip, jp, kp), (xcell_next, ycell_next, cell[2]), dims, sz_2 )
								local_index_next3 = index_global( (ip, jp, kp), (cell[0], cell[1], zcell_next), dims, sz_2 )
								local_index_next13 = index_global( (ip, jp, kp), (xcell_next, cell[1], zcell_next), dims, sz_2 )
								local_index_next23 = index_global( (ip, jp, kp), (cell[0], ycell_next, zcell_next), dims, sz_2 )
								local_index_next123 = index_global( (ip, jp, kp), (xcell_next, ycell_next, zcell_next), dims, sz_2 )

								W[ix, iy, iz] += sol_vec[local_index, 0] * vals_here[ip,0] * vals_here[jp,1] * vals_here[kp, 2]
								W[ix, iy, iz] += sol_vec[local_index_next1, 0] * vals_next[ip,0] * vals_here[jp,1] * vals_here[kp, 2]
								W[ix, iy, iz] += sol_vec[local_index_next2, 0] * vals_here[ip,0] * vals_next[jp,1] * vals_here[kp, 2]
								W[ix, iy, iz] += sol_vec[local_index_next12, 0] * vals_next[ip,0] * vals_next[jp,1] * vals_here[kp, 2]
								W[ix, iy, iz] += sol_vec[local_index3, 0] * vals_here[ip,0] * vals_here[jp,1] * vals_next[kp, 2]
								W[ix, iy, iz] += sol_vec[local_index_next13, 0] * vals_next[ip,0] * vals_here[jp,1] * vals_next[kp, 2]
								W[ix, iy, iz] += sol_vec[local_index_next23, 0] * vals_here[ip,0] * vals_next[jp,1] * vals_next[kp, 2]
								W[ix, iy, iz] += sol_vec[local_index_next123, 0] * vals_next[ip,0] * vals_next[jp,1] * vals_next[kp, 2]

				elif fem_type == "lagrange":

					x_here = 2 * x_here - 1
					y_here = 2 * y_here - 1
					z_here = 2 * z_here - 1

					vals_here = np.zeros( (sz_2, 3) )
					vals_next = 3 * [0]

					p_next = fem.lagrange(sz_1, sz_2, 0, array[:])

					vals_next[0] = p_next.evaluate(x_here)
					vals_next[1] = p_next.evaluate(y_here)
					vals_next[2] = p_next.evaluate(z_here)

					local_index_next123 = index_global( (0,0,0), (xcell_next, ycell_next, zcell_next), dims, sz_2 )
					W[ix, iy, iz] += sol_vec[local_index_next123, 0] * vals_next[0] * vals_next[1] * vals_next[2]

					for pp in range(0, sz_2):

						p_here = fem.lagrange(sz_1, pp, 0, array[:])

						vals_here[pp, 0] = p_here.evaluate(x_here)
						vals_here[pp, 1] = p_here.evaluate(y_here)
						vals_here[pp, 2] = p_here.evaluate(z_here)

					for ip in range(0, sz_2):

						local_index_next23 = index_global( (ip,0,0), (cell[0], ycell_next, zcell_next), dims, sz_2 )
						W[ix, iy, iz] += sol_vec[local_index_next23, 0] * vals_here[ip, 0] * vals_next[1] * vals_next[2]

						for jp in range(0, sz_2):

							local_index_next3 = index_global( (ip,jp,0), (cell[0], cell[1], zcell_next), dims, sz_2 )
							W[ix, iy, iz] += sol_vec[local_index_next3, 0] * vals_here[ip, 0] * vals_here[jp, 1] * vals_next[2]

							if ip == 0:

								local_index_next13 = index_global( (0,jp,0), (xcell_next, cell[1], zcell_next), dims, sz_2 )
								W[ix, iy, iz] += sol_vec[local_index_next13, 0] * vals_next[0] * vals_here[jp, 1] * vals_next[2]

							for kp in range(0, sz_2):

								local_index = index_global( (ip, jp, kp), cell, dims, sz_2 )
								W[ix, iy, iz] += sol_vec[local_index, 0] * vals_here[ip, 0] * vals_here[jp, 1] * vals_here[kp, 2]

								if jp == 0:

									local_index_next2 = index_global( (ip,0,kp), (cell[0], ycell_next, cell[2]), dims, sz_2 )
									W[ix, iy, iz] += sol_vec[local_index_next2, 0] * vals_here[ip, 0] * vals_next[1] * vals_here[kp, 2]

								if ip == 0:

									local_index_next1 = index_global( (0, jp, kp), (xcell_next, cell[1], cell[2]), dims, sz_2 )
									W[ix, iy, iz] += sol_vec[local_index_next1, 0] * vals_next[0] * vals_here[jp, 1] * vals_here[kp, 2]

									if jp == 0:

										local_index_next12 = index_global( (0,0,kp), (xcell_next, ycell_next, cell[2]), dims, sz_2 )
										W[ix, iy, iz] += sol_vec[local_index_next12, 0] * vals_next[0] * vals_next[1] * vals_here[kp, 2]

				elif fem_type == "lagrange_even":

					x_here = 2 * x_here - 1
					y_here = 2 * y_here - 1
					z_here = 2 * z_here - 1

					vals_here = np.zeros( (sz_2, 3) )
					vals_next = 3 * [0]

					p_next = fem.lagrange_even(sz_1, sz_2, 0)

					vals_next[0] = p_next.evaluate(x_here)
					vals_next[1] = p_next.evaluate(y_here)
					vals_next[2] = p_next.evaluate(z_here)

					local_index_next123 = index_global( (0,0,0), (xcell_next, ycell_next, zcell_next), dims, sz_2 )
					W[ix, iy, iz] += sol_vec[local_index_next123, 0] * vals_next[0] * vals_next[1] * vals_next[2]

					for pp in range(0, sz_2):

						p_here = fem.lagrange_even(sz_1, pp, 0)

						vals_here[pp, 0] = p_here.evaluate(x_here)
						vals_here[pp, 1] = p_here.evaluate(y_here)
						vals_here[pp, 2] = p_here.evaluate(z_here)

					for ip in range(0, sz_2):

						local_index_next23 = index_global( (ip,0,0), (cell[0], ycell_next, zcell_next), dims, sz_2 )
						W[ix, iy, iz] += sol_vec[local_index_next23, 0] * vals_here[ip, 0] * vals_next[1] * vals_next[2]

						for jp in range(0, sz_2):

							local_index_next3 = index_global( (ip,jp,0), (cell[0], cell[1], zcell_next), dims, sz_2 )
							W[ix, iy, iz] += sol_vec[local_index_next3, 0] * vals_here[ip, 0] * vals_here[jp, 1] * vals_next[2]

							if ip == 0:

								local_index_next13 = index_global( (0,jp,0), (xcell_next, cell[1], zcell_next), dims, sz_2 )
								W[ix, iy, iz] += sol_vec[local_index_next13, 0] * vals_next[0] * vals_here[jp, 1] * vals_next[2]

							for kp in range(0, sz_2):

								local_index = index_global( (ip, jp, kp), cell, dims, sz_2 )
								W[ix, iy, iz] += sol_vec[local_index, 0] * vals_here[ip, 0] * vals_here[jp, 1] * vals_here[kp, 2]

								if jp == 0:

									local_index_next2 = index_global( (ip,0,kp), (cell[0], ycell_next, cell[2]), dims, sz_2 )
									W[ix, iy, iz] += sol_vec[local_index_next2, 0] * vals_here[ip, 0] * vals_next[1] * vals_here[kp, 2]

								if ip == 0:

									local_index_next1 = index_global( (0, jp, kp), (xcell_next, cell[1], cell[2]), dims, sz_2 )
									W[ix, iy, iz] += sol_vec[local_index_next1, 0] * vals_next[0] * vals_here[jp, 1] * vals_here[kp, 2]

									if jp == 0:

										local_index_next12 = index_global( (0,0,kp), (xcell_next, ycell_next, cell[2]), dims, sz_2 )
										W[ix, iy, iz] += sol_vec[local_index_next12, 0] * vals_next[0] * vals_next[1] * vals_here[kp, 2]

				elif fem_type == "hermite_dg":

					half = int(0.5 * sz_2)

					vals_here = np.zeros( (half, 3) )
					vals_next = np.zeros( (half, 3) )

					for pp in range(0, half):

						p_here = fem.hermite(sz_1, pp, 0, 0, array)
						p_next = fem.hermite(sz_1, pp, 1, 0, array)

						vals_here[pp, 0] = p_here.evaluate(x_here)
						vals_here[pp, 1] = p_here.evaluate(y_here)
						vals_here[pp, 2] = p_here.evaluate(z_here)

						vals_next[pp, 0] = p_next.evaluate(x_here)
						vals_next[pp, 1] = p_next.evaluate(y_here)
						vals_next[pp, 2] = p_next.evaluate(z_here)

					for ip in range(0, half):
						for jp in range(0, half):
							for kp in range(0, half):

								local_index = index_global( (ip, jp, kp), cell, dims, sz_2)
								local_index_next1 = index_global( (ip + half, jp, kp), cell, dims, sz_2)
								local_index_next2 = index_global( (ip, jp + half, kp), cell, dims, sz_2)
								local_index_next12 = index_global( (ip + half, jp + half, kp), cell, dims, sz_2)
								local_index_next3 = index_global( (ip, jp, kp + half), cell, dims, sz_2)
								local_index_next13 = index_global( (ip + half, jp, kp + half), cell, dims, sz_2)
								local_index_next23 = index_global( (ip, jp + half, kp + half), cell, dims, sz_2)
								local_index_next123 = index_global( (ip + half, jp + half, kp + half), cell, dims, sz_2)

								W[ix, iy, iz] += sol_vec[local_index, 0] * vals_here[ip,0] * vals_here[jp,1] * vals_here[kp, 2]
								W[ix, iy, iz] += sol_vec[local_index_next1, 0] * vals_next[ip,0] * vals_here[jp,1] * vals_here[kp, 2]
								W[ix, iy, iz] += sol_vec[local_index_next2, 0] * vals_here[ip,0] * vals_next[jp,1] * vals_here[kp, 2]
								W[ix, iy, iz] += sol_vec[local_index_next12, 0] * vals_next[ip,0] * vals_next[jp,1] * vals_here[kp, 2]
								W[ix, iy, iz] += sol_vec[local_index_next3, 0] * vals_here[ip,0] * vals_here[jp,1] * vals_next[kp, 2]
								W[ix, iy, iz] += sol_vec[local_index_next13, 0] * vals_next[ip,0] * vals_here[jp,1] * vals_next[kp, 2]
								W[ix, iy, iz] += sol_vec[local_index_next23, 0] * vals_here[ip,0] * vals_next[jp,1] * vals_next[kp, 2]
								W[ix, iy, iz] += sol_vec[local_index_next123, 0] * vals_next[ip,0] * vals_next[jp,1] * vals_next[kp, 2]

				elif fem_type == "lagrange_dg":

					x_here = 2 * x_here - 1
					y_here = 2 * y_here - 1
					z_here = 2 * z_here - 1

					vals_here = np.zeros( (sz_2, 3) )

					for pp in range(0, sz_2):

						p_here = fem.lagrange(sz_1, pp, 0, array[:])

						vals_here[pp, 0] = p_here.evaluate(x_here)
						vals_here[pp, 1] = p_here.evaluate(y_here)
						vals_here[pp, 2] = p_here.evaluate(z_here)

					for ip in range(0, sz_2):
						for jp in range(0, sz_2):
							for kp in range(0, sz_2):

								local_index = index_global( (ip, jp, kp), cell, dims, sz_2 )
								W[ix, iy, iz] += sol_vec[local_index, 0] * vals_here[ip, 0] * vals_here[jp, 1] * vals_here[kp, 2]

				elif fem_type == "lagrange_even_dg":

					x_here = 2 * x_here - 1
					y_here = 2 * y_here - 1
					z_here = 2 * z_here - 1

					vals_here = np.zeros( (sz_2, 3) )

					for pp in range(0, sz_2):

						p_here = fem.lagrange_even(sz_1, pp, 0)

						vals_here[pp, 0] = p_here.evaluate(x_here)
						vals_here[pp, 1] = p_here.evaluate(y_here)
						vals_here[pp, 2] = p_here.evaluate(z_here)

					for ip in range(0, sz_2):
						for jp in range(0, sz_2):
							for kp in range(0, sz_2):

								local_index = index_global( (ip, jp, kp), cell, dims, sz_2 )
								W[ix, iy, iz] += sol_vec[local_index, 0] * vals_here[ip, 0] * vals_here[jp, 1] * vals_here[kp, 2]

	return x_axis, y_axis, z_axis, W



#MAIN:

if __name__ == "__main__":

	if sol.syst_dims != 3:
		print("ERROR: Solver incompatible with dimensionality chosen!")
	else:

		ds = 1.0 / m.sqrt( m.pow( sol.dx, -2) + m.pow( sol.dy, -2) + m.pow( sol.dz, -2) )

		print("\nC_eff = " + str(sol.c_eff) )
		print("p = " + str(sol.p), end = "\n\n" )

		cfl_lag = sol.c_eff / sol.p
		cfl_herm = 2 * sol.c_eff / (sol.p + 1)
		cfl_dg = sol.c_eff / (sol.p + 1)

		dt_lag = cfl_lag * ds
		dt_herm = cfl_herm * ds
		dt_dg = cfl_dg * ds

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

#Modified up to here

		for si in range(0, lsolv):

			s = sol.SOLVERS[si]
			print("Solver: " + s, end = "\n\n")

			discont = (s == "lagrange_even_dg") or (s == "lagrange_dg") or (s == "hermite_dg")

			print("Initialising solver variables: ", end = "")

			if s == "hermite":
				q_1 = int((sol.p - 1)/2)
				q_2 = q_1 + 1

				input_array = fem.hermite.hermite_gen_mat(q_1)

				w_1 = (sol.dx, sol.dy, sol.dz)
				w_2 = dims

				dt = dt_herm
				cfl = cfl_herm

			else:
				q_1 = sol.p
				q_2 = sol.p

				if s == "lagrange":
					input_array = fem.lagrange.GL_points(q_1)
				else:
					input_array = None

				w_1 = (0.5 * sol.dx, 0.5 * sol.dy, 0.5 * sol.dz)
				w_2 = (2.0 * sol.M, 2.0 * sol.N, 2.0 * sol.O)

				dt = dt_lag
				cfl = cfl_lag

			print("Done.\nGenerating FEM matrices: ", end = "")

			M_small_1 = fem.mass_1(q_1, s, input_array)
			M_small_2 = fem.mass_2(q_1, s, input_array)

			A_small_1 = fem.laplace_1(q_1, s, input_array)
			A_small_2 = fem.laplace_2(q_1, s, input_array)

			for dim_ind in range(0,3):

				M_1 = w_1[dim_ind] * M_small_1
				M_2 = w_1[dim_ind] * M_small_2

				A_1 = w_2[dim_ind] * A_small_1
				A_2 = w_2[dim_ind] * A_small_2

				MM = sp.lil_matrix( (dims[dim_ind] * q_2, dims[dim_ind] * q_2) )
				AM = sp.lil_matrix( (dims[dim_ind] * q_2, dims[dim_ind] * q_2) )

				for n in range(0, dims[dim_ind]):

					start_1 = n * q_2
					start_2 = ((n + 1) % dims[dim_ind]) * q_2

					#Add blocks to the sparse matrices
					for i in range(0, q_2):

						MM[start_1 + i, start_1: start_1 + q_2] = list(M_1[i,:])
						AM[start_1 + i, start_1: start_1 + q_2] = list(A_1[i,:])

						for j in range(0, q_2):

							MM[start_1 + i, start_2 + j] += M_2[i,j]
							MM[start_2 + i, start_1 + j] += M_2[j,i]

							AM[start_1 + i, start_2 + j] += A_2[i,j]
							AM[start_2 + i, start_1 + j] += A_2[j,i]

				MM = MM.tocsc()

				MM_2 = MM.todense()
				M_inv = np.linalg.inv(MM_2)

				if dim_ind == 0:
					B = M_inv @ AM

					if use_inits:
						MB = lu.splu( MM )

				elif dim_ind == 1:
					C = M_inv @ AM

					if use_inits:
						MC = lu.splu( MM )

				else:
					D = M_inv @ AM

					if use_inits:
						MD = lu.splu( MM )


			I_m = sp.eye( sol.M * q_2 )
			I_n = sp.eye( sol.N * q_2 )
			I_mn = sp.eye( sol.M * sol.N * q_2 * q_2 )
			I_o = sp.eye( sol.O * q_2 )

			#Usage of kronecker products gives the ordering of first x-element, then x-cell, then y-element, then y-cell
			B = sp.kron( I_n, B ) + sp.kron( C, I_m )
			B = sp.kron( I_o, B ) + sp.kron( D, I_mn )

			print("Done.\nLoading initial conditions: ", end = "")

			#Initialise the system with M U_0 = (u_0, phi), where phi is the basis function corresponding to a given row
			#of M

			if use_inits:

				b = np.load( s + "_3_p" + str(sol.p) + ".npy" )

				b_x = np.zeros( (sol.M * q_2, 1) )
				b_y = np.zeros( (sol.N * q_2, 1) )
				b_z = np.zeros( (sol.O * q_2, 1) )

				b_x[:,0] = b[:(sol.M * q_2),0]
				b_y[:,0] = b[:(sol.N * q_2),1]
				b_z[:,0] = b[:(sol.O * q_2),2]

				U_x = MB.solve(b_x)
				U_y = MC.solve(b_y)
				U_z = MD.solve(b_z)
				
				U_0 = sp.kron( U_z, sp.kron( U_y, U_x ) )

				U_0 = U_0.todense()
				U_t_0 = np.zeros( np.shape( U_0 ) )

			else:
				U_0 = np.zeros( (sol.M * sol.N * sol.O * q_2 * q_2 * q_2, 1) )
				U_0[ index_global( (0,0,0), (4,4,4), dims, q_2 ), 0] = 1
				U_t_0 = np.zeros( np.shape( U_0 ) )

			print("Done.\nProceeding to time integration: ", end = "")

			t = 0

			#Use the time-stepping method U_(n+1) = 2 U_n - U_(n-1) +dt^2 inv(M) A U_(n)
			U_1 = U_0 + dt * U_t_0 + 0.5 * (dt ** 2) * (B @ U_0)

			t = dt

			U_prev = U_0
			U_curr = U_1
			count = 1

			while t < sol.T:

				Y = B @ U_curr
				Y *= dt**2

				Y = Y + 2 * U_curr - U_prev

				U_prev = U_curr[:]
				U_curr = Y[:]
				t += dt
				count += 1

			#Plot the 3D FEM solution

			print("Done.\nGenerating plot data: ", end = "")

			X, Y, Z, U_calc_0 = sol_to_plot( U_0, s, input_array )
			X, Y, Z, U_calc = sol_to_plot( U_curr, s, input_array )

			XonZ, YonZ = np.meshgrid( X, Y )
			XonY, ZonY = np.meshgrid( X, Z )
			YonX, ZonX = np.meshgrid( Y, Z )

			CFL = round(cfl, 4)

			print("Done.\nSetting up solution cross-sections: ", end = "")

			fig_1 = plt.figure(1 + 2 * si, figsize = plt.figaspect( 1.0/3.0 ) )
			fig_1.suptitle( "FEM solution using " + s + " elements of order " + str(sol.p) + " and C_eff = " + str(c_eff) + " at t=" + str(t) )

			ax = fig_1.add_subplot(1, 3, 1, projection = "3d" )
			ax.plot_surface( XonZ, YonZ, U_calc[:, :, int(0.5 * grid_res)], cmap = 'spring', edgecolor='none')
			ax.set_xlabel("x")
			ax.set_ylabel("y")
			ax.set_zlabel("U(x,y,0,t)")
			ax.set_title("x-y plane")

			ax = fig_1.add_subplot(1, 3, 2, projection = "3d" )
			ax.plot_surface( XonY, ZonY, U_calc[:, int(0.5 * grid_res), :], cmap = 'spring', edgecolor='none')
			ax.set_xlabel("x")
			ax.set_ylabel("z")
			ax.set_zlabel("U(x,0,z,t)")
			ax.set_title("x-z plane")

			ax = fig_1.add_subplot(1, 3, 3, projection = "3d" )
			ax.plot_surface( YonX, ZonX, U_calc[int(0.5 * grid_res), :, :], cmap = 'spring', edgecolor="none")
			ax.set_xlabel("y")
			ax.set_ylabel("z")
			ax.set_zlabel("U(0,y,z,t)")
			ax.set_title("y-z plane")

			
			fig_2 = plt.figure(2 + 2 * si, figsize = plt.figaspect( 1.0/3.0 ) )
			fig_2.suptitle( "FEM solution using " + s + " elements of order " + str(sol.p) + " at time t = 0" )

			ax = fig_2.add_subplot(1, 3, 1, projection = "3d" )
			ax.plot_surface( XonZ, YonZ, U_calc_0[:, :, int(0.5 * grid_res)], cmap = 'spring', edgecolor='none')
			ax.set_xlabel("x")
			ax.set_ylabel("y")
			ax.set_zlabel("U(x,y,0,0)")
			ax.set_title("x-y plane")

			ax = fig_2.add_subplot(1, 3, 2, projection = "3d" )
			ax.plot_surface( XonY, ZonY, U_calc_0[:, int(0.5 * grid_res), :], cmap = 'spring', edgecolor='none')
			ax.set_xlabel("x")
			ax.set_ylabel("z")
			ax.set_zlabel("U(x,0,z,0)")
			ax.set_title("x-z plane")

			ax = fig_2.add_subplot(1, 3, 3, projection = "3d" )
			ax.plot_surface( YonX, ZonX, U_calc_0[int(0.5 * grid_res), :, :], cmap = 'spring', edgecolor="none")
			ax.set_xlabel("y")
			ax.set_ylabel("z")
			ax.set_zlabel("U(0,y,z,0)")
			ax.set_title("y-z plane")



			print("Done.\n")

		print("Plotting solutions: ", end = "")
		plt.show()

		print("Done.\n\nProgram terminating.\n")

#END 
