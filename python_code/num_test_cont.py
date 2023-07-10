# Ben Weber
# Solve a test wave equation in 1D up to T = 0.5 to test stability of methods

# Use the function 1 - cos(2 pi (x - t)) on periodic domain [0,1], which has a known
# analytical solution. Use 20 cells with h = 0.05, and test various values of p for
# the Hermite and Lagrange methods, with various maximum time-steps. Impose initial
# conditions weakly for imporved generality

#Import required modules

import math as m
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu

import random as rn
import matplotlib.pyplot as plt

#Import any custom modules

import polynomial as p
import fem_polys as fem

#Define functions and classes

OPTIONS = ("lagrange_even", "lagrange", "hermite", "lagrange_even_dg", "lagrange_dg", "hermite_dg")
SOLVERS_PLOT = (("+c", "xr", "ob", "*y", "pg", "xk"), ("--c", ":r", ":b", "-y", "--g", ":k"))


def exact_sol(x,t):

	return 0.0 - m.cos( 2.0 * m.pi * (x - t))
#	return m.sqrt(0.5)

def exact_sol_t(x,t):

	return - 2.0 * m.pi * m.sin( 2.0 * m.pi * (x - t))
#	return 0

def x_cell(x, n, N, fem_type = "lagrange"):

	if (fem_type == "hermite") or (fem_type == "hermite_dg"):
		a = 0
		b = 1
	else:
		a = -1
		b = 1

	#Assume x lies in [a,b], and map it to cell n = [n/N, (n+1)/N]
	x_h = (x - a)/(b - a)
	x_h = (x_h + n)/N

	return x_h


def error_norm(U, t_curr, M, N, q, fem_type = "lagrange", quad_pts = 101, input_array = None):

	# e^2 = (u,u) + (U,U) - 2(u,U) = 1 - 2(u,U) + U^T M U
	e = float(np.dot(np.transpose(U), M.dot(U)))
	e += 0.5

	if (fem_type == "lagrange_even") or (fem_type == "lagrange"):
		q_err = q
	else:
		q_err = q - 1

	in_ar = fem.array_check( input_array, q_err, fem_type )

	for n in range(0,N):
		#Define the exact solution on cell n
		u = lambda x: exact_sol(x_cell(x, n, N, fem_type),t_curr)

		#Calculate cell n's contributions to b_0 and b_1
		if fem_type == "hermite":

			for i in range(0, q):

				poly_1 = fem.hermite(q_err, i, 0, 0, in_ar)
				poly_2 = fem.hermite(q_err, i, 1, 0, in_ar)

				p_1 = lambda x: poly_1.evaluate(x)
				p_2 = lambda x: poly_2.evaluate(x)

				b_1 = (1/N) * fem.inner_product_funct(u, p_1, fem_type, quad_pts)
				b_2 = (1/N) * fem.inner_product_funct(u, p_2, fem_type, quad_pts)

				e -= 2.0 * b_1 * U[n * q + i]
				e -= 2.0 * b_2 * U[((n + 1) % N) * q + i]


		elif fem_type == "lagrange":

			poly_p = fem.lagrange(q, q, 0, in_ar[:])
			p_p = lambda x: poly_p.evaluate(x)

			b = (1/(2.0 * N)) * fem.inner_product_funct(u, p_p, fem_type, quad_pts)

			e -= 2.0 * b * U[((n + 1) % N) * q]

			for i in range(0, q):

				poly_i = fem.lagrange(q, i, 0, in_ar[:])
				p_i = lambda x: poly_i.evaluate(x)
						
				b = (1/(2.0 * N)) * fem.inner_product_funct(u, p_i, fem_type, quad_pts)

				e -= 2.0 * b * U[ n * q + i]

		elif fem_type == "lagrange_dg":

			for i in range(0, q):

				poly_i = fem.lagrange(q_err, i, 0, in_ar[:])
				p_i = lambda x: poly_i.evaluate(x)

				b = (1/(2.0 * N)) * fem.inner_product_funct(u, p_i, fem_type, quad_pts)

				e -= 2.0 * b * U[ n * q + i]

		elif fem_type == "lagrange_even_dg":

			for i in range(0, q):

				poly_i = fem.lagrange_even(q_err, i, 0)
				p_i = lambda x: poly_i.evaluate(x)

				b = (1/(2.0 * N)) * fem.inner_product_funct(u, p_i, fem_type, quad_pts)

				e -= 2.0 * b * U[ n * q + i]

		elif fem_type == "hermite_dg":

			for side_ind in range(0,2):
				for i in range(0, q):

					poly_i = fem.hermite( q_err, i, side_ind, 0, in_ar )
					p_i = lambda x: poly_i.evaluate(x)

					b = (1/N) * fem.inner_product_funct(u, p_i, fem_type, quad_pts)

					e -= 2.0 * b * U[ (2 * n + side_ind) * q + i]

		else:

			poly_p = fem.lagrange_even(q, q, 0)
			p_p = lambda x: poly_p.evaluate(x)

			b = (1/(2.0 * N)) * fem.inner_product_funct(u, p_p, fem_type, quad_pts)

			e -= 2.0 * b * U[((n + 1) % N) * q]

			for i in range(0, q):

				poly_i = fem.lagrange_even(q, i, 0)
				p_i = lambda x: poly_i.evaluate(x)
						
				b = (1/(2.0 * N)) * fem.inner_product_funct(u, p_i, fem_type, quad_pts)

				e -= 2.0 * b * U[n * q + i]


	e = m.sqrt(abs(e))

	return e


def fem_interp( funct, fem_type, grid_size, space_size, quad_pts = 21, input_array = [] ):
	"""Initialise the system with b = (u_0, phi), where phi is the basis function corresponding to a given row"""

	if (fem_type == "lagrange_even") or (fem_type == "lagrange"):
		q_int = space_size
	else:
		q_int = space_size - 1

	array = fem.array_check( input_array, q_int, fem_type )

	if fem_type == "hermite_dg":
		b = np.zeros( (2 * grid_size * space_size, 1) )
	else:
		b = np.zeros( (grid_size * space_size, 1) )

	for n in range(0, grid_size):
		f = lambda x_int : funct(x_cell(x_int, n, grid_size, fem_type),0)

		#Calculate cell n's contributions to b
		if fem_type == "hermite":

			wgt = 1.0 / grid_size

			for i in range(0, space_size):

				pq_1 = fem.hermite(q_int, i, 0, 0, array)
				pq_2 = fem.hermite(q_int, i, 1, 0, array)

				p_1 = lambda x: pq_1.evaluate(x)
				p_2 = lambda x: pq_2.evaluate(x)

				b[n * space_size + i] += fem.inner_product_funct(f, p_1, fem_type, quad_pts)
				b[((n + 1) % grid_size) * space_size + i] += fem.inner_product_funct(f, p_2, fem_type, quad_pts)


		elif fem_type == "lagrange_dg":

			wgt = 0.5 / grid_size

			for i in range(0, space_size):

				pq_1 = fem.lagrange( q_int, i, 0, array[:] )
				pp_1 = lambda x: pq_1.evaluate(x)

				b[n * space_size + i] = fem.inner_product_funct(f, pp_1, fem_type, quad_pts)

		elif fem_type == "lagrange_even_dg":

			wgt = 0.5 / grid_size

			for i in range(0, space_size):

				pq_1 = fem.lagrange_even( q_int, i, 0 )
				pp_1 = lambda x: pq_1.evaluate(x)

				b[n * space_size + i] = fem.inner_product_funct(f, pp_1, fem_type, quad_pts)

		elif fem_type == "hermite_dg":

			wgt = 1.0 / grid_size

			for side_ind in range(0, 2):
				for i in range(0, space_size):

					pq_1 = fem.hermite( q_int, i, side_ind, 0, array )
					pp_1 = lambda x: pq_1.evaluate(x)

					b[ (2 * n + side_ind) * space_size + i ] = fem.inner_product_funct(f, pp_1, fem_type, quad_pts)

		else:

			wgt = 0.5 / grid_size

			if fem_type == "lagrange":
				pq_2 = fem.lagrange(q_int, space_size, 0, array[:])
			else:
				pq_2 = fem.lagrange_even(q_int, space_size, 0)

			p_2 = lambda x: pq_2.evaluate(x)

			b[((n + 1) % grid_size) * space_size] += fem.inner_product_funct(f, p_2, fem_type, quad_pts)

			for i in range(0, space_size):

				if fem_type == "lagrange":
					pq_1 = fem.lagrange(q_int, i, 0, array[:])
				else:
					pq_1 = fem.lagrange_even(q_int, i, 0)

				p_1 = lambda x: pq_1.evaluate(x)
						
				b[n * space_size + i] += fem.inner_product_funct(f, p_1, fem_type, quad_pts)


	return wgt * b


def plot_fem_sol( U_in, no_cells, sp_size, fem_type, input_array = [], ax_len = 100, axis = False, dom_size = 1.0 ):
	"""Return a plottable version of the finite element solution given by an input solution vector U_in"""

	if (fem_type == "lagrange_even") or (fem_type == "lagrange"):
		q_int = sp_size
	else:
		q_int = sp_size - 1

	array = fem.array_check( input_array, q_int, fem_type )
	dx = dom_size / no_cells

	x_axis = np.linspace(0, 1, ax_len)
	u_plot = np.zeros( (ax_len, 1) )
			
	for ix in range(0,ax_len):

		x_here = x_axis[ix]
		cell = m.floor(x_here/dx)

		if cell == no_cells:
			cell -= 1

		x_here = no_cells * (x_here - cell * dx)

		if fem_type == "hermite":

			for ip in range(0, sp_size):

				p_here = fem.hermite(q_int, ip, 0, 0, array)
				p_next = fem.hermite(q_int, ip, 1, 0, array)

				u_plot[ix, 0] += U_in[cell * sp_size + ip, 0] * p_here.evaluate(x_here)
				u_plot[ix, 0] += U_in[((cell + 1) % no_cells) * sp_size + ip, 0] * p_next.evaluate(x_here)

		elif fem_type == "lagrange_dg":

			x_here = 2 * x_here - 1

			for ip in range(0, sp_size):

				p_i = fem.lagrange(q_int, ip, 0, array[:])
				u_plot[ix, 0] += U_in[cell * sp_size + ip, 0] * p_i.evaluate(x_here)

		elif fem_type == "lagrange_even_dg":

			x_here = 2 * x_here - 1

			for ip in range(0, sp_size):

				p_i = fem.lagrange_even(q_int, ip, 0)
				u_plot[ix, 0] += U_in[cell * sp_size + ip, 0] * p_i.evaluate(x_here)

		elif fem_type == "hermite_dg":

			for side_ind in range(0,2):
				for ip in range(0, sp_size):

					p_i = fem.hermite(q_int, ip, side_ind, 0, array)
					u_plot[ix, 0] += U_in[ (2 * cell + side_ind) * sp_size + ip, 0] * p_i.evaluate(x_here)

		else:

			x_here = 2 * x_here - 1

			if fem_type == "lagrange":
				p_next = fem.lagrange(q_int, sp_size, 0, array[:])
			else:
				p_next = fem.lagrange_even(q_int, sp_size, 0)

			u_plot[ix, 0] += U_in[((cell + 1) % no_cells) * sp_size, 0] * p_next.evaluate(x_here)

			for ip in range(0, sp_size):

				if fem_type == "lagrange":
					p_here = fem.lagrange(q_int, ip, 0, array[:])
				else:
					p_here = fem.lagrange_even(q_int, ip, 0)

				u_plot[ix, 0] += U_in[cell * sp_size + ip, 0] * p_here.evaluate(x_here)

	if axis:
		return x_axis, u_plot
	else:
		return u_plot



		
#MAIN:

if __name__ == "__main__":

	x_min = 0
	x_max = 1

	N = 6

	quad_pts = 21
	quad_pts_start = 201

	c = 2.0/m.sqrt(12)

	print("\nEnter a minimum time T to solve up to:", end = " ")
	T = float(input())

	print("Enter a value of C_eff to test:", end = " ")	
	c_eff = float(input())

	print("Enter a polynomial degree (odd) to test:", end = " ")
	p = float(input())
	p = 1 + 2 * m.floor( (p - 1)/2 )

	beta = 2 * p * (p + 1)

	print("\nUsing p = " + str(p), end = "\n\n")

	cfl_lag = c_eff * c / p
	cfl_herm = 2 * c_eff * c / (p + 1)
	cfl_lag_dg = c_eff * c / (p + 1)

	dx = (x_max - x_min)/N

	dt_lag = cfl_lag * dx
	dt_herm = cfl_herm * dx
	dt_lag_dg = cfl_lag_dg * dx

	print("""Enter 0 to for Lagrange with evenly spaced nodes, 
1 for Lagrange with Gauss-Lobatto nodes,
2 for Hermite, 3 for L-even-DG, 4 for L-GL-DG, 
5 for Hermite-DG or any combination thereof:""", end = " ")
	choice = str(input())
	print("")

	solvers = []

	for ic in range(0, 6):
		if str(ic) in choice:
			solvers.append( OPTIONS[ic] )

	lsolv = len(solvers)

	N_t = m.floor(T/dt_lag) + 1
	N_t_herm = m.floor(T/dt_herm) + 1
	N_t_dg = m.floor(T/dt_lag_dg) + 1

	time_steps = np.zeros( (N_t_dg + 1, lsolv ) )
	err_max = np.zeros( (N_t_dg + 1, lsolv ) )

	axsize = 100

	for si in range(0, lsolv):

		s = solvers[si]
		print(s)

		discont = (s == "lagrange_even_dg") or (s == "lagrange_dg") or (s == "hermite_dg")

		if s == "hermite":

			q_1 = int((p - 1)/2)
			q_2 = q_1 + 1

			input_array = fem.hermite.hermite_gen_mat(q_1)

			w_1 = dx
			w_2 = N

			dt = dt_herm
			cfl = cfl_herm

			offset = q_2

		elif s == "lagrange_even_dg":

			q_1 = p
			q_2 = p + 1

			input_array = None

			w_1 = 0.5 * dx
			w_2 = 2.0 * N

			dt = dt_lag_dg
			cfl = cfl_lag_dg

			offset = q_2

		elif s == "lagrange_dg":

			q_1 = p
			q_2 = p + 1

			input_array = fem.lagrange.GL_points(q_1)

			w_1 = 0.5 * dx
			w_2 = 2.0 * N

			dt = dt_lag_dg
			cfl = cfl_lag_dg

			offset = q_2

		elif s == "hermite_dg":

			q_1 = int((p - 1)/2)
			q_2 = q_1 + 1

			input_array = fem.hermite.hermite_gen_mat(q_1)

			w_1 = dx
			w_2 = N

			dt = dt_lag_dg
			cfl = cfl_lag_dg

			offset = 2 * q_2

		else:
			q_1 = p
			q_2 = p

			if s == "lagrange":
				input_array = fem.lagrange.GL_points(q_1)
			else:
				input_array = None

			w_1 = 0.5 * dx
			w_2 = 2.0 * N

			dt = dt_lag
			cfl = cfl_lag

			offset = q_2


		M_1 = w_1 * fem.mass_1(q_1, s, input_array)

		if discont:

			I = sp.identity( N, format = 'csr' )
			M_1_inv = np.linalg.inv(M_1)

			M = sp.kron( I, M_1 )
			M_inv = sp.kron( I, M_1_inv)

		else:

			M_2 = w_1 * fem.mass_2(q_1, s, input_array)
			M = sp.lil_matrix( (N * q_2, N * q_2) )

		A_1 = w_2 * fem.laplace_1(q_1, s, input_array, beta)
		A_2 = w_2 * fem.laplace_2(q_1, s, input_array, beta)

		A = sp.lil_matrix( (N * offset, N * offset) )

		for n in range(0, N):

			if discont:

				start_1 = n * offset
				start_2 = ((n + 1) % N) * offset

				for i in range(0, offset):

					A[start_1 + i, start_1: start_1 + offset] = list(A_1[i,:])

					for j in range(0, offset):

						A[start_1 + i, start_2 + j] += A_2[i,j]
						A[start_2 + i, start_1 + j] += A_2[j,i]

			else:
				start_1 = n * q_2
				start_2 = ((n + 1) % N) * q_2

				#Add blocks to the sparse matrices
				for i in range(0, q_2): 

					M[start_1 + i, start_1: start_1 + q_2] = list(M_1[i,:])
					A[start_1 + i, start_1: start_1 + q_2] = list(A_1[i,:])

					for j in range(0, q_2):

						M[start_1 + i, start_2 + j] += M_2[i,j]
						M[start_2 + i, start_1 + j] += M_2[j,i]

						A[start_1 + i, start_2 + j] += A_2[i,j]
						A[start_2 + i, start_1 + j] += A_2[j,i]

		dense_test = (offset > N)

		if dense_test:

			M = M.todense()
			A = A.todense()

			if discont:
				M_inv = M_inv.todense()
			else:
				M_inv = np.linalg.inv(M)

			B = M_inv @ A

		else:

			M = M.tocsc()
			A = A.tocsr()

			if discont:
				M_inv = M_inv.tocsr()
			else:
				M_inv = splu(M)


		#Initialise the system with M U_0 = (u_0, phi), where phi is the basis function corresponding to a given row
		#of M
		b_0 = fem_interp( exact_sol, s, N, q_2, quad_pts_start, input_array )
		b_1 = fem_interp( exact_sol_t, s, N, q_2, quad_pts_start, input_array )

		if dense_test:
			U_0 = M_inv @ b_0
			U_t_0 = M_inv @ b_1
		elif discont:
			U_0 = M_inv.dot(b_0)
			U_t_0 = M_inv.dot(b_1)
		else:
			U_0 = M_inv.solve(b_0)
			U_t_0 = M_inv.solve(b_1)

		t = 0
		err_max[0, si] = m.sqrt(abs(0.5 - np.dot(b_0[:,0], U_0[:,0])))
		time_steps[0, si] = 0

		#Use the time-stepping method U_(n+1) = 2 U_n - U_(n-1) +dt^2 inv(M) A U_(n)
		if dense_test:
			U_1 = U_0 + dt * U_t_0 + 0.5 * (dt ** 2) * (B @ U_0)
		elif discont:
			U_1 = U_0 + dt * U_t_0 + 0.5 * (dt ** 2) * M_inv.dot( A.dot( U_0 ) )
		else:
			U_1 = U_0 + dt * U_t_0 + 0.5 * (dt ** 2) * M_inv.solve( A.dot( U_0 ) )

		t = dt

		err_max[1, si] = error_norm(U_1, t, M, N, q_2, s, quad_pts, input_array)
		time_steps[1, si] = dt

		U_prev = U_0
		U_curr = U_1
		count = 1

		#Rather than computing inv(M)A once, compute Ax, then solve My = Ax for y at every time-step
		while t < T:

			if dense_test:
				Y = B @ U_curr
			else:
				Y = A.dot(U_curr)
				if discont:
					Y = M_inv.dot(Y)
				else:
					Y = M_inv.solve(Y)


			Y *= dt**2

			Y = Y + 2 * U_curr - U_prev

			U_prev = U_curr[:]
			U_curr = Y[:]
			t += dt
			count += 1

			err_max[count, si] = error_norm(U_curr, t, M, N, q_2, s, quad_pts, input_array)
			time_steps[count, si] = t

		#Plot the exact solution and FEM solution for comparison

		x_axis = np.linspace(0, 1, axsize)
		u_true = np.zeros( (axsize,2) )

		for ix in range(0, axsize):

			u_true[ix,0] = exact_sol( x_axis[ix], t )
			u_true[ix,1] = exact_sol( x_axis[ix], 0 )

		u_calc_0 = plot_fem_sol( U_0, N, q_2, s, input_array, axsize )
		u_calc = plot_fem_sol( U_curr, N, q_2, s, input_array, axsize )

		plt.figure(1)

		plt.title("Solutions at t = " + str(round(t,4)) + " using degree " + str(p) + " polynomials and C_eff = " + str(c_eff))

		if si == 0:
			plt.plot( x_axis, u_true[:,0], ":k")
			leg_list_1 = ["Exact solution at t = " + str(round(t,4))]
			leg_list_2 = ["Exact solution at t = 0"]

		plt.plot( x_axis, u_calc, SOLVERS_PLOT[1][si])

		CFL = round(cfl, 4)

		leg_list_1.append( "FEM solution using " + s + " elements")
		leg_list_2.append( "FEM solution using " + s + " elements")

		plt.legend( leg_list_1 )

		plt.figure(2)

		plt.title("Initial solutions using degree " + str(p) + " polynomials")

		if si == 0:
			plt.plot( x_axis, u_true[:,1], ":k")

		plt.plot( x_axis, u_calc_0, SOLVERS_PLOT[1][si] )
		plt.legend( leg_list_2 )

	print("\nPlotting data...", end = "\n\n")

	plt.figure(3)
	si = 0

	if "lagrange_even" in solvers:

		Y_elag = err_max[:(N_t + 1), si]
		Ts_elag = time_steps[:(N_t + 1), si]

		plt.plot(Ts_elag[:], Y_elag[:], SOLVERS_PLOT[0][si])

		si += 1

	if "lagrange" in solvers:

		Y_lag = err_max[:(N_t + 1), si]
		Ts_lag = time_steps[:(N_t + 1), si]

		plt.plot(Ts_lag[:], Y_lag[:], SOLVERS_PLOT[0][si])

		si += 1

	if "hermite" in solvers:

		Y_herm = err_max[:(N_t_herm + 1), si]
		Ts_herm = time_steps[:(N_t_herm + 1), si]

		plt.plot(Ts_herm[:], Y_herm[:], SOLVERS_PLOT[0][si])

		si += 1

	if "lagrange_even_dg" in solvers:

		Y_edg = err_max[:, si]
		Ts_edg = time_steps[:, si]

		plt.plot(Ts_edg[:], Y_edg[:], SOLVERS_PLOT[0][si])

		si += 1

	if "lagrange_dg" in solvers:

		Y_dg = err_max[:, si]
		Ts_dg = time_steps[:, si]

		plt.plot(Ts_dg[:], Y_dg[:], SOLVERS_PLOT[0][si])

		si += 1

	if "hermite_dg" in solvers:

		Y_hdg = err_max[:, si]
		Ts_hdg = time_steps[:, si]

		plt.plot(Ts_hdg[:], Y_hdg[:], SOLVERS_PLOT[0][si])

	si = 0

	if "lagrange_even" in solvers:
		plt.plot(Ts_elag[:], Y_elag[:], SOLVERS_PLOT[1][si])
		si += 1

	if "lagrange" in solvers:
		plt.plot(Ts_lag[:], Y_lag[:], SOLVERS_PLOT[1][si])
		si += 1

	if "hermite" in solvers:
		plt.plot(Ts_herm[:], Y_herm[:], SOLVERS_PLOT[1][si])
		si += 1

	if "lagrange_even_dg" in solvers:
		plt.plot(Ts_edg[:], Y_edg[:], SOLVERS_PLOT[1][si])
		si += 1

	if "lagrange_dg" in solvers:
		plt.plot(Ts_dg[:], Y_dg[:], SOLVERS_PLOT[1][si])
		si += 1

	if "hermite_dg" in solvers:
		plt.plot(Ts_hdg[:], Y_hdg[:], SOLVERS_PLOT[1][si])

	plt.legend(solvers)

	plt.xlabel("Time t")
	plt.ylabel("Error in L2-norm")
	plt.title("Error over time using degree " + str(p) + " polynomials and C_eff = " + str(c_eff) )

	plt.xlim( (0,T) )

	plt.show()

#END 
