# Ivy Weber, 2019-2020

# Calculate maximum time-steps of DG-Lagrange method for different choices of nodal points and different values of beta

# <Add outline of approach here>

#Import required modules

import math as m
import cmath as cm
import numpy as np
import random as rn
import matplotlib.pyplot as plt

#Import any custom modules

import polynomial as p
import fem_polys as fem
import least_squares_regr as ls

#Define functions and classes

def contract_mat(mat_1, mat_2, a):

    e = cm.exp(2*1j*a*m.pi)

    mat_2_t = np.transpose(mat_2)

    mat_out = e * mat_2
    mat_out += (1/e) * mat_2_t
    mat_out += mat_1

    return mat_out


#MAIN:

if __name__ == "__main__":

	#Run code over various degrees of polynomial (note: 2 q_hermite + 1 = q_lagrange)
	#Use code to find maximum eigenvalues of M^-1 A for different values of q

	max_deg = 18
	N_cells = 160
	alpha_vals = np.arange(0, 1.0, 1.0/N_cells)
	
	eig_tol = m.pow(10, -9)
	counter_max = 1000

#	print("Enter a penalty parameter beta to test:", end = " ")
#	beta_dg = float(input()) #200 # np.linspace(20, 400, 20)

	tstep_elag_dg = np.zeros( max_deg )
	eff_elag_dg = np.zeros( max_deg )

	tstep_lag_dg = np.zeros( max_deg )
	eff_lag_dg = np.zeros( max_deg )

	herm_test = (max_deg + 1)/2

	tstep_herm_dg = np.zeros( m.floor(herm_test) )
	eff_herm_dg = np.zeros( m.floor(herm_test) )

	for deg in range(1, max_deg + 1):

		beta_dg = 2 * deg * (deg + 1)

		print("p = " + str(deg))

		M_1_elag_dg = fem.mass_1(deg, "lagrange_even_dg")
		B_elag_dg_temp = np.linalg.inv( M_1_elag_dg )

		A_1_elag_dg = - fem.laplace_1(deg, "lagrange_even_dg", beta = beta_dg)
		A_2_elag_dg = - fem.laplace_2(deg, "lagrange_even_dg", beta = beta_dg)

		lam_elag_dg_max = 0

		gl_pts = fem.lagrange.GL_points(deg)

		M_1_lag_dg = fem.mass_1(deg, "lagrange_dg", gl_pts)
		B_lag_dg_temp = np.linalg.inv( M_1_lag_dg )

		A_1_lag_dg = - fem.laplace_1(deg, "lagrange_dg", gl_pts, beta_dg)
		A_2_lag_dg = - fem.laplace_2(deg, "lagrange_dg", gl_pts, beta_dg)

		lam_lag_dg_max = 0

		#Test if it's necessary to compute Hermite polynomials
		q_hermite = (deg - 1)/2
		herm = (q_hermite == m.floor(q_hermite))

		if herm:

			q_hermite = int(q_hermite)
			herm_matrix = fem.hermite.hermite_gen_mat(q_hermite)

			M_1_herm_dg = fem.mass_1(q_hermite, "hermite_dg", herm_matrix)
			B_herm_dg_temp = np.linalg.inv( M_1_herm_dg )

			A_1_herm_dg = - fem.laplace_1(q_hermite, "hermite_dg", herm_matrix, beta_dg)
			A_2_herm_dg = - fem.laplace_2(q_hermite, "hermite_dg", herm_matrix, beta_dg)

			lam_herm_dg_max = 0


		for a in alpha_vals:

			A = contract_mat(A_1_elag_dg, A_2_elag_dg, a)
			B_elag_dg = B_elag_dg_temp @ A

			A = contract_mat(A_1_lag_dg, A_2_lag_dg, a)
			B_lag_dg = B_lag_dg_temp @ A

			if herm:
				A = contract_mat(A_1_herm_dg, A_2_herm_dg, a)
				B_herm_dg = B_herm_dg_temp @ A

			else:
				ind_herm = 1




			for position in range( 0, min(deg, 2) ):
				#Use the power method on B_elag_dg, B_lag_dg and B_herm_dg (if necessary), and
				#select the required eigenvalue

				x_k_elag_dg = np.zeros( deg + 1 )
				x_k_elag_dg[position] = 1
				lam_elag_dg = B_elag_dg[position, position]

				x_k_lag_dg = np.zeros( deg + 1 )
				x_k_lag_dg[position] = 1
				lam_lag_dg = B_lag_dg[position, position]

				if herm:
					x_k_herm_dg = np.zeros( deg + 1)
					x_k_herm_dg[position] = 1
					lam_herm_dg = B_herm_dg[position, position]

					ind_herm = 0

				ind_elag = 0
				ind_lag = 0

				counter = 0

				while (counter < counter_max) and ((ind_elag == 0) or (ind_lag == 0) or (ind_herm == 0)):

					if ind_elag == 0:

						x_k_elag_dg = np.dot(B_elag_dg, x_k_elag_dg)
						norm = abs(np.dot(x_k_elag_dg.conjugate(), x_k_elag_dg))
					
						if norm == 0:
							ind_elag = 1
						else:

							x_k_elag_dg = (1.0/m.sqrt(norm)) * x_k_elag_dg
		
							lam_elag_dg_new = np.dot(x_k_elag_dg.conjugate(), np.dot(B_elag_dg, x_k_elag_dg))
	
							if abs(lam_elag_dg_new - lam_elag_dg) <= eig_tol:
								ind_elag = 1
							else:
								lam_elag_dg = lam_elag_dg_new


					if ind_lag == 0:

						x_k_lag_dg = np.dot(B_lag_dg, x_k_lag_dg)
						norm = abs(np.dot(x_k_lag_dg.conjugate(), x_k_lag_dg))

						if norm == 0:
							ind_lag = 1
						else:

							x_k_lag_dg = (1.0/m.sqrt(norm)) * x_k_lag_dg
		
							lam_lag_dg_new = np.dot(x_k_lag_dg.conjugate(), np.dot(B_lag_dg, x_k_lag_dg))
	
							if abs(lam_lag_dg_new - lam_lag_dg) <= eig_tol:
								ind_lag_dg = 1
							else:
								lam_lag_dg = lam_lag_dg_new

			
					if ind_herm == 0:

						x_k_herm_dg = np.dot(B_herm_dg, x_k_herm_dg)
						norm = abs(np.dot(x_k_herm_dg.conjugate(), x_k_herm_dg))
					
						if norm == 0:
							ind_herm = 1
						else:

							x_k_herm_dg = (1.0/m.sqrt(norm)) * x_k_herm_dg
	
							lam_herm_dg_new = np.dot(x_k_herm_dg.conjugate(), np.dot(B_herm_dg, x_k_herm_dg))
	
							if abs(lam_herm_dg_new - lam_herm_dg) <= eig_tol:
								ind_herm = 1
							else:
								lam_herm_dg = lam_herm_dg_new

					counter += 1

				lam_elag_dg_max = max(lam_elag_dg, lam_elag_dg_max)
				lam_lag_dg_max = max(lam_lag_dg, lam_lag_dg_max)

				if herm:
					lam_herm_dg_max = max(lam_herm_dg, lam_herm_dg_max)


		tstep_elag_dg[deg - 1] = 1.0/(2 * m.sqrt(lam_elag_dg_max.real))
		eff_elag_dg[deg - 1] = (deg + 1) * tstep_elag_dg[deg - 1]

		tstep_lag_dg[deg - 1] = 1.0/(2 * m.sqrt(lam_lag_dg_max.real))
		eff_lag_dg[deg - 1] = (deg + 1) * tstep_lag_dg[deg - 1]

		if herm:
			tstep_herm_dg[q_hermite] = 1.0/m.sqrt(lam_herm_dg_max.real)
			eff_herm_dg[q_hermite] = (deg + 1) * tstep_herm_dg[q_hermite]

	print("All degrees tested")

	#Should now have tstep_elag, tstep_lag and tstep_herm, representing maximum
	#stable time-steps for both Lagrange methods (even and GL) and the Hermite 
	#method for different degrees of polynomial

	xaxis_lag = range(1, max_deg + 1)
	xaxis_herm = range(1, 2 * m.floor((max_deg + 1)/2),2)

	eff_elag_dg *= m.sqrt(12)
	eff_lag_dg *= m.sqrt(12)
	eff_herm_dg *= m.sqrt(12)

	trend_1 = np.zeros( max_deg )
	trend_2 = np.zeros( max_deg )
	trend_3 = np.zeros( max_deg )

	for ix in range( 0, max_deg ):
		trend_1[ix] = 1.0 / (0.9640 - 0.0036 * (ix + 1) )
		trend_2[ix] = 1.0 / (0.7316 + 0.1733 * (ix + 1) )
		trend_3[ix] = 1.0 / (0.7378 + 0.6752 * (ix + 1) )

	output = open("tstep_dg_max.csv", "w")

	output.write("Poly. deg. p,C_{eff} (L-even-DG),C_{eff} (L-GL-DG),C_{eff} (Herm.-DG)\n")

	for io in range(1, max_deg + 1):

		htest = ( m.floor(io/2) != m.floor((io + 1)/2) )

		output.write( str(io) + "," + str(eff_elag_dg[io-1]) + "," + str(eff_lag_dg[io-1]) )

		if htest:
			output.write( "," + str(eff_herm_dg[int(io/2)]) )
		else:
			output.write( ", " )

		output.write( "\n" )

	output.close()


	exit = bool(0)

	while not exit:

		print("Enter 0 for normal graphs, 1 for semi-log graphs, 2 for log-log graphs, 3 for inverse graphs:", end = " ")
		gr_choice = int(input())

		if gr_choice == 0:

			plt.plot(xaxis_lag, eff_elag_dg, 'xb')
			plt.plot(xaxis_lag, eff_lag_dg, '+g')
			plt.plot(xaxis_herm, eff_herm_dg, 'or')

			plt.plot(xaxis_lag, trend_1, '--k')
			plt.plot(xaxis_lag, trend_2, ':k')
			plt.plot(xaxis_lag, trend_3, '-.k')


		elif gr_choice == 1:

			plt.semilogy(xaxis_lag, eff_elag_dg, 'xb')
			plt.semilogy(xaxis_lag, eff_lag_dg, '+g')
			plt.semilogy(xaxis_herm, eff_herm_dg, 'or')

			plt.semilogy(xaxis_lag, trend_1, '--k')
			plt.semilogy(xaxis_lag, trend_2, ':k')
			plt.semilogy(xaxis_lag, trend_3, '-.k')


		elif gr_choice == 2:

			plt.loglog(xaxis_lag, eff_elag_dg, 'xb')
			plt.loglog(xaxis_lag, eff_lag_dg, '+g')
			plt.loglog(xaxis_herm, eff_herm_dg, 'or')

			plt.loglog(xaxis_lag, trend_1, '--k')
			plt.loglog(xaxis_lag, trend_2, ':k')
			plt.loglog(xaxis_lag, trend_3, '-.k')


		else:

			inv_elag = np.reciprocal(eff_elag_dg)
			inv_lag = np.reciprocal(eff_lag_dg)
			inv_herm = np.reciprocal(eff_herm_dg)

			plt.plot(xaxis_lag, inv_elag, 'xb')
			plt.plot(xaxis_lag, inv_lag, '+g')
			plt.plot(xaxis_herm, inv_herm, 'or')

			plt.plot(xaxis_lag, 1.0 / trend_1, '--k')
			plt.plot(xaxis_lag, 1.0 / trend_2, ':k')
			plt.plot(xaxis_lag, 1.0 / trend_3, '-.k')



		plt.xlabel("Polynomial degree")

		plt.xticks(range(1, max_deg + 1), range(1, max_deg + 1))

		if gr_choice != 3:
			plt.ylabel("Rescaled C_eff")
		else:
			plt.ylabel("Reciprocal of C_eff")

		plt.show()
	
		print("Enter X to exit.")
		ans = input()

		if ans == "X" or ans == "x":
			exit = bool(1)


#END 
