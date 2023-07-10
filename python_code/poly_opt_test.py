# Ben Weber, <completion date here>

# <Add description of program here>

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

#Define functions and classes

def contract_mat(mat_1, mat_2, a, mat_type = 0):

    e = cm.exp(2*1j*a*m.pi)

    mat_2_t = ((-1) ** mat_type) * np.transpose(mat_2)

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

	tstep_elag = np.zeros( max_deg )
	eff_elag = np.zeros( max_deg )

	tstep_lag = np.zeros( max_deg )
	eff_lag = np.zeros( max_deg )

	herm_test = (max_deg + 1)/2

	tstep_herm = np.zeros(m.floor(herm_test))
	eff_herm = np.zeros( m.floor(herm_test) )

	for deg in range(1, max_deg + 1):

		print("p = " + str(deg))
		scale = m.pow( 10, deg )

		M_1_elag = fem.mass_1(deg)
		M_2_elag = fem.mass_2(deg)

		A_1_elag = - fem.laplace_1(deg)
		A_2_elag = - fem.laplace_2(deg)

		lam_elag_max = 0

		gl_pts = fem.lagrange.GL_points(deg)

		M_1_lag = fem.mass_1(deg, "lagrange", gl_pts)
		M_2_lag = fem.mass_2(deg, "lagrange", gl_pts)
		
		A_1_lag = - fem.laplace_1(deg, "lagrange", gl_pts)
		A_2_lag = - fem.laplace_2(deg, "lagrange", gl_pts)

		lam_lag_max = 0

		#Test if it's necessary to compute Hermite polynomials
		q_hermite = (deg - 1)/2
		herm = (q_hermite == m.floor(q_hermite))

		if herm:

			q_hermite = m.floor(q_hermite)
			herm_matrix = fem.hermite.hermite_gen_mat(q_hermite)

			M_1_herm = fem.mass_1(q_hermite, "hermite", herm_matrix)
			M_2_herm = fem.mass_2(q_hermite, "hermite", herm_matrix)

			A_1_herm = - fem.laplace_1(q_hermite, "hermite", herm_matrix)
			A_2_herm = - fem.laplace_2(q_hermite, "hermite", herm_matrix)

			lam_herm_max = 0


		for a in alpha_vals:

			M = contract_mat(M_1_elag, M_2_elag, a)
			A = contract_mat(A_1_elag, A_2_elag, a)

			B_elag = np.linalg.inv(M)
			B_elag = B_elag @ A

			M = contract_mat(M_1_lag, M_2_lag, a)
			A = contract_mat(A_1_lag, A_2_lag, a)

			B_lag = np.linalg.inv(M)
			B_lag = B_lag @ A

			if herm:

				M = scale * contract_mat(M_1_herm, M_2_herm, a)
				A = scale * contract_mat(A_1_herm, A_2_herm, a)

				B_herm = np.linalg.inv(M)
				B_herm = B_herm @ A

			else:
				ind_herm = 1




			for position in range( 0, min(deg, 2) ):
				#Use the power method on B_elag, B_lag and B_herm (if necessary), and
				#select the required eigenvalue

				x_k_elag = np.zeros( deg )
				x_k_elag[position] = 1
				lam_elag = B_elag[position, position]	

				x_k_lag = np.zeros( deg )
				x_k_lag[position] = 1
				lam_lag = B_lag[position, position]

				if herm:
					x_k_herm = np.zeros( q_hermite + 1)
					x_k_herm[position] = 1
					lam_herm = B_herm[position, position]

					ind_herm = 0

				ind_elag = 0
				ind_lag = 0

				counter = 0

				while (counter < counter_max) and ((ind_elag == 0) or (ind_lag == 0) or (ind_herm == 0)):

					if ind_elag == 0:

						x_k_elag = np.dot(B_elag, x_k_elag)
						norm = abs(np.dot(x_k_elag.conjugate(), x_k_elag))
					
						if norm == 0:
							ind_elag = 1
						else:

							x_k_elag = (1.0/m.sqrt(norm)) * x_k_elag

		
							lam_elag_new = np.dot(x_k_elag.conjugate(), np.dot(B_elag, x_k_elag))
	
							if abs(lam_elag_new - lam_elag) <= eig_tol:
								ind_elag = 1
							else:
								lam_elag = lam_elag_new


					if ind_lag == 0:

						x_k_lag = np.dot(B_lag, x_k_lag)
						norm = abs(np.dot(x_k_lag.conjugate(), x_k_lag))
					
						if norm == 0:
							ind_lag = 1
						else:

							x_k_lag = (1.0/m.sqrt(norm)) * x_k_lag
		
							lam_lag_new = np.dot(x_k_lag.conjugate(), np.dot(B_lag, x_k_lag))
	
							if abs(lam_lag_new - lam_lag) <= eig_tol:
								ind_lag = 1
							else:
								lam_lag = lam_lag_new

				
					if ind_herm == 0:

						x_k_herm = np.dot(B_herm, x_k_herm)
						norm = abs(np.dot(x_k_herm.conjugate(), x_k_herm))
					
						if norm == 0:
							ind_herm = 1
						else:

							x_k_herm = (1.0/m.sqrt(norm)) * x_k_herm
	
							lam_herm_new = np.dot(x_k_herm.conjugate(), np.dot(B_herm, x_k_herm))
	
							if abs(lam_herm_new - lam_herm) <= eig_tol:
								ind_herm = 1
							else:
								lam_herm = lam_herm_new

					counter += 1

				lam_elag_max = max(lam_elag, lam_elag_max)
				lam_lag_max = max(lam_lag, lam_lag_max)

				if herm:
					lam_herm_max = max(lam_herm, lam_herm_max)

		tstep_elag[deg - 1] = 1.0/(2 * m.sqrt(lam_elag_max.real))
		eff_elag[deg - 1] = deg * tstep_elag[deg - 1]

		tstep_lag[deg - 1] = 1.0/(2 * m.sqrt(lam_lag_max.real))
		eff_lag[deg - 1] = deg * tstep_lag[deg - 1]

		if herm:
			tstep_herm[q_hermite] = 1.0/m.sqrt(lam_herm_max.real)
			eff_herm[q_hermite] = (q_hermite + 1) * tstep_herm[q_hermite]

	print("All degrees tested")

	#Should now have tstep_elag, tstep_lag and tstep_herm, representing maximum
	#stable time-steps for both Lagrange methods (even and GL) and the Hermite 
	#method for different degrees of polynomial

	xaxis_lag = range(1, max_deg + 1)
	xaxis_herm = range(1, 2 * m.floor((max_deg + 1)/2),2)

	eff_elag *= m.sqrt(12)
	eff_lag *= m.sqrt(12)
	eff_herm *= m.sqrt(12)

#	trend_1 = np.ones( max_deg )
	trend_1a = np.zeros( max_deg )
#	trend_2 = np.zeros( max_deg )
	trend_2a = np.zeros( max_deg )
#	trend_3 = np.zeros( max_deg )
	trend_3a = np.zeros( max_deg )

	for ix in range(1, max_deg + 1):
		trend_1a[ix - 1] = 1.0 / ( 0.964 - 0.0036 * ix )
#		trend_2[ix - 1] = 1.0 / ( 1 + 0.16 * (ix - 1))
		trend_2a[ix - 1] = 1.0 / (0.7316 + 0.1733 * ix )
#		trend_3[ix - 1] = 1.0 / ( 0.6447 + 0.679 * ix)
		trend_3a[ix - 1] = 1.0 / (0.7378 + 0.6752 * ix )

	output = open("tstep_max.csv", "w")

	output.write("Poly. deg. p,C_{eff} (L-even),C_{eff} (L-GL),C_{eff} (Herm.)\n")

	for io in range(0, max_deg):

		itest = (m.floor(io/2) == m.floor((io + 1)/2))

		output.write(str(io + 1) + "," + str(eff_elag[io]) + "," + str(eff_lag[io]) + ",")

		if itest:

			output.write(str(eff_herm[int(io/2)]))

		output.write(" \n")


	output.close()
	

	exit = bool(0)

	while not exit:

		print("Enter 0 for normal graphs, 1 for semi-log graphs, 2 for log-log graphs, 3 for inverse graphs:", end = " ")
		gr_choice = int(input())

		if gr_choice == 0:

			plt.ylim([0, 1.7])

			plt.plot(xaxis_lag, eff_elag, 'xb')
			plt.plot(xaxis_lag, eff_lag, '+g')
			plt.plot(xaxis_herm, eff_herm, 'or')

			plt.plot(xaxis_lag, trend_1a, '--k')
			plt.plot(xaxis_lag, trend_2a, ':k')
			plt.plot(xaxis_lag, trend_3a, '-.k')


		elif gr_choice == 1:

			plt.semilogy(xaxis_lag, eff_elag, 'xb')
			plt.semilogy(xaxis_lag, eff_lag, '+g')
			plt.semilogy(xaxis_herm, eff_herm, 'or')

			plt.semilogy(xaxis_lag, trend_1a, '--k')
			plt.semilogy(xaxis_lag, trend_2a, ':k')
			plt.semilogy(xaxis_lag, trend_3a, '-.k')


		elif gr_choice == 2:

			plt.loglog(xaxis_lag, eff_elag, 'xb')
			plt.loglog(xaxis_lag, eff_lag, '+g')
			plt.loglog(xaxis_herm, eff_herm, 'or')

			plt.loglog(xaxis_lag, trend_1a, '--k')
			plt.loglog(xaxis_lag, trend_2a, ':k')
			plt.loglog(xaxis_lag, trend_3a, '-.k')


		else:

			inv_elag = np.reciprocal(eff_elag)
			inv_lag = np.reciprocal(eff_lag)
			inv_herm = np.reciprocal(eff_herm)

			plt.plot(xaxis_lag, inv_elag, 'xb')
			plt.plot(xaxis_lag, inv_lag, '+g')
			plt.plot(xaxis_herm, inv_herm, 'or')

			plt.plot(xaxis_lag, 1.0 / trend_1a, '--k')
			plt.plot(xaxis_lag, 1.0 / trend_2a, ':k')
			plt.plot(xaxis_lag, 1.0 / trend_3a, '-.k')


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
