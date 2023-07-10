import math as m
import numpy as np

x_min = -1.5
x_max = 1.5
y_min = x_min
y_max = x_max
z_min = x_min
z_max = x_max

MINS = (x_min, y_min, z_min)

syst_dims = 2

init_stretch = (1.0, 1.0, 0.5)


p = 5

M = 6
N = M
O = M

if syst_dims == 2:
	DIMS = (M, N)
elif syst_dims == 3:
	DIMS = (M, N, O)

dx = (x_max - x_min)/M
dy = (y_max - y_min)/N
dz = (z_max - z_min)/O

if syst_dims == 2:
	HERM_WGT = np.array((dx, dy))
	LAGR_WGT = np.array((0.5 * dx, 0.5 * dy))
elif syst_dims == 3:
	HERM_WGT = np.array((dx, dy, dz))
	LAGR_WGT = np.array((0.5 * dx, 0.5 * dy, 0.5 * dz))

quad_pts = 51
ax_len = 15

initial_conds_present = True
zero_flag = True

OPTIONS = ("lagrange_even", "lagrange", "hermite", "lagrange_even_dg", "lagrange_dg", "hermite_dg")
SOLVERS_PLOT = (("+c", "xr", "ob", "*y", "pg", "xk"), ("--c", ":r", ":b", "-y", "--g", ":k"))

#SOLVERS = ("lagrange", "lagrange_even", "hermite", "lagrange_dg", "lagrange_even_dg", "hermite_dg")

c_eff_1 = 0.6
c_eff = c_eff_1 / m.sqrt(3)	#gives c_eff for one spatial dimension

T = 0.5
