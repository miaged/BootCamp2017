# 3 generations OG model

# import packages
import numpy as np
import SS_3genOG as ss
import TPI_3genOG as tpi
import scipy.optimize as opt
import matplotlib.pyplot as plt
import execute_ss as ex



# hh parameters
S = 3
years = 60

# alternative values for beta
beta_a = 0.96
beta = (beta_a) ** (years / S)
#beta = 0.55

sigma = 3.0



# firms parameters 

A = 1.0
alpha = 0.35
delta_a = 0.05
delta = 1 - ((1 - delta_a) ** (years / S))
nvec = np.array([1.0, 1.0, 0.2])




# results for ss

b_ss = ex.results_b.x
b2_ss, b3_ss = b_ss
K_ss = ss.get_K(b_ss)
L_ss = ss.get_L(nvec)


# TPI params 

T = 50
b1vec = np.array([0.8 * b_ss[0], 1.1 * b_ss[1]])
K1 = ss.get_K(b1vec)
Kpath_init = np.zeros(T + 1)
Kpath_init[:-1] = np.linspace(K1, K_ss, T)
Kpath_init[-1] = K_ss

dist = 10
mindist =  1e-09
maxiter = 500
tpi_iter = 0
xi = 0.2

r_params = (alpha, A, delta)
w_params = (alpha, A)

# TPI
while dist > mindist and tpi_iter < maxiter:
    tpi_iter += 1
    # Get r and w paths
    rpath = ss.get_r(Kpath_init, L_ss, r_params)
    wpath = ss.get_w(Kpath_init, L_ss, w_params)
    bpath = np.zeros((S - 1, T + 1))
    bpath[:, 0] = b1vec
    # Solve for the lone individual problem in period 1
    bl_init = b1vec[1]
    bl_args = (b1vec[0], nvec[1:], rpath[:2], wpath[:2], beta, sigma)
    results_bl = opt.root(tpi.EulerL, bl_init, args=(bl_args))
    bpath[1, 1] = results_bl.x
    for t in range(T - 1):
        bvec_init = np.array([bpath[0, t], bpath[1, t + 1]])
        b_args = (nvec, rpath[t + 1:t + 3], wpath[t:t + 3], beta, sigma)
        results_bt = opt.root(tpi.EulerFT, bvec_init, args=(b_args))
        b2, b3 = results_bt.x
        bpath[0, t + 1] = b2
        bpath[1, t + 2] = b3

    Kpath_new = bpath.sum(axis=0)
    dist = ((Kpath_init[:-1] - Kpath_new[:-1]) ** 2).sum()
    Kpath_init[:-1] = xi * Kpath_new[:-1] + (1 - xi) * Kpath_init[:-1]
    print('iter:', tpi_iter, ' dist: ', dist)

print(results_bt)


plt.plot(Kpath_new)
plt.show()
