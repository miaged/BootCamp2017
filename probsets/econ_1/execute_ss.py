# 3 generations OG model

# import packages
import numpy as np
import SS_3genOG as ss
import scipy.optimize as opt
'''

'''



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



# Root finder

b2_init = 0.1
b3_init = 0.1
bvec_init = np.array([b2_init, b3_init])
b_args = (nvec, beta, sigma, alpha, A, delta) 
results_b = opt.root(ss.EulerF, bvec_init, args=(b_args))

# print results

print(results_b)

b_ss = results_b.x
b2_ss, b3_ss = b_ss
print('SS savings: ', b_ss)
K_ss = ss.get_K(b_ss)
L_ss = ss.get_L(nvec)
print('K_ss and L_ss', np.array([K_ss, L_ss]))
r_params = (alpha, A, delta)
w_params = (alpha, A)
r_ss = ss.get_r(K_ss, L_ss, r_params)
w_ss = ss.get_w(K_ss, L_ss, w_params)
print('SS r and w: ', np.array([r_ss, w_ss]))
c1_ss = w_ss * nvec[0] - b2_ss
c2_ss = (1 + r_ss) * b2_ss + w_ss * nvec[1] - b3_ss
c3_ss = (1 + r_ss) * b3_ss + w_ss * nvec[2]
print('SS c1, c2, c3: ', np.array([c1_ss, c2_ss, c3_ss]))