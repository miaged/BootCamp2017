# DP , OSM lab

import numpy as np

#data 


A = [[0.6, 0.1, -0.3],
     [0.5, -0.4, 0.2],
     [1.0, -0.2, 1.1]]

b = [[12],
     [10],
     [-1]]



A, b = map(np.asarray, (A, b))


from scipy.linalg import eigvals, solve
evs = eigvals(A)
ρ = max(abs(evs))
print(ρ)

''' the marix A is convergent if and only if 
its spectral radius is strictly less than one: p(A) < 1. 
As p is 0.96 the the equation x = A x + b has a unique solution
'''

#Compute the solution using matrix algebra
x = np.linalg.solve(A, b)
print(x)


# compute the solution by successive approximations 


xt = np.array([[.1], [.1], [.1]])
xtp1 = np.empty_like(xt)

for i in range (100):
	xtp1 = np.dot(A, xt) + b
	if xtp1.all == xt.all:
		print(xtp1)
	else:
		i += 1 

print(xtp1)
