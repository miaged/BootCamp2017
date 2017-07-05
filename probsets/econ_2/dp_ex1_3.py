# ex 3 
import numpy as np
import matplotlib.pyplot as plt

beta = 0.96
w = np.array([0.5, 1.0, 1.5])
p = np.array([0.2, 0.4, 0,4])
c_vals = np.linspace(1, 2, 100)
x = np.ones(len(w))
x_new = np.empty_like(x) 




def get_xnew(beta, c_vals, w, x, p):
	beta = 0.96
	x = np.ones(len(w))
	w = np.array([0.5, 1.0, 1.5])
	p = np.array([0.2, 0.4, 0,4])
	c_vals = np.linspace(1, 2, 100)
	for c in c_vals:
		x_new = c * (1 - beta) + beta * np.sum(np.maximum(w , x) * p)
	return x_new




print(x_new)
plt.plot(x_new)
plt.show()





