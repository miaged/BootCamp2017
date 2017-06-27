'''
Computation lab 1, 
exercises NumpyIntro

'''

import numpy as np 


# exercise 1
# define matrix

A = np.array([[3, -1, 4], [1, 5, -9]])
B = np.array([[2, 6, -5, 3], [5, -8, 9, 7], [9, -3, -2, -3]])

# def function to get matrix multiplication
def get_matmul(A,B):

	y = np.dot(A,B)
	return y


print(get_matmul(A,B))


# exercise 2



def get_eq1():
	C = np.array([[3, 1, 4], [1, 5, 9], [-5, 3, 1]])
	D = get_matmul(C,C)
	eq1 = - get_matmul(D,C) + 9 * D - 15 * C
	return eq1

print(get_eq1())


# exercise 3


def get_triu(x):
	triu = np.triu(x)
	return triu

def get_tril(x):
	tril = np.tril(x)
	return tril


M = get_triu(np.ones((7, 7)))
N = get_tril(np.full((7, 7), -6))+5


print(M)
print(N)

MN = get_matmul(M, N)
MNM = get_matmul(MN,M).astype(np.int64)

print(MNM)


# exercise 4 
z = np.arange(0, 30, 5)
x = np.copy(z)
mask = x > 0
x[mask] = - x[mask]
print(x)


# exercise 5

W = np.arange(6).reshape((3,2))
W1 = W.T
T = np.full((3,3), 3)
T1 = np.tril(T)
J = np.diag([-2, -2, -2])
Z1 = np.zeros((3, 3))
Z2 = np.zeros((2, 2))
Z3 = np.zeros((2, 3))
Z4 = np.zeros((3, 2))

W2 = W1.T
I1 =  np.eye(3)
YY1 = np.hstack((Z1,W2,I1))
YY2 = np.hstack((W1,Z2,Z3))
YY3 = np.hstack((T1,Z4,J))

YYfinal = np.vstack((YY1, YY2, YY3))

print(W1)
print(T1)
print(J)
print(Z1)
print(W2)
print(YYfinal)



# exercise 6

H1  = np.arange(9).reshape((3,3))
v = H1.sum(axis=1)
vt = v.reshape((3,1))
H3 = H1 / vt
check = H3.sum(axis=1)

print(H1)
print(vt)
print(H3)
print(check)


# exercise 7

grid = np.load("grid.npy")
#print(grid)

# horizontal
grid[:,:-3]
grid[:,1:-2]
grid[:,2:-1]
grid[:,3:]
print(np.max(grid[:,:-3] * grid[:,1:-2] * grid[:,2:-1] * grid[:,3:]))



# vertical 
grid[:-3,:]
grid[1:-2,:]
grid[2:-1,:] 
grid[3:,:]
print(np.max(grid[:-3,:] * grid[1:-2,:] * grid[2:-1,:] * grid[3:,:]))

# diag 1
grid[:-3,3:]
grid[1:-2,2:-1]
grid[2:-1,1:-2] 
grid[3:,:-3]
print(np.max(grid[:-3,3:]* grid[1:-2,2:-1] * grid[2:-1,1:-2]* grid[3:,:-3]))

# diag 2

grid[:-3,:-3]
grid[1:-2,1:-2]
grid[2:-1,2:-1] 
grid[3:,3:]
print(np.max(grid[:-3,:-3] * grid[1:-2,1:-2] * grid[2:-1,2:-1] * grid[3:,3:]))




