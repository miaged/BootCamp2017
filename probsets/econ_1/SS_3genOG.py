
import numpy as  np
import scipy.optimize as opt


# def functions

def get_L(nvec):
	L = nvec.sum()
	return L

def get_K(bvec):
	K = bvec.sum()
	return K

def get_w(K, L, params):
	alpha, A = params
	w = (1 - alpha) * A * ((K / L) ** alpha)
	return w

def get_r(K, L, params):
	alpha, A, delta = params
	r = alpha * A * ((L / K) ** (1 - alpha)) - delta
	return r

def get_cvec(r, w, bvec, nvec):
    b_s = np.append([0], bvec)
    b_sp1 = np.append(bvec, [0])
    cvec = (1 + r) * b_s + w * nvec - b_sp1
    return cvec



def MU_c(cvec, sigma):
	MU_c = cvec ** (-sigma)
	return MU_c


def EulerF(bvec, *args):
	b2, b3 = bvec
	nvec, beta, sigma, alpha, A, delta= args
	K = get_K(bvec)
	L = get_L(nvec)
	r_params = alpha, A, delta
	r = get_r(K, L, r_params)
	w_params = alpha, A
	w = get_w(K, L, w_params)
	cvec = get_cvec(r, w, bvec, nvec)
	MU_cvec = cvec ** (-sigma)
	error1 = MU_cvec[0] - beta * (1 + r) * MU_cvec[1]
	error2 = MU_cvec[1] - beta * (1 + r) * MU_cvec[2]
	errors = np.array([error1, error2])
	return errors





