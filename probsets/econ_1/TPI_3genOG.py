# tpi in Chicago
import numpy as np




def get_cvec_L(rpath, wpath, nvec, b2):
    b_s = b2
    b_sp1 = np.append(b2, [0])
    cvecL = (1 + rpath) * b_s + wpath * nvec - b_sp1
    return cvecL


def get_MUc(cvec, sigma):
	MU_c = cvec ** (-sigma)
	return MU_c





def EulerFT(bvec, *args):
    b2, b3 = bvec
    nvec, rpath, wpath, beta, sigma = args
    r2, r3 = rpath
    w1, w2, w3 = wpath
    c1 = w1 * nvec[0] - b2
    c2 = (1 + r2) * b2 + w2 * nvec[1] - b3
    c3 = (1 + r3) * b3 + w3 * nvec[2]
    MU_c1 = c1 ** (-sigma)
    MU_c2 = c2 ** (-sigma)
    MU_c3 = c3 ** (-sigma)
    error1 = MU_c1 - beta * (1 + r2) * MU_c2
    error2 = MU_c2 - beta * (1 + r3) * MU_c3
    errors = np.array([error1, error2])

    return errors


def EulerL(b3, *args):
    b2, nvec, rpath, wpath, beta, sigma = args
    n2, n3 = nvec
    r1, r2 = rpath
    w1, w2 = wpath
    cvecL = get_cvec_L(rpath, wpath, nvec, b2)
    MU_cvec = cvecL ** (-sigma)
    error = MU_cvec[0] - beta * (1 + r2) * MU_cvec[1]
    return error