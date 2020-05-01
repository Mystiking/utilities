import numpy as np

# CSG operations using Signed Distance Fields
def sphere(radius):
    return lambda p: np.sqrt((np.dot(p,p))) - radius

def box(w, h):
    def corner_check_return(p):
        if p[0] > 0 and p[1] > 0:
            return np.sqrt(np.dot(p.T, p))
        else:
            return np.max(p)

    return lambda x: corner_check_return(np.array([np.abs(x[0])-w, np.abs(x[1])-h]))

def union(phi, psi):
    return lambda x: min(phi(x), psi(x))

def intersection(phi, psi):
    return lambda x: max(phi(x), psi(x))

def difference(phi, psi):
    return lambda x: max(phi(x), - psi(x))

def translate(phi, t):
    return lambda x: phi(x - np.array(t))

def scale(phi, s):
    return lambda x: s * phi(np.array(x) / s)

def rotate(phi, R):
    return lambda x: phi(np.dot(np.array(x), R))

def erosion(phi, delta):
    return lambda x: phi(x) - delta

def dilation(phi, delta):
    return erosion(phi, -delta)

def opening(phi, delta):
    return dilation(erosion(phi, delta), delta)
