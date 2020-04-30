import numpy as np
import meshing


def find_intersection_point(p1, p2, u1, u2):
    e1 = p2 - p1
    e2 = u2 - u1
    e1xe2 = np.sum(np.cross(e1, e2))
    upxe1 = np.sum(np.cross(u1 - p1, e1))
    
    out = None
    # Test for collinearity
    if e1xe2 == 0 and upxe1 == 0:        
        if (0 <= np.dot(u1 - p1, e1) and np.dot(u1 - p1, e1) <= np.dot(e1, e1)) or\
           (0 <= np.dot(u1 - p1, e2) and np.dot(u1 - p1, e2) <= np.dot(e2, e2)):
           return True, out
        return False, out
    # Test if parallel
    if e1xe2 == 0 and upxe1 != 0:
        return False, out

    # Try to find intersection point
    t = np.sum(np.cross(u1 - p1, e2)) / e1xe2
    u = np.sum(np.cross(u1 - p1, e1)) / e1xe2

    if e1xe2 != 0 and 0 <= t and t <= 1 and 0 <= u and u <= 1:
        out = p1 + t * e1
        return True, out
    # Otherwise the line segments doesn't intersect
    return False, out

def is_left(p0, p1, p2):
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]) > 0.0

def compute_polygon_area(poly):
    A = 0.0
    for i in range(poly.shape[0]-1):
        a = poly[i,:]
        b = poly[i+1,:]

        A += a[0] * b[1] - b[0] * a[1]
    A = abs(A / 2.0)
    return A

'''
An implementation of the Sutherland-Hodgman algorithm (https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm)
The main difference here is how the "inside" test is performed. This implementation therefore only works for triangles.
The algorithm finds the overlap between two triangles, and returns this as the set of points of "targetTri" w.r.t. "clipTri".
'''
def sutherland_hodgman_clipping_tri(clipTri, targetTri):
    resultPolygon = []
    A = np.abs(np.cross(clipTri[1] - clipTri[0], clipTri[2] - clipTri[0]))

    if A[0] > A[1]:
        if A[0] > A[2]: # A[0] is greatest
            i0 = 1
            i1 = 2
        else:           # A[2] is greatest
            i0 = 0
            i1 = 1
    else:               # A[0] <= A[1]
        if A[2] > A[1]: # A[2] is greatest
            i0 = 0
            i1 = 1
        else:           # A[1] is greatest
            i0 = 0
            i1 = 2

    for i in range(len(clipTri)):
        for j in range(len(targetTri)):
            currentPoint = targetTri[j]
            prevPoint    = targetTri[j - 1]

            # Find intersection between the two lines (if it exists)
            if meshing.point_in_tri(currentPoint, *clipTri, i0, i1):
                resultPolygon.append(currentPoint)
            
            intersection, intersectingPoint = find_intersection_point(prevPoint, currentPoint, clipTri[i], clipTri[i-1])
            if intersection:
                resultPolygon.append(intersectingPoint)

    return np.unique(np.array(resultPolygon), axis=0)

'''
2D-implementation of the Jarvis-March algorithm (1973) (otherwise known as the "Gift-Wrapping" algorithm).
Given a set of unordered points (in 2D), 
'''
def jarvis_march(S):
    most_left = [S[0]]
    for i in range(1, len(S)):
        if most_left[0][0] > S[i][0]:
            most_left[0] = S[i]
        elif most_left[0][0] == S[i][0]:
            most_left.append(S[i])

    if len(most_left) == 1:
        pointOnHull =  most_left[0] # left-most point on S
    else:
        most_left = np.array(most_left)
        pointOnHull = most_left[np.argsort(most_left[:, 1])[-1]]
    P = np.empty((0, 2)) # The final ordered polygon
    endPoint = None
    while len(P) == 0 or np.sum(np.abs(endPoint - P[0])) > 1e-5:
        P = np.append(P, [pointOnHull], axis=0)
        endPoint = S[0]
        for j in range(len(S)):
            if np.sum(np.abs(endPoint - pointOnHull)) < 1e-5 or is_left(pointOnHull, endPoint, S[j]):
                endPoint = S[j]
        pointOnHull = endPoint
    return np.append(P, [P[0]], axis=0)