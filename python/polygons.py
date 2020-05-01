import numpy as np
import dmsh
import meshing
import csg2d as csg

'''
Computes the area of a polygon.
Notice that the polygon must be repeating (i.e. poly[0] == poly[-1]).
'''
def compute_polygon_area(poly):
    A = 0.0
    for i in range(poly.shape[0]-1):
        a = poly[i,:]
        b = poly[i+1,:]

        A += a[0] * b[1] - b[0] * a[1]
    A = abs(A / 2.0)
    return A

'''
Computes normals of a polygon.
'''
def compute_polygon_normals(poly):
    normals = []
    theta = -np.pi/2
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    for i in range(poly.shape[0] - 1):
        a = poly[i,:]
        b = poly[i+1,:]
        n = np.dot((b - a) / np.linalg.norm((b - a)), R)
        normals.append(n)
    return normals

'''
Computes the area of a proxy.
'''
def compute_proxy_area(proxy):
    if len(proxy) == 3:
        return proxy[0]**2 * np.pi
    else:
        return proxy[0] * proxy[1]

'''
Compute randomly distributed points within a proxy.
'''
def compute_random_points(proxy, num_samples=1e3):
    if len(proxy) == 3:
        r = proxy[0] * np.sqrt(np.random.rand(int(num_samples)))
        theta = np.random.rand(int(num_samples)) * 2.0 * np.pi
        points = np.zeros((int(num_samples), 2))
        points[:, 0] = proxy[1] + r * np.cos(theta)
        points[:, 1] = proxy[2] + r * np.sin(theta)
        return points
    else:
        sx = proxy[0]
        sy = proxy[1]
        
        cx = proxy[2]
        cy = proxy[3]
        
        theta = proxy[4]
        
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        
        min_coord = np.array([cx - sx * 0.5, cy - sy * 0.5])
        max_coord = np.array([cx + sx * 0.5, cy + sy * 0.5])
        s = max_coord - min_coord
        random_samples = np.random.rand(int(num_samples), 2) * s
        points = np.dot(random_samples, R)
        return points - np.mean(points, axis=0) + np.array([cx, cy])

'''
Computes uniformly spaced points within a proxy.
'''
def compute_uniformly_sampled_points(proxy, min_coord, max_coord, spacing = .5, grid_points = None ):
    if len(proxy) == 3:
        num_rows = int(np.ceil((max_coord[0] - min_coord[0]) /  spacing))
        num_cols  = int(np.ceil((max_coord[1] - min_coord[1]) /  spacing))
        
        if grid_points is None:
            grid_points = compute_grid_points(min_coord, max_coord, spacing)
        
        points = grid_points.reshape((num_rows, num_cols, 2))
        
        X = np.linspace(min_coord[0], max_coord[0], num_rows)
        Y = np.linspace(min_coord[1], max_coord[1], num_cols)

        Xv, Yv = np.meshgrid(X, Y)

        circle_mask = np.sqrt((Xv - spacing / 2. - proxy[1])**2 + (Yv-proxy[2])**2) <= proxy[0]
        
        return points[circle_mask.T]
    else:
        num_rows = int(np.ceil((max_coord[0] - min_coord[0]) /  spacing))
        num_cols  = int(np.ceil((max_coord[1] - min_coord[1]) /  spacing))
        
        if grid_points is None:
            grid_points = compute_grid_points(min_coord, max_coord, spacing)
        
        min_idx = np.floor((get_proxy_min_coord(proxy) + np.array([spacing, spacing]) - grid_points[0]) / spacing)
        max_idx = np.floor((get_proxy_max_coord(proxy) + np.array([spacing, spacing]) - grid_points[0]) / spacing)
        
        points = grid_points.reshape((num_rows, num_cols, 2))
        points = points[int(min_idx[0]):int(max_idx[0]), int(min_idx[1]):int(max_idx[1])]
        points = points.reshape((points.shape[0] * points.shape[1], 2))
        theta = proxy[4]
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        return np.dot(points - np.array([proxy[2], proxy[3]]), R) + np.array([proxy[2], proxy[3]])

'''
Finds the intersection point between lines (p1, p2) and (u1, u2).
Returns (bool, point), where the boolean indicates if there was an intersection or not.
'''
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


'''
Gets the minimum amount of points (to some degree) needed to perform a triangulation of the surface.
'''
def get_minimal_proxy_points(proxy):
    if len(proxy) == 3:
        r = proxy[0]
        cx = proxy[1]
        cy = proxy[2]
        circumference = 2 * np.pi
        th = np.array([[t for t in np.arange(0, circumference, 0.5)]])
        th = th.T
        pcircle = np.array([cx, cy]) + r * np.hstack((np.cos(th), np.sin(th)))
        return pcircle
    else:
        sx = proxy[0]
        sy = proxy[1]
        cx = proxy[2]
        cy = proxy[3]
        theta = proxy[4]
        pbox = np.array([[-0.5, -0.5], [-0.5,0.5], [0.5, 0.5], [0.5, -0.5]])
        S = np.diag([sx, sy])
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        pbox = np.dot(np.dot(pbox, S), R) + np.array([cx, cy])
        return pbox

'''
Returns the min coordinate of a proxy.
'''
def get_proxy_max_coord(proxy):
    if len(proxy) == 3:
        return np.array([proxy[0] + proxy[1], proxy[0] + proxy[2]])
    else:
        return np.array([proxy[0] / 2. + proxy[2], proxy[1] / 2. + proxy[3]])

'''
Triangulates a proxy.
'''
def get_proxy_mesh(proxy, edgesize = .5):
    proxypoints = get_minimal_proxy_points(proxy)
    vs, fs = dmsh.generate(dmsh.Polygon(proxypoints), edgesize)
    return vs, fs


'''
Returns the min coordinate of a proxy.
'''
def get_proxy_min_coord(proxy):
    if len(proxy) == 3:
        return np.array([-proxy[0] + proxy[1], -proxy[0] + proxy[2]])
    else:
        return np.array([-proxy[0] / 2. + proxy[2], -proxy[1] / 2. + proxy[3]])


def get_proxy_sdf(proxy):
    if len(proxy) == 3:
        proxysdf = csg.translate(csg.sphere(proxy[0]), np.array([proxy[1], proxy[2]]))
    else:
        R = np.array([[np.cos(proxy[4]), -np.sin(proxy[4])],
                      [np.sin(proxy[4]), np.cos(proxy[4])]])
        proxysdf = csg.translate(csg.rotate(csg.box(proxy[0]/2., proxy[1]/2.), R.T), np.array([proxy[2], proxy[3]]))    
    return proxysdf

'''
Returns true if the point p2 is left of the edge spanned by p0 and p1.
'''
def is_left(p0, p1, p2):
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]) > 0.0


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

'''
Refines a polygon to have maximum edgelength L.
'''
def refinePoly(poly, l):
    polynew = []
    for i in range(poly.shape[0]-1):
        a = poly[i,:]
        b = poly[i+1,:]
        L = np.linalg.norm(a - b)
        if l < L:
            N = int(np.ceil(L / l))
            if len(polynew) == 0:
                polynew = np.array(np.linspace(a, b, N))
            else:
                polynew = np.append(polynew, np.linspace(a, b, N)[1:], axis=0)
        else:
            if len(polynew) == 0:
                polynew = np.array([a]) 
            polynew = np.append(polynew, np.array([b]), axis=0)
    return polynew

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
Turns a proxy into a polygon.
'''
def unpack_proxy(proxy):
    if len(proxy) == 3:
        r = proxy[0]
        cx = proxy[1]
        cy = proxy[2]

        circumference = 2 * np.pi

        th = np.array([[t for t in np.arange(0, circumference, 0.05)]])
        th = th.T

        pcircle = np.array([cx, cy]) + r * np.hstack((np.cos(th), np.sin(th)))
        return pcircle
    else:
        sx = proxy[0]
        sy = proxy[1]
        cx = proxy[2]
        cy = proxy[3]
        theta = proxy[4]

        pbox = np.array([[-0.5, -0.5], [-0.5,0.5], [0.5, 0.5], [0.5, -0.5], [-0.5,-0.5]])

        S = np.diag([sx, sy])
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        pbox = np.dot(np.dot(pbox, S), R) + np.array([cx, cy])

        pbox = refinePoly(pbox, 0.25)

        return pbox

'''
Method for updating a triangle mesh withot having to re-triangulate.
Assumes that the mesh is centered at (0, 0) and has unit size.
'''
def update_proxy_mesh(proxy, vs, fs):
    if len(proxy) == 3:
        r = proxy[0]
        cx = proxy[1]
        cy = proxy[2]
        
        return vs * r + np.array([cx, cy]), fs
    else:
        sx = proxy[0]
        sy = proxy[1]
        cx = proxy[2]
        cy = proxy[3]
        theta = proxy[4]
        S = np.diag([sx, sy])
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        
        return np.dot(np.dot(vs, S), R) + np.array([cx, cy]), fs