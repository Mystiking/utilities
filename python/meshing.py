import numpy as np
from scipy.spatial import Delaunay
import utilities.python.polygons as polygons

'''
Early-out AABB intersection testing in arbitrary dimension.
'''
def aabb_intersection(aabb0, aabb1):
    dims = int(aabb0.shape[0] / 2)
    for i in range(dims):
        local_intersection = aabb0[i] <= aabb1[i] and aabb1[i] <= aabb0[i + dims]
        local_intersection = local_intersection or (aabb0[i] <= aabb1[i + dims] and aabb1[i + dims] <= aabb0[i + dims])
        local_intersection = local_intersection or (aabb1[i] <= aabb0[i] and aabb0[i] <= aabb1[i + dims])
        if not local_intersection:
            return False
    return True

'''
Computes the polygon corresponding to the intersection between two triangle meshes (in 2D).
'''
def compute_intersection_polygon(vs0, vs1, fs0, fs1):
    # 1) Compute intersecting triangles
    intersecting_triangles = mesh_mesh_intersection_aabbs(vs0, vs1, fs0, fs1)
    # 2) Compute points of the final polygon
    polygonpoints = np.empty((0, 3))
    for intersection in intersecting_triangles:
        tri0 = [p for p in intersection[0]]
        tri1 = [p for p in intersection[1]]
        # Find the points of the overlapping polygon
        newpolygonpoints = polygons.sutherland_hodgman_clipping_tri(tri0, tri1)
        newpolygonpoints = np.unique(np.append(newpolygonpoints,
                                               polygons.sutherland_hodgman_clipping_tri(tri1, tri0),
                                               axis=0).round(decimals=5), axis=0)
        if newpolygonpoints.shape[0] > 2:
            polygonpoints = np.append(polygonpoints, newpolygonpoints, axis=0)
    # 3) Perform a Jarvis march to get the convex hull
    if len(polygonpoints) != 0:    
        polygonpoints = polygons.jarvis_march(polygonpoints[:, :-1])
    return polygonpoints

'''
Computes the projected intervals of a triangle onto a plane, as described in
"A Fast Triangle-Triangle Intersection Test" by Moeller (1997).
'''
def compute_intervals(vp0, vp1, vp2, dv0, dv1, dv2, dv0dv1, dv0dv2):
    if dv0dv1 > 0:
        a = vp2
        b = (vp0 - vp2) * dv2
        c = (vp1 - vp2) * dv2
        x0 = dv2 - dv0
        x1 = dv2 - dv1
        return a, b, c, x0, x1
    elif dv0dv2 > 0:
        a = vp1
        b = (vp0 - vp1) * dv1
        c = (vp2 - vp1) * dv1
        x0 = dv1 - dv0
        x1 = dv1 - dv2
        return a, b, c, x0, x1
    elif dv1 * dv2 > 0 or dv0 != 0:
        a = vp0
        b = (vp1 - vp0) * dv0
        c = (vp2 - vp0) * dv0
        x0 = dv0 - dv1
        x1 = dv0 - dv2
        return a, b, c, x0, x1
    elif dv1 != 0:
        a = vp1
        b = (vp0 - vp1) * dv1
        c = (vp2 - vp1) * dv1
        x0 = dv1 - dv0
        x1 = dv1 - dv2
        return a, b, c, x0, x1
    elif dv2 != 0:
        a = vp2
        b = (vp0 - vp2) * dv2
        c = (vp1 - vp2) * dv2
        x0 = dv2 - dv0
        x1 = dv2 - dv1
        return a, b, c, x0, x1
    else:
        return None

def compute_mesh_area(vs, fs):
    A = 0.0
    for f in fs:
        t = [vs[f[0]], vs[f[1]], vs[f[2]]]
        A += compute_triangle_area(t)
    return A

'''
Computes the area of a triangle using the cross product.
'''
def compute_triangle_area(tri):
    AB = tri[1] - tri[0]
    AC = tri[2] - tri[0]
    return np.linalg.norm(np.cross(AB, AC)) / 2.

'''
Performs a coplanarity test between two triangles.
'''
def coplanar_tri_tri(N, v0, v1, v2, u0, u1, u2):
    A = np.abs(N)

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
    # Test all edges of triangle 1 against the edges of triangle 2
    if edge_against_tri_edges(v0, v1, u0, u1, u2, i0, i1):
        return True
    if edge_against_tri_edges(v1, v2, u0, u1, u2, i0, i1):
        return True
    if edge_against_tri_edges(v2, v0, u0, u1, u2, i0, i1):
        return True
    # Test if triangle 1 is completely contained in triangle 2
    if point_in_tri(v0, u0, u1, u2, i0, i1):
        return True
    # Test if triangle 2 is completely contained in triangle 1
    if point_in_tri(u0, v0, v1, v2, i0, i1):
        return True
    return False



def intersection2d(polymesh, proxymesh, polytarget, proxypoly):
    zero_padded_polyvs = np.zeros((polymesh[0].shape[0], 3))
    zero_padded_polyvs[:, :-1] = polymesh[0][:, :2]
    zero_padded_proxyvs = np.zeros((proxymesh[0].shape[0], 3))
    zero_padded_proxyvs[:, :-1] = proxymesh[0][:, :2]
    intersecting_triangles = mesh_mesh_intersection_aabbs(zero_padded_polyvs,
                                                          zero_padded_proxyvs,
                                                          polymesh[1],
                                                          proxymesh[1])
    intersection_points = np.empty((0, 3))
    # All points inside
    for itri in intersecting_triangles:
        for p in itri[0]:
            if polygons.is_inside(proxypoly, p):
                intersection_points = np.append(intersection_points, [p], axis=0)
        for p in itri[1]:
            if polygons.is_inside(polytarget, p):
                intersection_points = np.append(intersection_points, [p], axis=0)
    # All points intersecting the two
    vertices = np.array(get_unique_vertices(intersection_points))
    tris = Delaunay(vertices, qhull_options='QJ')
    return vertices, tris.simplices.copy()


'''
Computes the mesh corresponding to polymesh `minus` proxymesh
'''
def difference(polymesh, proxymesh, proxypoly):
    zero_padded_polyvs = np.zeros((polymesh[0].shape[0], 3))
    zero_padded_polyvs[:, :-1] = polymesh[0][:, :2]
    zero_padded_proxyvs = np.zeros((proxymesh[0].shape[0], 3))
    zero_padded_proxyvs[:, :-1] = proxymesh[0][:, :2]
    polytriangles = np.array([[zero_padded_polyvs[i] for i in f] for f in polymesh[1]])
    intersecting_triangles = mesh_mesh_intersection_aabbs(zero_padded_polyvs,
                                                                  zero_padded_proxyvs,
                                                                  polymesh[1],
                                                                  proxymesh[1])
    polyintersectingtriangles = np.array([itri[0] for itri in intersecting_triangles])
    proxyintersectingtriangles = np.array([itri[1] for itri in intersecting_triangles])
    # Triangles in polymesh that aren't affected by the proxymesh
    diff = np.empty((0, 3, 3))
    for tri in polytriangles:
        add_triangle = True
        for itri in intersecting_triangles:
            if all([all([tri[i][j] == itri[0][i][j] for j in range(len(tri[i]))]) for i in range(len(tri))]):
                add_triangle = False
                break
        if add_triangle:
            diff = np.append(diff, [tri], axis=0)
    # Triangles that need to be fixed, due to an intersection with the polymesh
    for itri in intersecting_triangles:
        if not (all([polygons.is_inside(proxypoly, q) for q in itri[0]])):
            # Triangles that are not completely contained inside the proxy polygon
            corrected_points = polygons.compute_triangle_polygon_intersection_points(itri[0][:, :-1], proxypoly[:-2])
            for p in itri[0]:
                if not polygons.is_inside(proxypoly, p):
                    corrected_points = np.append(corrected_points, [p], axis=0)
            if corrected_points.shape[0] > 2:
                corrected_points = polygons.jarvis_march(corrected_points[:, :-1])
                try:
                    tris = Delaunay(corrected_points)
                    zero_padded_corrected_points = np.zeros((corrected_points.shape[0], 3))
                    zero_padded_corrected_points[:, :-1] = corrected_points
                    for f in tris.simplices.copy():
                        diff = np.append(diff, [[zero_padded_corrected_points[i] for i in f]], axis=0)
                except:
                    continue
    return reconstruct_mesh(diff) if diff.shape[0] > 0 else diff


def is_tri_in_list(tri, tris):
    for t in tris:
        if (all(t[0] == tri[0]) and all(t[1] == tri[1]) and all(t[2] == tri[2])) or\
           (all(t[0] == tri[0]) and all(t[2] == tri[1]) and all(t[1] == tri[2])) or \
           (all(t[1] == tri[0]) and all(t[0] == tri[1]) and all(t[2] == tri[2])) or \
           (all(t[1] == tri[0]) and all(t[2] == tri[1]) and all(t[0] == tri[2])) or \
           (all(t[2] == tri[0]) and all(t[0] == tri[1]) and all(t[1] == tri[2])) or \
           (all(t[2] == tri[0]) and all(t[1] == tri[1]) and all(t[0] == tri[2])):
           return True
    return False


def union(polymesh, proxymesh, polytarget, proxypoly):
    zero_padded_polyvs = np.zeros((polymesh[0].shape[0], 3))
    zero_padded_polyvs[:, :-1] = polymesh[0][:, :2]
    zero_padded_proxyvs = np.zeros((proxymesh[0].shape[0], 3))
    zero_padded_proxyvs[:, :-1] = proxymesh[0][:, :2]
    polytriangles = np.array([[zero_padded_polyvs[i] for i in f] for f in polymesh[1]])
    proxytriangles = np.array([[zero_padded_proxyvs[i] for i in f] for f in proxymesh[1]])
    intersecting_triangles = mesh_mesh_intersection_aabbs(zero_padded_polyvs,
                                                          zero_padded_proxyvs,
                                                          polymesh[1],
                                                          proxymesh[1])
    polyintersectingtriangles = np.array([itri[0] for itri in intersecting_triangles])
    proxyintersectingtriangles = np.array([itri[1] for itri in intersecting_triangles])
    result_tris = np.empty((0, 3, 3))
    # All tris in poly which are not in the proxy
    for tri in polytriangles:
        #if all([not polygons.is_inside(proxypoly, q) for q in tri]) and not is_tri_in_list(tri, polyintersectingtriangles):
        result_tris = np.append(result_tris, [tri], axis=0)
        '''
        if any([not polygons.is_inside(proxypoly, q) for q in tri]) or is_tri_in_list(tri, polyintersectingtriangles):
            corrected_points = polygons.compute_triangle_polygon_intersection_points(tri[:, :-1], proxypoly[:-2])
            for p in tri:
                if not polygons.is_inside(proxypoly, p):
                    corrected_points = np.append(corrected_points, [p], axis=0)
            if corrected_points.shape[0] > 2:
                corrected_points = polygons.jarvis_march(corrected_points[:, :-1])
                try:
                    tris = Delaunay(corrected_points)
                    zero_padded_corrected_points = np.zeros((corrected_points.shape[0], 3))
                    zero_padded_corrected_points[:, :-1] = corrected_points
                    for f in tris.simplices.copy():
                        result_tris = np.append(result_tris, [[zero_padded_corrected_points[i] for i in f]], axis=0)
                        #pass
                except:
                    continue
        '''
    diffmesh = difference(proxymesh, polymesh, np.flip(polytarget, 0))
    if len(diffmesh) > 0:
        for f in diffmesh[1]:
            result_tris = np.append(result_tris, [[diffmesh[0][f[0]], diffmesh[0][f[1]], diffmesh[0][f[2]]]], axis=0)



    return reconstruct_mesh(result_tris) if result_tris.shape[0] > 0 else result_tris

'''
Returns true if any edge of triangle (v0, v1, v2) intersects and edge of triangle (u0, u1, u2),
projected onto the plane of most significance.
'''
def edge_against_tri_edges(v0, v1, u0, u1, u2, i0, i1):
    Ax = v1[i0] - v0[i0]
    Ay = v1[i1] - v0[i1]
    # Test edge u0, u1 against v0, v1
    if edge_edge_test(v0, u0, u1, i0, i1, Ax, Ay):
        return True
    # Test edge u1, u2 against v0, v1
    if edge_edge_test(v0, u1, u2, i0, i1, Ax, Ay):
        return True
    # Test edge u2, u0 against v0, v1
    if edge_edge_test(v0, u2, u0, i0, i1, Ax, Ay):
        return True
    return False

# This edge to edge test is based on Franlin Antonio's gem:
#   "Faster Line Segment Intersection", in Graphics Gems III,  pp. 199-202
def edge_edge_test(v0, u0, u1, i0, i1, Ax, Ay):
    Bx = u0[i0] - u1[i0]
    By = u0[i1] - u1[i1]
    Cx = v0[i0] - u0[i0]
    Cy = v0[i1] - u0[i1]
    f = Ay * Bx - Ax * By
    d = By * Cx - Bx * Cy
    if (f > 0 and d >= 0 and d <= f) or (f < 0 and d <= 0 and d >= f):
        e = Ax * Cy - Ay * Cx
        if f > 0:
            if e >= 0 and e <= f:
                return True
        else:
            if e <= 0 and e >= f:
                return True
    return False

'''
Python implementation of "A Fast Triangle-Triangle Intersection Test" by Moeller (1997)

The implementation is based on the code available from the authors webpage:
    (http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/)

This implementation uses no divides - which is fast - but unfortunately also makes it a bit less readable.
'''
def fast_triangle_triangle_intersection_test(T1, T2):
    v10, v11, v12 = T1
    v20, v21, v22 = T2
    # 1) Compute plane equation of triangle 2
    N2 = np.cross((v21 - v20), (v22 - v20))
    d2 = -np.dot(N2, v20)
    # 1.1) Compute signed distance from T1 vertices to plane of T2
    dv10 = np.dot(N2, v10) + d2
    dv11 = np.dot(N2, v11) + d2
    dv12 = np.dot(N2, v12) + d2

    dv10dv11 = dv10 * dv11
    dv10dv12 = dv10 * dv12

    # 2) Reject as trivial if all points lie on one side of T2
    if (dv10dv11 > 0 and dv10dv12 > 0):
        return False

    # 3) Compute plane equations of triangle 1
    N1 = np.cross((v11 - v10), (v12 - v10))
    d1 = -np.dot(N1, v10)
    # 3.1) Compute signed distance from T1 vertices to plane of T2
    dv20 = np.dot(N1, v20) + d1
    dv21 = np.dot(N1, v21) + d1
    dv22 = np.dot(N1, v22) + d1

    dv20dv21 = dv20 * dv21
    dv20dv22 = dv20 * dv22

    # 4) Reject as trivial if all points lie on one side of T2
    if (dv20dv21 > 0 and dv20dv22 > 0):
        return False

    # 5.1) Compute direction of intersection line
    D = np.cross(N1, N2)
    
    # 5.2) Compute simplified projection onto L
    max_ = np.abs(D[0])
    index = 0
    bb = np.abs(D[1])
    cc = np.abs(D[2])
    if bb > max_:
        max_ = bb
        index = 1
    if cc > max_:
        max_ = cc
        index = 2
    
    vp0 = v10[index]
    vp1 = v11[index]
    vp2 = v12[index]

    up0 = v20[index]
    up1 = v21[index]
    up2 = v22[index]

    # 6) Compute the intervals for each triangle
    # 6.1) The interval for triangle 1
    interval_1 = compute_intervals(vp0, vp1, vp2, dv10, dv11, dv12, dv10dv11, dv10dv12)
    if interval_1 is None:
        return coplanar_tri_tri(N1, v10, v11, v12, v20, v21, v22)
    # 6.2) The interval for triangle 2
    interval_2 = compute_intervals(up0, up1, up2, dv20, dv21, dv22, dv20dv21, dv20dv22)
    if interval_2 is None:
        return coplanar_tri_tri(N1, v10, v11, v12, v20, v21, v22)

    a, b, c, x0, x1 = interval_1
    d, e, f, y0, y1 = interval_2

    # 7) Intersect the intervals

    xx = x0 * x1
    yy = y0 * y1
    xxyy = xx * yy

    axxyy = a * xxyy
    isect10 = axxyy + b * x1 * yy
    isect11 = axxyy + c * x0 * yy

    dxxyy = d * xxyy
    isect20 = dxxyy + e * xx * y1
    isect21 = dxxyy + f * xx * y0

    isect10, isect11 = sort_interval(isect10, isect11)
    isect20, isect21 = sort_interval(isect20, isect21)

    if isect11 < isect20 or isect21 < isect10:
        return False
    return True

'''
Returns the unique triangles from a set of triangles.
'''
def get_unique_triangles(tris):
    unique_tris = [tris[0]]
    for tri in tris[1:]:
        unique = True
        for t in unique_tris:
            if (all(t[0] == tri[0]) and all(t[1] == tri[1]) and all(t[2] == tri[2])) or\
               (all(t[0] == tri[0]) and all(t[2] == tri[1]) and all(t[1] == tri[2])) or \
               (all(t[1] == tri[0]) and all(t[0] == tri[1]) and all(t[2] == tri[2])) or \
               (all(t[1] == tri[0]) and all(t[2] == tri[1]) and all(t[0] == tri[2])) or \
               (all(t[2] == tri[0]) and all(t[0] == tri[1]) and all(t[1] == tri[2])) or \
               (all(t[2] == tri[0]) and all(t[1] == tri[1]) and all(t[0] == tri[2])):
                unique = False
                break
        if unique:
            unique_tris.append(tri)
    return unique_tris

'''
Returns the unique triangles from a set of vertices.
'''
def get_unique_vertices(vertices):
    unique_vertices = [vertices[0]]
    for v in vertices[1:]:
        unique = True
        for uv in unique_vertices:
            if (uv[0] == v[0] and uv[1] == v[1] and uv[2] == v[2]) or\
               (uv[0] == v[0] and uv[2] == v[1] and uv[1] == v[2]) or \
               (uv[1] == v[0] and uv[0] == v[1] and uv[2] == v[2]) or \
               (uv[1] == v[0] and uv[2] == v[1] and uv[0] == v[2]) or \
               (uv[2] == v[0] and uv[0] == v[1] and uv[1] == v[2]) or \
               (uv[2] == v[0] and uv[1] == v[1] and uv[0] == v[2]):
                unique = False
                break
        if unique:
            unique_vertices.append(v)
    return unique_vertices

'''
Naive way to reduce intersection tests required using AABB's.
This is particularly helpful in the 2D setting, where we would otherwise need to perform
the most expensive test (co-planarity) a bunch of time.
'''
def mesh_mesh_intersection_aabbs(vs0, vs1, fs0, fs1):
    aabbs0 = np.append(np.min(vs0, axis=0), np.max(vs0, axis=0))
    aabbs1 = np.append(np.min(vs1, axis=0), np.max(vs1, axis=0))
    intersections = np.empty((0, 2, vs0.shape[1], vs0.shape[1]))
    if not aabb_intersection(aabbs0, aabbs1):
        return intersections
        
    for ti in fs0:
        t0 = [vs0[ti[0]], vs0[ti[1]], vs0[ti[2]]]
        for tj in fs1:
            t1 = [vs1[tj[0]], vs1[tj[1]], vs1[tj[2]]]
            if aabb_intersection(np.append(np.min(t0, axis=0), np.max(t0, axis=0)),
                                 np.append(np.min(t1, axis=0), np.max(t1, axis=0))):
                if fast_triangle_triangle_intersection_test(t0, t1):
                    intersections = np.append(intersections, [[t0, t1]], axis=0)

    return intersections

'''
Brute-force mesh-mesh intersection computation.
'''
def mesh_mesh_intersection_brute_force(vs0, vs1, fs0, fs1):
    intersections = np.empty((0, 2, vs0.shape[1], vs0.shape[1]))
    for ti in fs0:
        t0 = [vs0[ti[0]], vs0[ti[1]], vs0[ti[2]]]
        for tj in fs1:
            t1 = [vs1[tj[0]], vs1[tj[1]], vs1[tj[2]]]
            if fast_triangle_triangle_intersection_test(t0, t1):
                intersections = np.append(intersections, [[t0, t1]], axis=0)
    return intersections

'''
Order a list of triangles - i.e. (N x 3) matrix - into a mesh of vertices and faces.
'''
def reconstruct_mesh(tris):
    tris = np.array(get_unique_triangles(tris))
    vertices = np.empty((0, 3))
    faces = np.empty((tris.shape[0], 3))
    
    for f, tri in enumerate(tris):
        for i, v in enumerate(tri):
            new = True
            for j, w in enumerate(vertices):
                # v has been seen before
                if all([v[k] == w[k] for k in range(len(v))]):
                    faces[f, i] = j
                    new = False
                    break
            # If we get here, then v is a new vertex
            if new:
                vertices = np.append(vertices, [v], axis=0)
                faces[f, i] = vertices.shape[0] - 1
            
    return vertices, np.array(faces, dtype=np.int32)

'''
Returns true if the point v0 is located inside triangle (u0, u1, u2), projected to
onto the plane of most significance.
'''
def point_in_tri(v0, u0, u1, u2, i0, i1):
    a = u1[i1] - u0[i1]
    b = -(u1[i0] - u0[i0])
    c = -a * u0[i0] - b * u0[i1]
    d0 = a * v0[i0] + b * v0[i1] + c

    a = u2[i1] - u1[i1]
    b = -(u2[i0] - u1[i0])
    c = -a * u1[i0] - b * u1[i1]
    d1 = a * v0[i0] + b * v0[i1] + c

    a = u0[i1] - u2[i1]
    b = -(u0[i0] - u2[i0])
    c = -a * u2[i0] - b * u2[i1]
    d2 = a * v0[i0] + b * v0[i1] + c

    if d0 * d1 > 0:
        if d0 * d2 > 0:
            return True
    return False


# Sort s.t. a <= b
def sort_interval(a, b):
    if a > b:
        return b, a