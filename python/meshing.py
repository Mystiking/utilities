import numpy as np
import polygons


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