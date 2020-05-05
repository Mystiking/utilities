import numpy as np
import utilities.python.csg2d as csg
import utilities.python.polygons as polygons

class Grid:

    def __init__(self, min_coord, max_coord, I, J, boundary = 0.5):
        self.min_coord = min_coord - boundary
        self.max_coord = max_coord + boundary
        self.I = I  # Number of nodes along x axis
        self.J = J  # Number of nodes along y axis
        res = np.array([I-1, J-1], dtype=np.float64)
        dims = (self.max_coord - self.min_coord)
        self.spacing = np.divide( dims, res)
        self.gridphi = None
        self.phi = None

    def get_node_value(self,i, j):
        row_idx = i + self.I * j
        return self.phi[row_idx]

    def get_node_coord(self, i, j):
        return self.min_coord + np.multiply(self.spacing, np.array([i, j],dtype=np.float64))

    def get_enclosing_cell_idx(self, p):
        idx = np.floor( (p - self.min_coord) / self.spacing )
        i = int(idx[0])
        j = int(idx[1])
        
        return i, j
    
    def get_nodes_array(self):
        nodes = np.zeros( ((self.I*self.J), 2), dtype=np.float64) 
        for j in range(self.J):
            y = self.min_coord[1] + self.spacing[1] * j
            for i in range(self.I):
                x = self.min_coord[0] + self.spacing[0] * i
                coord = np.array([x, y],dtype=np.float64)
                row_idx = i + self.I * j
                nodes[row_idx,:] = coord
        return nodes

    def set_phi(self, poly, sdf=lambda p, c: polygons.sdf(p, c)):
        self.phi = np.zeros( (self.I*self.J), dtype=np.float64) 
        for j in range(self.J):
            y = self.min_coord[1] + self.spacing[1] * j
            for i in range(self.I):
                x = self.min_coord[0] + self.spacing[0] * i
                coord = np.array([x, y],dtype=np.float64)
                row_idx = i + self.I * j
                self.phi[row_idx] = sdf(poly, coord)

    def set_gridphi(self):
        self.gridphi = csg.box((self.max_coord[0]-self.min_coord[0]) / 2, (self.max_coord[1]-self.min_coord[1]) / 2)
        self.gridphi = csg.translate(self.gridphi, np.array([(self.max_coord[0]+self.min_coord[0]) / 2, (self.max_coord[1]+self.min_coord[1]) / 2]))


def write_matlab_file(filename, grid):
    data = {}
    data['I'] = grid.I
    data['J'] = grid.J
    data['min_coord'] = grid.min_coord
    data['max_coord'] = grid.max_coord
    data['spacing'] = grid.spacing
    data['phi'] = grid.phi
    from scipy.io import savemat
    savemat(file_name=filename, mdict=data, appendmat=False)


def read_matlab_file(filename):
    from scipy.io import loadmat
    data = {}
    loadmat(file_name=filename, mdict=data, appendmat=False)
    I = int(data['I'])
    J = int(data['J'])
    min_coord = data['min_coord']
    max_coord = data['max_coord']
    grid = Grid(min_coord=min_coord,max_coord=max_coord, I=I, J=J)
    grid.phi = data['phi']
    return grid


def get_value(grid, p):
    i, j = grid.get_enclosing_cell_idx(p)

    outside = 0
    # If the point is outside of the grid, we project it onto the grid
    # and continue evaluating the distance from the projected point
    if i > grid.I - 1 or i < 0 or j > grid.J - 1 or j < 0:
        outside = 1
        pproj = p - grid.gridphi(p)*approximate_gradient(grid.gridphi, p)
        i, j = grid.get_enclosing_cell_idx(pproj)
        i = int(np.clip(i, 0, grid.I - 1))
        j = int(np.clip(j, 0, grid.J - 1))
    

    if i == grid.I - 1 and j == grid.J - 1:
        d00 = grid.get_node_value(i, j)
        v = d00
    elif i == grid.I - 1 and j != grid.J - 1:
        d00 = grid.get_node_value(i, j)
        d01 = grid.get_node_value(i, j+1)
        t = (p[1]  - ((j * grid.spacing[1]) + grid.min_coord[1] )) / grid.spacing[1]    
        v = (1 - t) * d00 + t * d01
    elif i != grid.I - 1 and j == grid.J - 1:
        d00 = grid.get_node_value(i, j)        
        d10 = grid.get_node_value(i+1, j)
        s = (p[0]  - ((i * grid.spacing[0]) + grid.min_coord[0] )) / grid.spacing[0]
        v = (1 - s) * d00 + s * d10
    else:
        d00 = grid.get_node_value(i, j)
        d01 = grid.get_node_value(i, j+1)

        d10 = grid.get_node_value(i+1, j)
        d11 = grid.get_node_value(i+1, j+1)

        s = (p[0]  - ((i * grid.spacing[0]) + grid.min_coord[0] )) / grid.spacing[0]
        t = (p[1]  - ((j * grid.spacing[1]) + grid.min_coord[1] )) / grid.spacing[1]

        v = (1 - t) * ((1 - s) * d00 + s * d10) + t * ((1 - s) * d01 + s * d11)
    
    if outside:
        # Closest point on phi
        cphi = pproj - get_gradient_from_node_indices(grid, pproj, np.clip(i, 0, grid.I - 2), np.clip(j, 0, grid.J - 2)) * v
        # Distance from p to cphi is now the distance value
        v = np.linalg.norm(cphi - p)
    return v


def approximate_gradient(phi, p, eps=1e-6):
    deltax = eps
    deltay = eps
    
    gradx = phi(p + np.array([deltax, 0])) - phi(p - np.array([deltax, 0]))
    gradx = gradx / (2*deltax)
    grady = phi(p + np.array([0, deltay])) - phi(p - np.array([0, deltay]))
    grady = grady / (2*deltay)

    gsize = np.linalg.norm(np.array([gradx, grady]))

    return np.array([gradx, grady]) / gsize if gsize > 0.0 else np.array([0.0, 0.0])

def get_gradient(grid, p):
    i, j = grid.get_enclosing_cell_idx(p)

    if i >= grid.I or j >= grid.J:
        return np.array([np.inf, np.inf])

    d00 = grid.get_node_value(i, j)
    d01 = grid.get_node_value(i, j+1)

    d10 = grid.get_node_value(i+1, j)
    d11 = grid.get_node_value(i+1, j+1)

    s = (p[0]  - ((i * grid.spacing[0]) + grid.min_coord[0] )) / grid.spacing[0]
    t = (p[1]  - ((j * grid.spacing[1]) + grid.min_coord[1] )) / grid.spacing[1]

    ds_dx = 1.0 / grid.spacing[0]
    dt_dy = 1.0 / grid.spacing[1]
    
    dphi_ds = (1 - t) * (d10 - d00) + t * (d11 - d01)
    dphi_dt = (1 - s) * (d01 - d00) + s * (d11 - d10)

    return np.array([dphi_ds*ds_dx, dphi_dt*dt_dy], dtype=np.float64)

def get_gradient_from_node_indices(grid, p, i, j):
    d00 = grid.get_node_value(i, j)
    d01 = grid.get_node_value(i, j+1)

    d10 = grid.get_node_value(i+1, j)
    d11 = grid.get_node_value(i+1, j+1)

    s = (p[0]  - ((i * grid.spacing[0]) + grid.min_coord[0] )) / grid.spacing[0]
    t = (p[1]  - ((j * grid.spacing[1]) + grid.min_coord[1] )) / grid.spacing[1]

    ds_dx = 1.0 / grid.spacing[0]
    dt_dy = 1.0 / grid.spacing[1]
    
    dphi_ds = (1 - t) * (d10 - d00) + t * (d11 - d01)
    dphi_dt = (1 - s) * (d01 - d00) + s * (d11 - d10)

    return np.array([dphi_ds*ds_dx, dphi_dt*dt_dy], dtype=np.float64)


def create_signed_distance(poly, I, J, boundary = 0.5):    
    min_coord = poly.min(axis=0) - boundary
    max_coord = poly.max(axis=0) + boundary
    grid = Grid(min_coord, max_coord, I, J)
    grid.set_phi(poly)
    grid.set_gridphi()
    return grid


def show(grid):
    import matplotlib as ml
    import matplotlib.pyplot as plt

    layer = grid.I*grid.J
    img = grid.phi.reshape((grid.I,grid.J))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Signed Distance Field')
    plt.imshow(img)
    ax.set_aspect('equal')
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()


def is_inside(grid, p, boundary=0.5):
    if p[0] > (grid.max_coord[0] - boundary):
        return False
    if p[1] > (grid.max_coord[1] - boundary):
        return False
    
    if p[0] < (grid.min_coord[0] + boundary):
        return False
    if p[1] < (grid.min_coord[1] + boundary):
        return False
    return True
    
