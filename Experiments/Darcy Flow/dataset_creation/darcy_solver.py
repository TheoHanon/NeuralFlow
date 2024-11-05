import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def solve_Darcy_PDE(Nx:int, Ny:int, a:np.ndarray, f : callable):
    """
    Creates the system Au = f for the given PDE discretization and solves it. 

    => Homogenous Dirichlet boundary conditions are applied to the solution w.
    
    Nx, Ny: Number of internal grid points in x and y directions
    a: Coefficient function, given as a 2D array (Nx+2, Ny+2) (includes boundaries)
    f: Right-hand side function, given as a 2D array (Nx+2, Ny+2) (includes boundaries)
    delta_x: Grid spacing in the x direction
    delta_y: Grid spacing in the y direction
    u_bc: Boundary conditions, given as a 2D array (Nx+2, Ny+2)
    """
    # Total number of points, including boundary conditions
    N = (Nx + 2) * (Ny + 2)
    x, y = np.meshgrid(np.linspace(0, 1, Nx+2), np.linspace(0, 1, Ny+2))
    delta_x = x[0, 1] - x[0, 0]
    delta_y = y[1, 0] - y[0, 0]
    X = np.array([x.ravel(), y.ravel()]).T

    
    # Initialize the sparse matrix and the right-hand side vector
    A = lil_matrix((N, N))
    b = np.zeros(N)
    
    def index(i, j):
        # Maps the 2D grid point (i, j) to a 1D index for the matrix A
        return i * (Ny + 2) + j

    # Loop over internal grid points (i, j)
    for i in range(1, Nx+1):
        for j in range(1, Ny+1):
            # Get the 1D index for the current point
            k = index(i, j)
        
            # Discretization for u_xx and u_yy
            A[k, index(i-1, j)] = a[i,j] / delta_x**2 - (a[i+1,j] - a[i-1,j]) / (4 * delta_x**2)
            A[k, index(i+1, j)] = a[i,j] / delta_x**2 + (a[i+1,j] - a[i-1,j]) / (4 * delta_x**2)
            A[k, index(i, j-1)] = a[i,j] / delta_y**2 - (a[i,j+1] - a[i,j-1]) / (4 * delta_y**2)
            A[k, index(i, j+1)] = a[i,j] / delta_y**2 + (a[i,j+1] - a[i,j-1]) / (4 * delta_y**2)
            A[k, k] = -2 * a[i,j] * (1/delta_x**2 + 1/delta_y**2)
            
            # Right-hand side vector b
            b[k] = -f(X[index(i, j)])   

            
    # Apply Dirichlet boundary conditions by modifying the system
    for i in range(Nx+2):
        for j in range(Ny+2):
            if i == 0 or i == Nx+1 or j == 0 or j == Ny+1:
                k = index(i, j)
                A[k, k] = 1.0
                b[k] = 0.0
    
    # Convert A to CSR format for faster computations
    A = A.tocsr()
    u_flat = spsolve(A, b)
    
    # Reshape the solution to a 2D grid (including boundaries)
    u = u_flat.reshape((Nx+2, Ny+2))
    
    return u
