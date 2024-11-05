import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import os
from darcy_solver import solve_Darcy_PDE
import tensorflow as tf
import tqdm

# Dataset creation script for random permeability fields and corresponding pressure fields
n_points = 421  # Number of grid points in each dimension
grid_size = 1.0
dx = grid_size / n_points

# Create a 2D grid of points
x = np.linspace(0, grid_size, n_points)
y = np.linspace(0, grid_size, n_points)
X, Y = np.meshgrid(x, y)

# Function to create Laplacian operator (-Δ)
def create_laplacian_2d(n, dx):
    diagonals = [-4*np.ones(n*n), np.ones(n*n), np.ones(n*n), np.ones(n*n), np.ones(n*n)]
    offsets = [0, -1, 1, -n, n]
    laplacian = sp.diags(diagonals, offsets, shape=(n*n, n*n)) / dx**2
    return laplacian

L = create_laplacian_2d(n_points, dx)

# Generate random permeability fields
def generate_random_fields(n_field, n_points):
    random_fields = []
    identity = sp.eye(n_points*n_points, format="csc")
    operator = sp.csc_matrix(-L + 9 * identity)

    for _ in tqdm.tqdm(range(n_field)):
        mean = np.zeros(n_points * n_points)
        random_field = np.random.randn(n_points * n_points)
        random_field, _ = splinalg.cg(operator, random_field)
        random_fields.append(random_field)
    return random_fields

# Binarize random field
def phi(x):
    out = 12*np.ones_like(x)
    out[x < 0] = 3
    return out

# Generate dataset
n_field = 1000
random_fields = generate_random_fields(n_field, n_points)


random_fields_binarized = []
with tqdm.tqdm(total=n_field) as pbar:
    for random_field in random_fields:
        binarized_field = phi(random_field.reshape(n_points, n_points))
        random_fields_binarized.append(binarized_field)
        pbar.update(1)

# Function f

def f(x):
    if x.ndim == 1:
        return 1.0
    elif x.ndim == 2:
        return np.ones(x.shape[0])

# Solve Darcy PDE for each permeability field
u_sol = []
with tqdm.tqdm(total=n_field) as pbar:
    for a in random_fields_binarized:
        Nx, Ny = a.shape[0] - 2, a.shape[1] - 2
        u = solve_Darcy_PDE(Nx, Ny , a, f)
        u_sol.append(u)
        pbar.update(1)

# Create and save TensorFlow Dataset
features = tf.data.Dataset.from_tensor_slices((random_fields_binarized, u_sol))
features.save("darcy_dataset")

print("Dataset saved to darcy_dataset folder.")
