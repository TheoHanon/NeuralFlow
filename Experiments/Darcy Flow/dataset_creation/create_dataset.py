import numpy as np
import json
import os
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import tensorflow as tf
from darcy_solver import solve_Darcy_PDE
import tqdm

def create_laplacian_2d(n, dx):
    diagonals = [-4*np.ones(n*n), np.ones(n*n), np.ones(n*n), np.ones(n*n), np.ones(n*n)]
    offsets = [0, -1, 1, -n, n]
    return sp.diags(diagonals, offsets, shape=(n*n, n*n)) / dx**2


def generate_random_field(n_points, L):
    identity = sp.eye(n_points * n_points, format="csc")
    operator = sp.csc_matrix(-L + 9 * identity)

    while True:
        random_field = np.random.randn(n_points * n_points)

        random_field, _ = splinalg.cg(operator, random_field)
        random_field, _ = splinalg.cg(operator, random_field)

        random_field = np.where(random_field >= 0, 12.0, 3.0)

        if np.any(random_field  != 3.0) and np.any(random_field != 12.0):
            break

    return random_field.reshape(n_points, n_points)

def serialize_example(permeability_field, solution_field):
    feature = {
        "permeability_field": tf.train.Feature(float_list=tf.train.FloatList(value=permeability_field.ravel())),
        "solution_field": tf.train.Feature(float_list=tf.train.FloatList(value=solution_field.ravel())),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def save_metadata(tfrecord_file, image_size, total_count):
    """Save metadata to a JSON file."""
    metadata = {
        "shape": image_size,
        "n_sample": total_count
    }
    with open(tfrecord_file + "_metadata.json", "w") as f:
        json.dump(metadata, f)

def f(x):
    return np.ones(x.shape[0]) if x.ndim == 2 else 1.0

def solve_and_save(path, n_field, n_points, L):
    dirname = "DarcyDataset"
    path = os.path.join(path, dirname)

    if not os.path.exists(path):
        os.makedirs(path)

    save_metadata(path + "/", (n_points, n_points), n_field)

    with tf.io.TFRecordWriter(path + "/data.tfrecord") as writer:
        for _ in range(n_field):
            random_field = generate_random_field(n_points, L)
        
            # Solve the Darcy PDE
            Nx, Ny = n_points - 2, n_points - 2
            u = solve_Darcy_PDE(Nx, Ny, random_field, f)
            example = serialize_example(random_field, u)
            writer.write(example)


if __name__ == "__main__":

    n_points = 128
    n_field = 10_000
    grid_size = 1.0
    dx = grid_size / n_points
    x = np.linspace(0, grid_size, n_points)
    y = np.linspace(0, grid_size, n_points)
    X, Y = np.meshgrid(x, y)

    L = create_laplacian_2d(n_points, dx)

    solve_and_save("../", n_field, n_points, L)
    print("Dataset saved in TFRecord format.")

