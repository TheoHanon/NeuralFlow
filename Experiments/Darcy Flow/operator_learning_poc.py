import tensorflow as tf
from deeponet import DeepONet, CosineAnnealingSchedule
from utils import *
from flow import Flow_v2
import numpy as np
from sklearn.preprocessing import StandardScaler
import gc
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--std", type = float, required=True)
parser.add_argument("--dirname", type = str, required = True)
args = parser.parse_args()

def model_fn():
    model = DeepONet(
        n_branch = 32, 
        n_trunk = 2, 
        width = 100, 
        depth = 3, 
        output_dim = 100,
        activation = "relu",
    )
    return model

def optimizer_fn():
    lr_schedule = CosineAnnealingSchedule(1e-3, decay_steps=100)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule)
    return optimizer


dataloader = DarcyDatasetLoader("DarcyDataset")
train_dataset, test_dataset = dataloader.get_split(test_size = 0.1, batch_size = -1, frac = 0.20)

def downsample_with_boundary(a, u):
    # For a: Select every 4th index, keeping boundary values
    a_downsampled = a[::4, ::4]
    a_downsampled = tf.tensor_scatter_nd_update(
        a_downsampled,
        indices=[[0], [a_downsampled.shape[0] - 1]], 
        updates=[a[0, ::4], a[-1, ::4]],
    )

    # For u: Select every 2nd index, keeping boundary values
    u_downsampled = u[::2, ::2]
    u_downsampled = tf.tensor_scatter_nd_update(
        u_downsampled,
        indices=[[0], [u_downsampled.shape[0] - 1]],
        updates=[u[0, ::2], u[-1, ::2]],
    )

    interior = tf.ones([u_downsampled.shape[0] - 2, u_downsampled.shape[0] - 2], dtype=u_downsampled.dtype)
    mask = tf.pad(interior, paddings=[[1, 1], [1, 1]], mode='CONSTANT', constant_values=0)

    # Apply the mask
    u_downsampled = u_downsampled * mask
    
    return a_downsampled, u_downsampled


def sample_coords_and_values(a, u, grid, num_samples):
    
    indices = tf.random.uniform(
        shape=(num_samples,), minval=0, maxval=tf.shape(grid)[0], dtype=tf.int32
    )
    
    sampled_coords = tf.gather(grid, indices)  # Shape: (num_samples, 2)
    sampled_u = tf.gather(tf.reshape(u, [-1]), indices)  # Shape: (num_samples,)
    
    sampled_u = tf.expand_dims(sampled_u, axis=1)            # Shape: (num_samples, 1)
    
    a_expanded = tf.expand_dims(a, axis=0)  # Shape: (1, 32, 32)
    a_tiled = tf.tile(a_expanded, [num_samples, 1, 1])  # Shape: (num_samples, 32, 32)
    
    return (a_tiled, sampled_coords), sampled_u



grid = tf.meshgrid(
    tf.linspace(0, 1, 64),
    tf.linspace(0, 1, 64)
)

grid = tf.reshape(tf.stack(grid, axis=-1), (64*64, 2))
grid = tf.cast(grid, dtype=tf.float32)

train_dataset = train_dataset.map(
    lambda a, u : downsample_with_boundary(a, u/1e-2), 
    num_parallel_calls=tf.data.AUTOTUNE 
    )

train_dataset = train_dataset.map(
    lambda a, u : sample_coords_and_values(a, u, grid, 5), 
    num_parallel_calls=tf.data.AUTOTUNE 
    ).unbatch().shuffle(1000)

test_dataset  = test_dataset.map(
    lambda a, u : downsample_with_boundary(a, u/1e-2),
    num_parallel_calls=tf.data.AUTOTUNE
    )

test_dataset  = test_dataset.map(
    lambda a, u : ((a, grid), u),
    num_parallel_calls=tf.data.AUTOTUNE
    ) 


def run_flow_experiments(N_MODELS, EPOCHS = 1_000, MEMORY_EPOCHS=[100, 500, 1000, 2500, 5000]) :

    flow = Flow_v2(
        model_fn = model_fn,
        n_models = N_MODELS,
        noise_stddev=args.std,
        lam = tf.cast(np.sqrt(2.0), tf.float32))

    flow.compile(
        optimizer_fn=optimizer_fn,
        loss_fn=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM), 
        metrics=["mae"], 
    )

    flow.fit(
        x=train_dataset, 
        y=None, 
        epochs = EPOCHS, 
        memory_epochs = MEMORY_EPOCHS, 
        verbose = 2, 
        batch_size = 128,
    )

    test_samples = list(test_dataset.take(20))

    for n_models in range(1, N_MODELS+1):

        est_IS = flow.get_estimator(name='importance_sampling', n_models = n_models)
        est_DP = flow.get_estimator(name='deep_ensemble', n_models = n_models)

        
        list_u_pred_IS, list_std_pred_IS = [], []
        list_u_pred_DP, list_std_pred_DP = [], []
        list_error_IS, list_error_DP = [], []

        IS_weights = est_IS.importance_weight

        if n_models == 1:
            list_a_true, list_u_true =  [], []

        for (a, coords), u in test_samples:

            u_pred_IS, std_pred_IS = est_IS((a[None, ...], coords))
            u_pred_DP, std_pred_DP = est_DP((a[None, ...], coords))

            u_pred_IS = u_pred_IS.squeeze(-1)
            std_pred_IS = std_pred_IS.squeeze(-1)

            error_IS = np.abs(u_pred_IS - tf.reshape(u, [-1]))**2 / np.max(tf.reshape(u, [-1])**2)
            error_DP = np.abs(u_pred_DP - tf.reshape(u, [-1]))**2 / np.max(tf.reshape(u, [-1])**2)

            if n_models == 1:
                list_a_true.append(a)
                list_u_true.append(u)

            list_u_pred_IS.append(u_pred_IS)
            list_std_pred_IS.append(std_pred_IS)
            
            list_u_pred_DP.append(u_pred_DP)
            list_std_pred_DP.append(std_pred_DP)

            list_error_IS.append(error_IS)
            list_error_DP.append(error_DP)


        list_u_pred_IS, list_std_pred_IS = np.array(list_u_pred_IS), np.array(list_std_pred_IS)
        list_u_pred_DP, list_std_pred_DP = np.array(list_u_pred_DP), np.array(list_std_pred_DP)
        list_error_IS, list_error_DP = np.array(list_error_IS), np.array(list_error_DP)

        if n_models == 1:
            list_a_true, list_u_true = np.array(list_a_true), np.array(list_u_true)

        output_dir = args.dirname  # The directory where the file will be saved
        os.makedirs(output_dir, exist_ok=True)

        np.savez_compressed(os.path.join(output_dir, f"{n_models}.npz"), 
                u_pred_IS=list_u_pred_IS,
                std_pred_IS=list_std_pred_IS,
                u_pred_DP=list_u_pred_DP,
                std_pred_DP=list_std_pred_DP,
                error_IS=list_error_IS,
                error_DP= list_error_DP, 
                IS_weights = IS_weights
                )

        del est_IS, est_DP
        del list_u_pred_IS, list_std_pred_IS, list_u_pred_DP, list_std_pred_DP
        del list_error_IS, list_error_DP
        gc.collect()

        print(f"==n_models = {n_models} Done!")     
    

    np.savez_compressed(os.path.join(output_dir, "gt.npz"), 
                        a=list_a_true, 
                        u=list_u_true)



run_flow_experiments(N_MODELS=20, MEMORY_EPOCHS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500], EPOCHS = 500)
# run_flow_experiments(N_MODELS=1, MEMORY_EPOCHS = [2], EPOCHS = 2)

