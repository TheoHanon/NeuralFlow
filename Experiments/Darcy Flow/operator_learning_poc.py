import tensorflow as tf
from deeponet import DeepONet
from utils import *
import matplotlib.pyplot as plt
import os

# Boundary function
@tf.function(reduce_retracing=True)
def boundary_fn(x, boundaries=(0.0, 1.0)):
    return (x[..., 0] - boundaries[0]) * (x[..., 0] - boundaries[1]) * (x[..., 1] - boundaries[0]) * (x[..., 1] - boundaries[1])

# Downsample function
def downsample(a, u):

    a = tf.expand_dims(a, axis=-1)
    u = tf.expand_dims(u, axis=-1)

    pooling_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

    a_downsampled = pooling_layer(a)
    a_downsampled = pooling_layer(a_downsampled)
    a_downsampled = tf.squeeze(a_downsampled, axis=-1)

    u_downsampled = pooling_layer(u)
    u_downsampled = tf.squeeze(u_downsampled, axis =-1)

    return a_downsampled, u_downsampled

# Training step function
@tf.function
def train_step(a_batch, u_batch, model, optimizer, loss_fn, coords):
    with tf.GradientTape() as tape:
        boundaries = boundary_fn(coords)
        predictions = model(tf.reshape(a_batch, (-1, 32*32)), coords) * boundaries
        loss = loss_fn(tf.reshape(u_batch, (-1, 64*64)), predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def main():
    # Configure device and directories
    device = "/gpu:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"
    print("Device :", device)
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_{epoch:04d}.ckpt")

    # Initialize model, loss, and optimizer
    model = DeepONet(
        n_branch=32 * 32, 
        n_trunk=2, 
        width=40, 
        depth=3, 
        output_dim=64 * 64,
        activation="relu",
    )
    loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4, 
        decay_steps=500,
        decay_rate=0.96,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Load and process data
    dataloader = DarcyDatasetLoader("DarcyDataset")
    train_dataset, test_dataset = dataloader.get_split(0.2, batch_size=32)
    train_dataset = train_dataset.map(lambda a, u: downsample(a, u))
    test_dataset = test_dataset.map(lambda a, u: downsample(a, u))

    # Coordinates for training and testing
    coords = tf.meshgrid(tf.linspace(0.0, 1.0, 64), tf.linspace(0.0, 1.0, 64))
    coords = tf.stack(coords, axis=-1)
    coords = tf.reshape(coords, (-1, 2))

    # Training loop parameters
    epochs = 10_000

    # Main training loop
    for epoch in range(epochs):
        loss_avg = 0.0
        num_batches = 0
        for step, (a_batch, u_batch) in enumerate(train_dataset):
            loss_avg += train_step(a_batch, u_batch, model, optimizer, loss_fn, coords)
            num_batches += 1
        loss_avg /= num_batches

        if (epoch + 1) % 100:
            print(f"Epoch[{epoch+1}/{epochs}] Loss : {loss_avg.numpy():.9e}")
        
        # Save model checkpoints periodically
        if (epoch + 1) % 2000 == 0:
            model.save_weights(checkpoint_path.format(epoch=epoch+1))

    model.save_weights(checkpoint_path.format(epoch=epochs))
    print("Training completed and model parameters saved.")

if __name__ == "__main__":
    main()
