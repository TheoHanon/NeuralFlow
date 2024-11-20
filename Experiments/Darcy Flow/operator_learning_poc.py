import tensorflow as tf
from deeponet import DeepONet, CosineAnnealingSchedule
from utils import *
from flow import Flow_v2
import numpy as np
import gc

def model_fn():
    model = DeepONet(
        n_branch = 64, 
        n_trunk = 2, 
        width = 100, 
        depth = 3, 
        output_dim = 100,
        activation = "relu",
        grid_size = (128, 128)
    )
    return model

def optimizer_fn():
    lr_schedule = CosineAnnealingSchedule(
        initial_learning_rate=0.001, 
        decay_steps=100,  
        alpha=1e-4 
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    return optimizer

def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis = 1), axis = 0)

dataloader = DarcyDatasetLoader("DarcyDataset")
train_dataset, test_dataset = dataloader.get_split(test_size = 0.1, batch_size = -1, frac = 0.8)

def downsample(a, u):
    a = tf.expand_dims(a, axis=-1) 
    a = tf.expand_dims(a, axis=0) 
    pooling_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    a_downsampled = pooling_layer(a)
    a_downsampled = tf.squeeze(a_downsampled)
    return a_downsampled, u

train_dataset = train_dataset.map(lambda a, u : downsample(a, u/1e-2))
test_dataset  = test_dataset.map(lambda a, u : downsample(a, u/1e-2))

X_train = tf.convert_to_tensor(
    list(train_dataset.map(lambda a, u: a).as_numpy_iterator())
)
y_train = tf.convert_to_tensor(
    list(train_dataset.map(lambda a, u: tf.reshape(u, [-1])).as_numpy_iterator())
)

def run_flow_experiments(N_MODELS, EPOCHS = 1_000, MEMORY_EPOCHS=[100, 500, 1000, 2500, 5000]) :

    flow = Flow_v2(
        model_fn = model_fn,
        n_models = N_MODELS,
        noise_stddev=tf.cast(1e-9, tf.float32), 
        lam = tf.cast(1e0, tf.float32))

    flow.compile(
        optimizer_fn=optimizer_fn,
        loss_fn=loss_fn, 
        metrics=["mae"], 
    )

    flow.fit(
        x=X_train, 
        y=y_train, 
        epochs = EPOCHS, 
        batch_size = 128,
        memory_epochs = MEMORY_EPOCHS, 
        verbose = 2
    )

    for n_models in range(1, N_MODELS+1):

        est_IS = flow.get_estimator(name='importance_sampling', n_models = n_models)
        est_DP = flow.get_estimator(name='deep_ensemble', n_models = n_models)

        list_u_pred_IS, list_std_pred_IS = [], []
        list_u_pred_DP, list_std_pred_DP = [], []
        list_error_IS, list_error_DP = [], []

        if n_models == 1:
            list_a_true, list_u_true =  [], []

        for (a, u) in test_dataset.take(20):
        
            u_pred_IS, std_pred_IS = est_IS(a[None, ...])
            u_pred_DP, std_pred_DP = est_DP(a[None, ...])

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


        np.savez_compressed(f"results/{n_models}.npz", 
                u_pred_IS=list_u_pred_IS,
                std_pred_IS=list_std_pred_IS,
                u_pred_DP=list_u_pred_DP,
                std_pred_DP=list_std_pred_DP,
                error_IS=list_error_IS,
                error_DP= list_error_DP
                )

        del est_IS, est_DP
        del list_u_pred_IS, list_std_pred_IS, list_u_pred_DP, list_std_pred_DP
        del list_error_IS, list_error_DP
        gc.collect()

        print(f"==n_models = {n_models} Done!")     
    
    np.savez_compressed(f"results/gt.npz", 
                        a=list_a_true, 
                        u=list_u_true)

if __name__=="__main__":

    run_flow_experiments(N_MODELS=15, MEMORY_EPOCHS = [100, 500, 1000, 1500, 2000], EPOCHS = 2000)