import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from scipy.special import logsumexp
from typing import Union
from tqdm.notebook import tqdm
import estimator as est
import parallel_model as pm

class Flow:

    def __init__(self, 
                 prior: Union[tfp.distributions.Distribution, bool] = tfp.distributions.Uniform(-1, 1),
                 likelihood_std: float = 1.0, 
                 k: Union[float, np.ndarray] = 1.0,
                 n_epochs_recorded: int = 0):
        """
        Initialize the Flow class.

        Args:
            prior (tfp.distributions.Distribution or bool): Prior distribution over model weights.
                If False, no prior is used.
            likelihood_std (float): Standard deviation of the likelihood function.
            k (float or np.ndarray): Noise covariance. If scalar, uses scalar noise; if array,
                uses the provided covariance matrix.
            n_epochs_recorded (int): Number of epochs to record for weight and distribution tracking.
        """
        self.prior = prior
        self.likelihood_std = tf.cast(likelihood_std, tf.float32)
        self.log_p = self.setup_log_p()
        self.n_epochs_recorded = n_epochs_recorded
        self.n_models = None

        if np.isscalar(k):
            self.is_scalar_noise = True
            self.noise_cov_matrix = k
        else:
            self.is_scalar_noise = False
            self.noise_cov_matrix = tf.convert_to_tensor(k, dtype=tf.float32)

        # Training parameters

        self.callbacks = []
        self.optimizer = None
        self.metrics = None

        # Recording for gradient and diffusion steps
        self.model = None
        self.gradient_weights = []
        self.diffusion_weights = []
        self.logp = []
        self.learning_rates = []

        self.__flowed = False
        self.__compiled = False

    def _reset_flow(self):
        """
        Reset the flow instance.
        """
        self.__flowed = False
        self.model = None
        self.gradient_weights = []
        self.diffusion_weights = []
        self.logp = []
        self.learning_rates = []

    def setup_log_p(self):
        """
        Set up the log probability function based on the prior.

        Returns:
            function: A function that computes the log probability.
        """
        likelihood_std = self.likelihood_std
        prior = self.prior

        if not prior:
            def log_p(y_true, y_pred, models_weights):
                residuals = y_true - y_pred  # Shape: (batch_size, num_models, output_dim)
                negative_log_likelihood = -tf.reduce_mean(
                    tf.reduce_sum(residuals**2, axis=2), axis=1
                ) / (2 * likelihood_std**2)
                return negative_log_likelihood
        else:
            def log_p(y_true, y_pred, models_weights):
                residuals = y_true - y_pred  # Shape: (batch_size, num_models, output_dim)
                negative_log_likelihood = -tf.reduce_mean(
                    tf.reduce_sum(residuals**2, axis=2), axis=1
                ) / (2 * likelihood_std**2)
                # Flatten and concatenate model weights
                models_weights_flat = [
                    tf.concat([tf.reshape(w, [-1]) for w in model_weights], axis=0)
                    for model_weights in models_weights
                ]
                prior_log_prob = tf.reduce_sum(prior.log_prob(models_weights_flat), axis=1)
                return negative_log_likelihood + prior_log_prob

        return log_p


    
    def reset_metrics(self):
        """
        Reset all metrics at the start of each epoch.
        """
        for metric in self.metrics:
            metric.reset_state()

    def on_epoch_start(self, epoch, logs=None):
        """
        Callbacks on epoch start.

        Args:
            epoch (int): Current epoch number.
            logs (dict or None): Dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """
        Callbacks on epoch end.

        Args:
            epoch (int): Current epoch number.
            logs (dict or None): Dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def loss_and_metrics_update(self, logs, model, y, y_pred, losses):
        """
        Update the loss and metrics for each model.

        Args:
            logs (dict): Dictionary to store logs.
            model (tf.keras.Model): Model(s) being trained.
            y (tf.Tensor): Target data.
            y_pred (list of tf.Tensor): Predictions from each model.
            losses (tf.Tensor): Losses for each model.
        """
        for i, single_model in enumerate(model.models):
            logs[f"loss_model_{i}"] = losses[i].numpy()
            for metric in self.metrics:
                metric.update_state(y, y_pred[i])
                logs[f"{metric.name}_model_{i}"] = metric.result().numpy()

    def _setup_training(self, model, callbacks):
        """
        Set up the training environment.

        Args:
            model (tf.keras.Model): Model(s) to train.
            callbacks (list): List of callbacks to use during training.
        """
        self.model = model
        self.callbacks = callbacks or []

    def setup_model(self, model : tf.keras.Model) -> None:
        self.model = pm.ParallelModel(model, self.n_models)
    
    def compile(self, optimizer, n_models, lr_schedule = None, metrics=None):
        """
        Compile the Flow instance with optimizer and metrics.

        Args:
            optimizer (str or tf.optimizers.Optimizer): Optimizer to use.
            lr (float): Learning rate.
            metrics (list or None): List of metrics to evaluate during training.
        """
    
        if isinstance(optimizer, str):
            self.optimizer = tf.optimizers.get(optimizer)
        else:
            self.optimizer = optimizer

        self.lr_schedule = lr_schedule or (lambda iter : self.optimizer.learning_rate)
        self.metrics = [tf.keras.metrics.get(metric) for metric in (metrics or [])]
        self.n_models = n_models
        self.__compiled = True

    @tf.function
    def gradient_step(self, models: tf.keras.Model, x, y):
        """
        Perform one training step.

        Args:
            models (tf.keras.Model): Model(s) to train.
            x (tf.Tensor): Input data.
            y (tf.Tensor): Target data.

        Returns:
            tuple: (losses, y_pred)
                losses (tf.Tensor): Losses for each model.
                y_pred (list of tf.Tensor): Predictions from each model.
        """
        with tf.GradientTape() as tape:
            y_pred = models(x)  # y_pred is a list of predictions from each model
            if not self.prior:
                losses = -self.log_p(y, y_pred, None)
            else:
                losses = -self.log_p(y, y_pred, [m.trainable_variables for m in models.models])
        
        # Compute gradients
        grads = tape.gradient(losses, models.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, models.trainable_variables))

        return losses, y_pred

    def diffusion_step(self):
        """
        Add Gaussian noise to the model variables with covariance K.

        Args:
            models (tf.keras.Model): Model(s) being trained.
        """
        if self.is_scalar_noise:
            lr = self.lr_schedule(self.optimizer.iterations)
            stddev = tf.sqrt(2 * lr * self.noise_cov_matrix)
            for v in self.model.trainable_variables:
                noise = tf.random.normal(shape=tf.shape(v), mean=0.0, stddev=stddev)
                v.assign_add(noise)
        else:
            # TODO: Implement non-scalar noise covariance matrix
            raise NotImplementedError("Non-scalar noise covariance matrix not implemented yet.")

    def flow(self, 
             model: tf.keras.Model, 
             x, 
             y, 
             epochs, 
             batch_size, 
             validation_set = None,
             callbacks=None):
        """
        The main training loop that runs for the specified number of epochs.

        Args:
            model (tf.keras.Model): Model(s) to train.
            x (tf.Tensor): Input data.
            y (tf.Tensor): Target data.
            epochs (int): Number of epochs to train.
            batch_size (int): Batch size for training.
            validation_set (tuple or None): Validation data as a tuple of (x_val, y_val).
            callbacks (list or None): List of callbacks to use during training.
        """

        self._reset_flow()
        
        if not self.__compiled:
            raise ValueError("Flow not compiled. Run the compile method first.")
        
        self.setup_model(model)
        self._setup_training(self.model, callbacks)

        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size = 1024).batch(batch_size)

        with tqdm(total=epochs, desc="Training Epochs", unit="epoch") as epoch_bar:
            for epoch in range(epochs):
                self.reset_metrics()
                self.on_epoch_start(epoch)
                logs = {}

                for x_batch, y_batch in dataset:
                    losses, y_pred = self.gradient_step(self.model, x_batch, y_batch)                    
                    self.loss_and_metrics_update(logs, self.model, y_batch, y_pred, losses)

                if epoch >= epochs - self.n_epochs_recorded:
                    
                    self.gradient_weights.append(self.model.weights)
                    self.diffusion_step()
                    self.diffusion_weights.append(self.model.weights)
                    self.learning_rates.append(self.lr_schedule(self.optimizer.iterations))

                    y_pred = self.model(x)
                    self.logp.append(self.log_p(y, y_pred, [m.trainable_variables for m in self.model.models]))

                else : 
                    self.diffusion_step()

                self.on_epoch_end(epoch, logs)


                if validation_set is not None:
                    x_val, y_val = validation_set
                    y_val_pred = self.model(x_val)
                    val_losses = -self.log_p(y_val, y_val_pred, [m.trainable_variables for m in self.model.models])
                    val_logs = {}
                    self.loss_and_metrics_update(val_logs, self.model, y_val, y_val_pred, val_losses)
                    val_loss_str = ", ".join([f'{k}: {v}' for k, v in val_logs.items()])
                

                postfix = {f'Loss_Model_{i}': f'{logs[f"loss_model_{i}"]:.4f}' for i in range(len(self.model.models))}
                loss_str = ", ".join([f'{k}: {v}' for k, v in postfix.items()])

                if validation_set is not None:
                    epoch_bar.set_postfix_str(f"Epoch {epoch + 1}/{epochs} - {loss_str} - {val_loss_str}")
                else:
                    epoch_bar.set_postfix_str(f"Epoch {epoch + 1}/{epochs} - {loss_str}")

                # Update the epoch progress bar after each epoch
                epoch_bar.update(1)

        self.__flowed = True
        self.__compiled = False

        return
    
    def get_estimator(self, name: str):
        """
        Get the estimator with the given name.

        Args:
            name (str): Name of the estimator.

        Returns:
            Estimator: Estimator object.
        """
        if name == 'deep_ensemble':

            weights = np.array(self.gradient_weights)
            return est.Ensemble(weights, self.model)
        
        elif name == 'importance_sampling':

            weights, logq, logp = self.compute_weight_and_distribution()
            return est.get_estimator(name, weights, logq, logp, self.model)
    
        else:
            raise ValueError(f"Invalid estimator name: {name}")
        
    
    def compute_weight_and_distribution(self):
        """
        Compute and return the weights and distributions after flow has been applied.

        Returns:
            tuple: (diff_weights, log_q, logp)
                diff_weights (np.ndarray): Diffused weights.
                log_q (np.ndarray): Log probability of the diffusion steps.
                logp (np.ndarray): Log probability of the model predictions.

        Raises:
            ValueError: If flow has not been computed yet.
        """
        if not self.__flowed:
            raise ValueError("Flow not yet computed. Run the flow method first.")
        
        grad_weights = np.array(self.gradient_weights)
        diff_weights = np.array(self.diffusion_weights)

        M = len(self.model.models)  # Number of models
        d = grad_weights.shape[2]

        logq = np.zeros((self.n_epochs_recorded, M))
        logp = np.array(self.logp)

        for i, (wgrad, wdiff) in enumerate(zip(grad_weights, diff_weights)):
            diff = wdiff[:, None, :] - wgrad[None, :, :]  # Shape: (M, M, d)
            # Assuming scalar noise covariance
            lr = self.learning_rates[i]
            sigma = self.noise_cov_matrix * np.ones(d) / (2 * lr)
            exponents = -0.5 * np.einsum('mij,j,mij->mi', diff, sigma, diff)
            logq[i] = logsumexp(exponents, axis=1)

        return diff_weights, logq, logp
