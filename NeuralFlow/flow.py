import tensorflow as tf
import numpy as np
from tqdm.notebook import tqdm
import estimator as est
from callback import NFCallback


class Flow_v2:

    def __init__(self, 
                model_fn : callable,
                n_models : int,
                prior = None, #TODO : equivalent to regularizer
                lam : float = 1.0,
                noise_stddev: float = 1.0) -> None:
        

        self.n_models = n_models
        self.model_fn = model_fn
        self.loss_fn = None
        self.models = []
        self.callbacks = []

        self.prior = prior
        self.lam = lam
        self.noise_stddev = noise_stddev
        self.n_epochs_recorded = None


        self.weights_before_noise = None
        self.weights_after_noise  = None
        self.learning_rates_noise = None
        self.losses = None



    def _get_scaled_loss(self, loss_fn : callable) -> callable:

        def scaled_loss(y_true, y_pred):
            return (1/(2*self.lam**2)) * loss_fn(y_true, y_pred)
        
        return scaled_loss

    def compile(self, optimizer_fn : callable, loss_fn : callable, metrics = None):

        self.loss_fn = self._get_scaled_loss(loss_fn)

        for _ in range(self.n_models):
            model = self.model_fn()
            optimizer = optimizer_fn()
            model.compile(optimizer = optimizer, loss = self.loss_fn, metrics = metrics)
            self.models.append(model)
            

    def fit(self, x, y, epochs, batch_size, memory_epochs = -1, validation_set = None, callbacks = None, verbose = 2):

        if memory_epochs == -1:
            memory_epochs = [epochs]

        self.n_epochs_recorded = len(memory_epochs)
        self.weights_before_noise = {epoch: [] for epoch in memory_epochs}
        self.weights_after_noise = {epoch: [] for epoch in memory_epochs}
        self.learning_rates_noise = {epoch: [] for epoch in memory_epochs}
        self.losses = {epoch: [] for epoch in memory_epochs}


        for model in self.models:

            nf_callback = NFCallback(self.noise_stddev, memory_epochs = memory_epochs, loss_fn = self.loss_fn, dataset_train = (x, y))
            self.callbacks.append(nf_callback)

            callbacks = (callbacks or []) + [nf_callback]
            model.fit(x = x, y = y, batch_size = batch_size, epochs = epochs, callbacks = callbacks, validation_data = validation_set, verbose = verbose)


            for epoch in memory_epochs:
                self.weights_after_noise[epoch].append(nf_callback.weights_after[epoch])
                self.weights_before_noise[epoch].append(nf_callback.weights_before[epoch])
                self.learning_rates_noise[epoch].append(nf_callback.learning_rate[epoch])
                self.losses[epoch].append(nf_callback.losses[epoch])


    def compute_distribution(self):
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
        
        wBefore = [ [] for _ in range(self.n_epochs_recorded)]
        wAfter = [ [] for _ in range(self.n_epochs_recorded)]
        logp = []
        lr = []
    

        for iRecord, (learning_rate, loss, before_weights_list, after_weights_list) in enumerate(zip(self.learning_rates_noise.values(), self.losses.values(), self.weights_before_noise.values(), self.weights_after_noise.values())):
            for iModel in range(self.n_models):

                flattened_before = np.concatenate([w.flatten() for w in before_weights_list[iModel]], axis=0)
                flattened_after = np.concatenate([w.flatten() for w in after_weights_list[iModel]], axis=0)

                wBefore[iRecord].append(flattened_before)
                wAfter[iRecord].append(flattened_after)

            wBefore[iRecord] = np.stack(wBefore[iRecord], axis=0) # Shape : (n_epochs_recorded, n_models, n_weights)
            wAfter[iRecord] = np.stack(wAfter[iRecord], axis=0) # Shape : (n_epochs_recorded, n_models, n_weights)

            logp.append(loss)
            lr.append(learning_rate)

        wBefore = np.array(wBefore)
        wAfter = np.array(wAfter)
        logp = -np.stack(logp, axis=0) 
        lr = np.unique(np.stack(lr, axis=0), axis=1)

        logq = np.zeros((self.n_epochs_recorded, self.n_models))
        d = wBefore.shape[-1]

        for i, (wbefore, wafter) in enumerate(zip(wBefore, wAfter)):

            diff = wafter[:, None, :] - wbefore[None, :, :]  # Shape: (n_models, n_models, d)
            sigma = np.ones(diff.shape[-1]) / (lr[i] * self.noise_stddev**2)
            exponents = -0.5 * np.einsum('mij,j,mij->mi', diff, sigma, diff)
            logq[i] = tf.reduce_logsumexp(exponents, axis=1)# - tf.cast(diff.shape[-1]/2 * tf.math.log(2*tf.constant(np.pi)*lr[i]*self.noise_stddev**2*self.n_models), exponents.dtype)

        return logp, logq
    

    def get_estimator(self, name: str, n_models : int = -1):
        """
        Get the estimator with the given name.

        Args:
            name (str): Name of the estimator.

        Returns:
            Estimator: Estimator object.
        """
        if name == 'deep_ensemble':
            return est.Ensemble(self.weights_before_noise, self.models, n_models)
        
        elif name == 'importance_sampling':
            logp, logq = self.compute_distribution()
            return est.get_estimator(name, self.weights_before_noise, logq, logp, self.models, n_models)
    
        else:
            raise ValueError(f"Invalid estimator name: {name}")
