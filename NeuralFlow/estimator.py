import numpy as np
import tensorflow as tf
from parallel_model import ParallelModel
from typing import List

def get_estimator(name: str, *args, **kwds):
    if name == 'deep_ensemble':
        return Ensemble(*args, **kwds)
    elif name == 'importance_sampling':
        return ImportanceSampling(*args, **kwds)
    else:
        raise ValueError(f"Invalid estimator name: {name}")
    
class Ensemble:

    def __init__(self, weights: tf.Tensor, models : List[tf.keras.Model], n_models : int = -1) -> None:
        """
        Args:
            sample_weight: tf.Tensor, shape (nStep, M, d) : Weights
            model: Model : Model
        """

        self.weights = weights
        self.models = models[:n_models]
        self.n_models = n_models
    

    def __call__(self, x_pred : tf.Tensor, *args, **kwds) :
        
        mean = []
        var  = []

        for i, epoch in enumerate(self.weights.keys()):
            
            y_pred = []

            for model_weights, model in zip(self.weights[epoch], self.models):
                model.set_weights(model_weights)
                y_pred.append(model(x_pred))

            mean.append(tf.reduce_mean(y_pred, axis = 0))
            var.append(tf.reduce_mean((y_pred - mean[i][None, ...])**2, axis = 0))

        return np.array(mean), np.array(var)


class ImportanceSampling:

    def __init__(self, weights : tf.Tensor, logq: tf.Tensor, logp : tf.Tensor,  models : List[tf.keras.Model], n_models : int = -1) -> None:

        self.weights = weights
        self.logq = logq[:, :n_models]
        self.logp = logp[:, :n_models]
        self.models = models[:n_models]
        self.M = n_models
        self.nStep = self.logq.shape[0]
    
        self.importance_weight = np.zeros_like(self.logq)
        self._precompute_importance_weight()
        
    def _precompute_importance_weight(self) -> None: 
        for t in range(self.logq.shape[0]):
            log_ratio = self.logp[t] - self.logq[t]
            self.importance_weight[t] = tf.exp(tf.clip_by_value(log_ratio - tf.reduce_logsumexp(log_ratio, axis = 0), clip_value_min = -300, clip_value_max = 300))
        return 


    def __call__(self, x_pred : np.ndarray, *args, **kwds):
        
        mean = []
        var  = []

        for i, epoch in enumerate(self.weights.keys()):
            y_pred = []

            for model_weights, model in zip(self.weights[epoch], self.models):
                model.set_weights(model_weights)
                y_pred.append(model(x_pred))

    
            mean.append(tf.reduce_sum(y_pred * self.importance_weight[i][:, None, None], axis = 0))
            var.append(tf.reduce_sum((y_pred - mean[i][None, ...])**2 * self.importance_weight[i][:, None, None], axis = 0))

  
        return np.array(mean), np.array(var)
    
            
    




