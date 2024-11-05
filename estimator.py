import numpy as np
import tensorflow as tf
from Models.parallel_model import ParallelModel

class Ensemble:

    def __init__(self, weights: tf.Tensor):
        """
        Args:
            sample_weight: tf.Tensor, shape (nStep, M, d) : Weights
            model: Model : Model
        """

    
        self.nStep = weights.shape[0]
        self.M = weights.shape[1]
        self.d = weights.shape[2]

        self.weights = weights

    def estimate(self, x_pred: tf.Tensor, model : ParallelModel):
        """
        Args:
            x_pred: tf.Tensor, shape (B, n) : Input

        Returns:
            tf.Tensor, shape (nStep, B, m) : Mean of the samples
        """
        mean = np.zeros((self.nStep, x_pred.shape[0], model.output_shape[1]))
        var  = np.zeros((self.nStep, x_pred.shape[0], model.output_shape[1])) 
        
        for t in range(self.nStep):
            model.set_weights(self.weights[t])
            y_pred = model(x_pred)
            mean[t] = tf.reduce_mean(y_pred, axis = 0)
            var[t]  = tf.reduce_mean((y_pred - mean[t][None, ...])**2, axis = 0)

        mean = mean.squeeze()
        var = var.squeeze()

        return mean, var
    

class ImportanceSampling:

    def __init__(self, weights : tf.Tensor, logq: tf.Tensor, logp : tf.Tensor) -> None:

        self.weights = weights
        self.logq = logq
        self.logp = logp

        self.nStep = self.logq.shape[0]
        self.M = self.logq.shape[1]

        self.importance_weight = np.zeros_like(self.logq)
        
        for t in range(self.logq.shape[0]):
            log_ratio = self.logp[t] - self.logq[t]
            self.importance_weight[t] = tf.exp(tf.clip_by_value(log_ratio - tf.reduce_logsumexp(log_ratio, axis = 0), clip_value_min = -500, clip_value_max = 500))

    def estimate(self, x_pred : tf.Tensor, model : ParallelModel):

        mean = np.zeros((self.nStep, x_pred.shape[0], model.output_shape[1]))
        var  = np.zeros((self.nStep, x_pred.shape[0], model.output_shape[1]))

        for t in range(self.nStep):
            model.set_weights(self.weights[t])
            y_pred = model(x_pred)

            mean[t] = tf.reduce_sum(y_pred * self.importance_weight[t][:, None, None], axis = 0)
            var[t]  = tf.reduce_sum((y_pred - mean[t][None, ...])**2 * self.importance_weight[t][:, None, None], axis = 0)

        mean = mean.squeeze()
        var = var.squeeze()

        return mean, var
    
            
    




