import tensorflow as tf
import numpy as np


class ParallelModel(tf.keras.Model):
    """
    A Keras model that encapsulates multiple instances of a base model, each with its own set of weights,
    and trains and predicts them in parallel efficiently.

    Parameters:
    - base_model: A tf.keras.Model instance representing the base model architecture.
    - n_models: An integer specifying the number of model instances to create.
    """


    def __init__(self, base_model : tf.keras.Model, n_models: int):

        super(ParallelModel, self).__init__()
        self.n_models = n_models    
        self.models = [tf.keras.models.clone_model(base_model) for _ in range(n_models)]

        weight_vars = self.models[0].weights  # Assuming all models have the same architecture

        self.weight_shapes = [w.shape.as_list() for w in weight_vars]
        self.weight_sizes = [tf.reduce_prod(shape) for shape in self.weight_shapes]
        self.weight_sizes = [int(s) for s in self.weight_sizes]

        self.output_shape = self.models[0].output_shape


    @property
    def weights(self):
        return [tf.identity(tf.stop_gradient(tf.concat([tf.reshape(w, [-1]) for w in model.weights], axis=0))) for model in self.models]
    
    # @weights.setter
    def set_weights(self, weights):
        for model, w in zip(self.models, weights):
            splits = tf.split(w, self.weight_sizes, axis = 0)

            reshaped_w = [
                tf.reshape(split, shape)
                for split, shape in zip(splits, self.weight_shapes)
            ]
            for var, new_w in zip(model.weights, reshaped_w):
                var.assign(new_w)

    @tf.function
    def call(self, inputs):
        return [model(inputs) for model in self.models]


    
    





