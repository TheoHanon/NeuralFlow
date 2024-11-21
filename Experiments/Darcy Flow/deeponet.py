import tensorflow as tf
import numpy as np
from typing import Union
import math

class CosineAnnealingSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0):
        super(CosineAnnealingSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha  # The minimum learning rate fraction (alpha * initial_learning_rate)

    def __call__(self, step):
        cosine_decay = 0.5 * (1 + tf.cos(math.pi * (step / self.decay_steps)))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.initial_learning_rate * decayed


class DeepONet(tf.keras.Model):
    def __init__(self, 
                 n_branch:int, 
                 n_trunk : int, 
                 width:int, 
                 depth:int, 
                 output_dim:int, 
                 activation : Union[str, callable], 
                 x_bounds : tuple = (0, 1),
                 t_bounds : tuple = (0, 1), 
                 **kwargs
                 ) -> None:
        """
        DeepONet model
        :param n_branch: number of branches
        :param width: width of each layer
        :param depth: depth of each branch
        :param output_dim: output dimension
        :param activation: activation function        
        """


        super(DeepONet, self).__init__()

        self.n_branch = n_branch
        self.n_trunk = n_trunk
        self.width = width
        self.depth = depth
        self.output_dim = output_dim
        self.activation = activation
        
        self.x_bounds = x_bounds
        self.t_bounds = t_bounds
    
        self.branch_net = CNNBranchNet2D(in_shape = (self.n_branch, self.n_branch), output_dim = self.output_dim, activation = self.activation)
        self.trunk_net = MLPTrunkNet2D(in_shape = (self.n_trunk,) , hidden_size = self.width, output_size = self.output_dim, depth = self.depth, activation = self.activation, final_activation=True)

        self.bias = tf.Variable([1.0])


    def call(self, inputs : tf.Tensor) -> tf.Tensor:
        """
        Forward pass
        :inputs: shape 
        :return: shape (batch, n_trunk)
        """

        x_branch, x_trunk = inputs

        bounds = (x_trunk[..., 0] - self.x_bounds[0]) * (x_trunk[..., 0] - self.x_bounds[1]) * (x_trunk[..., 1] - self.t_bounds[0]) * (x_trunk[..., 1] - self.t_bounds[1]) 
        branch_out = self.branch_net(x_branch) # (batch, output_dim)
        # branch_out = tf.expand_dims(branch_out, axis = 1)
    
        trunk_out = self.trunk_net(x_trunk) # (batch, output_dim)  
  
        return (tf.reduce_sum(branch_out * trunk_out, axis = -1)) * bounds # (batch, outup_dim)


    def get_config(self):
        config = super(DeepONet, self).get_config()
        config.update({
            "n_branch": self.n_branch,
            "n_trunk": self.n_trunk,
            "width": self.width,
            "depth": self.depth,
            "output_dim": self.output_dim,
            "activation": self.activation,
            "grid_size": self.grid_size,
            "x_bounds": self.x_bounds,
            "t_bounds": self.t_bounds,
            # "output_shape" : self.output_shape
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    
class CNNBranchNet2D(tf.keras.Model):

    def __init__(self, in_shape: tuple, output_dim: int, activation: Union[str, callable]) -> None:
        """
        CNN-based branch network for DeepONet (2D)
        :param input_shape: shape of the input (2D structure)
        :param output_dim: output dimension
        :param activation: activation function
        """
        super(CNNBranchNet2D, self).__init__()

        self.in_shape = in_shape
        self.output_dim = output_dim
        self.activation = activation

        seed = np.random.randint(0, 10000)
        initializer = tf.keras.initializers.GlorotUniform(seed =seed)

        self.cnn = tf.keras.Sequential([
            tf.keras.Input(shape=in_shape),
            tf.keras.layers.Reshape((*self.in_shape, 1)),
            tf.keras.layers.Conv2D(16, kernel_size=3, padding="same", activation=activation, kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            

            tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation=activation, kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            

            tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation=activation, kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation = activation, kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(output_dim, kernel_initializer=initializer)
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass for CNN branch
        :param x: input tensor
        :return: output tensor
        """
        return self.cnn(x)



class MLPTrunkNet2D(tf.keras.Model):

    def __init__(self, in_shape:int, hidden_size : int, output_size : int, depth:int, activation : Union[str, callable], final_activation : bool) -> None:
        """
        Multi-layer perceptron model
        :param input_size: input size
        :param hidden_size: hidden size
        :param output_size: output size
        :param activation: activation function
        """
        super(MLPTrunkNet2D, self).__init__()

        self.in_shape = in_shape
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.activation = activation

        seed = np.random.randint(0, 10000)
        initializer = tf.keras.initializers.GlorotUniform(seed =seed)
        
        self.mlp = tf.keras.Sequential()
        self.mlp.add(tf.keras.Input(shape=self.in_shape))
        
        for _ in range(depth):
            self.mlp.add(tf.keras.layers.Dense(hidden_size, activation=activation, kernel_initializer=initializer))

        if final_activation:
            self.mlp.add(tf.keras.layers.Dense(output_size, activation=activation, kernel_initializer=initializer))
        else:
            self.mlp.add(tf.keras.layers.Dense(output_size, activation=None, kernel_initializer=initializer))
    
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass
        :param x: input tensor
        :return: output tensor
        """

        return self.mlp(x)
        