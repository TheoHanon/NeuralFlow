import tensorflow as tf
import numpy as np
from typing import Union


class DeepONet(tf.keras.Model):
    def __init__(self, 
                 n_branch:int, 
                 n_trunk : int, 
                 width:int, 
                 depth:int, 
                 output_dim:int, 
                 activation : Union[str, callable], 
                 x_bounds : tuple = (0, 1.0),
                 t_bounds : tuple = (0, 1.0)
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


        self.branch_net = CNNBranchNet2D(input_shape = (self.n_branch, self.n_branch, 1), output_dim = self.output_dim, activation = self.activation)
        self.trunk_net = MLP(input_size = self.n_trunk, hidden_size = self.width, output_size = self.output_dim, depth = self.depth, activation = self.activation, final_activation=True)

        self.bias = tf.Variable([1.0])

    def compile(self, optimizer: tf.keras.optimizers.Optimizer, loss_fn) -> None:
        super(DeepONet, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def call(self, x_branch: tf.Tensor, x_trunk: tf.Tensor) -> tf.Tensor:
        """
        Forward pass
        :param x_branch: input tensor for branch
        :param x_trunk: input tensor for trunk
        :return: output tensor
        """

        x, y = x_trunk[..., 0], x_trunk[..., 1]
        bounds = (x - self.x_bounds[0]) * (x - self.x_bounds[1]) * (y - self.t_bounds[0]) * (y - self.t_bounds[1])

        branch_out = self.branch_net(x_branch)
        trunk_out = self.trunk_net(x_trunk)
        
        return (tf.reduce_sum(branch_out * trunk_out, axis=1, keepdims=True) + self.bias) * bounds
        

    

class CNNBranchNet2D(tf.keras.Model):

    def __init__(self, input_shape: tuple, output_dim: int, activation: Union[str, callable]) -> None:
        """
        CNN-based branch network for DeepONet (2D)
        :param input_shape: shape of the input (2D structure)
        :param output_dim: output dimension
        :param activation: activation function
        """
        super(CNNBranchNet2D, self).__init__()
        
        self.cnn = tf.keras.Sequential([
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation=activation),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation=activation),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(output_dim, activation=None)
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass for CNN branch
        :param x: input tensor
        :return: output tensor
        """
        return self.cnn(x)




class MLP(tf.keras.Model):

    def __init__(self, input_size:int, hidden_size : int, output_size : int, depth:int, activation : Union[str, callable], final_activation : bool) -> None:
        """
        Multi-layer perceptron model
        :param input_size: input size
        :param hidden_size: hidden size
        :param output_size: output size
        :param activation: activation function
        """
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.activation = activation

        self.mlp = tf.keras.Sequential()

        self.mlp.add(tf.keras.Input(shape=(input_size,)))
        
        for _ in range(depth):
            self.mlp.add(tf.keras.layers.Dense(hidden_size, activation=activation))

        if final_activation:
            self.mlp.add(tf.keras.layers.Dense(output_size, activation=activation))
        else:
            self.mlp.add(tf.keras.layers.Dense(output_size, activation=None))
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass
        :param x: input tensor
        :return: output tensor
        """
        return self.mlp(x)
        