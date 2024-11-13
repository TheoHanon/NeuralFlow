import tensorflow as tf
import numpy as np
from typing import Union


class DeepONet(tf.keras.Model):
    def __init__(self, n_branch:int, n_trunk : int, width:int, depth:int, output_dim:int, activation : Union[str, callable], boundary_fn : callable = None) -> None:
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
        self.boundary_fn = boundary_fn if boundary_fn is not None else lambda x: 1.0

        self.branch_net = MLP(input_size = self.n_branch, hidden_size = self.width, output_size = self.output_dim, depth = self.depth, activation = self.activation, final_activation=False)
        self.trunk_net = MLP(input_size = self.n_trunk, hidden_size = self.width, output_size = self.output_dim, depth = self.depth, activation = self.activation, final_activation=True)

        self.bias = tf.Variable([1.0])


    def call(self, x_branch: tf.Tensor, x_trunk: tf.Tensor) -> tf.Tensor:
        """
        Forward pass
        :param x_branch: input tensor for branch
        :param x_trunk: input tensor for trunk
        :return: output tensor
        """

        boundary_constraints = self.boundary_fn(x_trunk)
        
        branch_out = self.branch_net(x_branch)
        trunk_out = self.trunk_net(x_trunk)

        return (branch_out @ tf.transpose(trunk_out) + self.bias) * boundary_constraints



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

        self.mlp.add(tf.keras.layers.InputLayer(shape=(input_size,)))
        
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
        