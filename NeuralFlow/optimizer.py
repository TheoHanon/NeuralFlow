import tensorflow as tf

class DiffusionOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer_name, learning_rate, noise_stddev, memory_iterations=None, name="Flowpt", **kwargs):
        """
        Custom optimizer to add noise to weights after gradient updates and store memory for model recovery.

        Args:
            optimizer: Base optimizer (e.g., Adam, SGD).
            noise_stddev: Standard deviation of the Gaussian noise.
            memory_epochs: List of epochs where memory should be activated. Default is None (no memory).
            name: Name of the optimizer.
        """
        super().__init__(learning_rate, **kwargs)

        self.optimizer = tf.keras.optimizers.get({
            "class_name": optimizer_name,
            "config": {"learning_rate": learning_rate}
        })
        self.noise_stddev = noise_stddev
        self.memory_iterations = memory_iterations if memory_iterations is not None else []
        self.memory_before = {}  # Dictionary to store weights by epoch
        self.memory_after  = {}

    def apply_gradients(self, grads_and_vars, **kwargs):
        """
        Apply gradients, followed by adding noise to weights, with conditional memory storage.
        """
        # Apply the base optimizer's gradient step
        train_op = self.optimizer.apply_gradients(
            grads_and_vars, **kwargs
        )

        # Add noise to the weights after the update
        with tf.control_dependencies([train_op]):
            update_ops = []
            for grad, var in grads_and_vars:
                if grad is not None:  # Ensure gradient exists
                    current_iteration = tf.keras.backend.get_value(self.optimizer.iterations) 

                    if current_iteration in self.memory_iterations:
                        # Initialize memory for the current epoch if not already done
                        if current_iteration not in self.memory_before:
                            self.memory_before[current_iteration] = {}

                        # Save the current weight (after optimizer step and before diffusion)
                        self.memory_before[current_iteration][var.name] = tf.identity(var).numpy()

                    # Add diffusion noise
                    noise = tf.random.normal(shape=tf.shape(var), mean=0.0, stddev=self.noise_stddev)
                    update_ops.append(var.assign_add(noise))

                    if current_iteration in self.memory_iterations:
                        if current_iteration not in self.memory_after:
                            self.memory_after[current_iteration] = {}

                        # Save the current weight (after optimizer step and before diffusion)
                        self.memory_after[current_iteration][var.name] = tf.identity(var).numpy()



            # Group all updates into a single operation
            return tf.group(*update_ops)

    def get_config(self):
        """
        Save optimizer configuration.
        """
        config = super().get_config()
        config.update({
            "optimizer": tf.keras.optimizers.serialize(self.optimizer),
            "noise_stddev": self.noise_stddev,
            "memory_epochs": self.memory_epochs,
        })
        return config

    @classmethod
    def from_config(cls, config):
        optimizer = tf.keras.optimizers.deserialize(config.pop("optimizer"))
        return cls(optimizer=optimizer, **config)

    def get_memory(self):
        """
        Retrieve the stored memory for analysis or model recovery.
        """
        return self.memory_before, self.memory_after
