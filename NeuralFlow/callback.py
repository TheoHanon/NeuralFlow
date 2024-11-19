import tensorflow as tf

class NFCallback(tf.keras.callbacks.Callback):
    def __init__(self, noise_stddev, memory_epochs = None, dataset_train = None, loss_fn = None):
        """
        Custom callback to add noise to weights after gradient updates.

        Args:
            noise_stddev: Standard deviation of the Gaussian noise.
        """
        super(NFCallback, self).__init__()

        self.noise_stddev = noise_stddev
        self.memory_epochs = memory_epochs
        
        x_train, y_train = dataset_train

        self.x_train = x_train if x_train is not None else None
        self.y_train = y_train if y_train is not None else None
        self.loss_fn = loss_fn

        self.weights_before = {}
        self.weights_after = {}
        self.learning_rate = {}
        self.losses = {}


    def on_epoch_end(self, epoch, logs=None):
        
        epoch += 1
        lr = self.model.optimizer.learning_rate

        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            self.learning_rate[epoch] = float(lr(self.current_epoch))
        else :
            self.learning_rate[epoch] = float(lr.numpy())

        if epoch in self.memory_epochs:
            weights_before_noise = self._get_weights()
            self.weights_before[epoch] = weights_before_noise

        for var in self.model.trainable_variables:
            noise = tf.cast(tf.sqrt(2 * self.learning_rate[epoch] * self.noise_stddev**2), var.dtype) * tf.random.normal(shape=tf.shape(var), mean=0.0, stddev=1.0, dtype=var.dtype)
            var.assign_add(noise)

        if epoch in self.memory_epochs:
            weights_after_noise = self._get_weights()
            self.weights_after[epoch] = weights_after_noise

            if self.x_train is not None and self.y_train is not None:
                y_pred = self.model(self.x_train)
                self.losses[epoch] = self.loss_fn(self.y_train, y_pred).numpy()


    def _get_weights(self):
        # Helper method to get a copy of the current weights
        return [w.copy() for w in self.model.get_weights()]
    

