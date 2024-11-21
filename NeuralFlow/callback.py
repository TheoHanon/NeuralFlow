import tensorflow as tf

class NFCallback(tf.keras.callbacks.Callback):
    def __init__(self, noise_stddev, x, y = None, memory_epochs = None,  loss_fn = None):
        """
        Custom callback to add noise to weights after gradient updates.

        Args:
            noise_stddev: Standard deviation of the Gaussian noise.
        """
        super(NFCallback, self).__init__()

        self.noise_stddev = noise_stddev
        self.memory_epochs = memory_epochs
        
        self.x = x 
        self.y = y

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

            if self.x is not None and self.y is not None:
                y_pred = self.model(self.x)
                self.losses[epoch] = self.loss_fn(self.y, y_pred).numpy()

            elif self.y is None:
                loss = 0
                count = 0
                for (inputs, y_true) in self.x:
                    count +=1
                    y_pred = self.model(inputs)
                    loss += self.loss_fn(y_true, y_pred)

                self.losses[epoch] = loss / count



    def _get_weights(self):
        # Helper method to get a copy of the current weights
        return [w.copy() for w in self.model.get_weights()]
    

