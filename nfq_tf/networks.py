"""Networks for NFQ."""
import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import Dense, Input


class NFQNetwork(tf.keras.Model):
    def __init__(self):
        """Networks for NFQ."""
        super().__init__()

        # Initialize weights to [-0.5, 0.5]
        # TODO(seungjaeryanlee): What about bias?
        self.dense1 = Dense(
            5, kernel_initializer=RandomUniform(-0.5, 0.5), activation="sigmoid"
        )
        self.dense2 = Dense(
            5, kernel_initializer=RandomUniform(-0.5, 0.5), activation="sigmoid"
        )
        self.dense3 = Dense(
            1, kernel_initializer=RandomUniform(-0.5, 0.5), activation="sigmoid"
        )

        # Need to specify input to save
        self._set_inputs(Input(5))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward propagation.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of observation and action concatenated.

        Returns
        -------
        y : tf.Tensor
            Forward-propagated observation predicting Q-value.

        """
        return self.dense3(self.dense2(self.dense1(x)))
