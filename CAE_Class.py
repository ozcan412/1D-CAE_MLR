import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics




class CAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(CAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean()
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.val_loss_tracker,
        ]
    
    def call(self, inputs):
        x = self.encoder(inputs)
        return self.decoder(x)
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            H = self.encoder(data)
            reconstruction = self.decoder(H)
            total_loss=keras.metrics.mse(data, reconstruction)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            #"val_loss": self.val_loss_tracker.result(),
        }
    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        H = self.encoder(data)
        reconstruction = self.decoder(H)
        val_loss=keras.metrics.mse(data, reconstruction)
        self.val_loss_tracker.update_state(val_loss)
        return {
           # "loss": self.total_loss_tracker.result(),
            "val_loss": self.val_loss_tracker.result(),
            }
   
