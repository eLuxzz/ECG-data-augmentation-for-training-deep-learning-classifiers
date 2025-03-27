import tensorflow as tf
import gc

class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        #tf.keras.backend.clear_session()  # Clears TensorFlow backend session
        gc.collect()  # Runs garbage collection
class UpdateDA(tf.keras.callbacks.Callback):
    def __init__(self, dataloader):
        self.dataloader = dataloader
    def on_epoch_end(self, epoch, logs=None):
        self.dataloader.update_DAMethod_index()
        self.dataloader.current_epoch = epoch