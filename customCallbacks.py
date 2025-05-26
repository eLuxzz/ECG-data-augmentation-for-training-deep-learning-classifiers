import tensorflow as tf

class UpdateDA(tf.keras.callbacks.Callback):
    def __init__(self, dataloader):
        self.dataloader = dataloader
    def on_epoch_end(self, epoch, logs=None):
        self.dataloader.update_DAMethod_index()
        self.dataloader.current_epoch = epoch
    def on_batch_end(self, epoch, logs=None):
        self.dataloader.update_DAMethod_index()