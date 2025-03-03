import math
from tensorflow.keras.utils import Sequence
import numpy as np


class ECGSequence(Sequence):
    def __init__(self, signalData, annotations = None, batch_size=8):
        self.x = signalData
        self.y = annotations
        self.batch_size = batch_size

        self.end_idx = len(self.x)
        self.start_idx = 0

    @property
    def n_classes(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        if self.y is None:
            return np.array(self.x[start:end, :, :])
        else:
            return np.array(self.x[start:end, :, :]), np.array(self.y[start:end])

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)