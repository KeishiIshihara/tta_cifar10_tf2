import numpy as np
import tensorflow as tf


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size, shuffle=True):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.steps_per_epoch = int(np.ceil(len(self.x) / float(self.batch_size)))
        self.index_array = np.arange(len(self.x))
        if shuffle:
            self.index_array = np.random.permutation(len(self.x))
        self.tmp = None

    def __getitem__(self, idx):
        batch_indices = self.index_array[self.batch_size * idx:
                                        self.batch_size * (idx + 1)]
        batch_x = self.x[batch_indices]
        batch_y = self.y[batch_indices]
        batch_x = self.transform_data(batch_x)
        return batch_x, batch_y

    def __len__(self):
        return self.steps_per_epoch

    def transform_data(self, batch_x):
        return batch_x


class DataLoaderTransform(DataLoader):
    def __init__(self, transform_list, transform_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform_list = transform_list
        self.transform_rate = transform_rate

    def transform_data(self, batch_x):
        for transformer in self.transform_list:
            if self.transform_rate > np.random.rand():
                batch_x = transformer(batch_x)
        return batch_x
