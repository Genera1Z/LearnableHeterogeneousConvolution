import numpy as np
import tensorflow.keras.datasets as KD
import tensorflow.keras.preprocessing.image as KPI
import tensorflow.keras.utils as KU


class Cifar10:
    CATES = ("airplane", "automobile", "bird", "cat", "dear", "dog", "frog", "horse", "ship", "truck")
    NCATE = 10
    DSIZE = (32, 32, 3)
    TRAIN_SET, VAL_SET = KD.cifar10.load_data()

    def __init__(self):
        self.ds_train, self.ds_val = self.TRAIN_SET, self.VAL_SET
        self.ds_train = self.preprocess(*self.ds_train)
        self.ds_val = self.preprocess(*self.ds_val)

    @staticmethod
    def preprocess(data, label):
        data = Cifar10.mean_sub(data.astype('float32')) / 255
        label = KU.to_categorical(label, 10)
        return data, label

    @staticmethod
    def augment(data, label, batch_size):
        aug = KPI.ImageDataGenerator(width_shift_range=0.125, height_shift_range=0.125, horizontal_flip=True)
        aug.fit(data)
        dsgen = aug.flow(data, label, batch_size=batch_size)
        return dsgen

    def get(self, subset, nepoch, batch_size):
        if subset == 'train':
            return self.setup_t(nepoch, batch_size)
        elif subset == 'val':
            return self.setup_v(nepoch, batch_size)

    def setup_t(self, nepoch, bsize):
        ds_train = self.augment(*self.ds_train, bsize)
        setattr(ds_train, 'nstep', self.ds_train[0].shape[0] // bsize)
        return ds_train

    def setup_v(self, nepoch, bsize):
        # setattr(self.ds_val, 'nstep', self.ds_val[0].shape[0] // bsize)
        return self.ds_val

    @staticmethod
    def mean_sub(data):
        mean_r, mean_g, mean_b = np.mean(data, axis=(0, 1, 2))
        data[:, :, :, 0] -= mean_r
        data[:, :, :, 1] -= mean_g
        data[:, :, :, 2] -= mean_b
        return data
