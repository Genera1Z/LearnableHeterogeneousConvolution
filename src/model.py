import os

import h5py
import numpy as np
import tensorflow.keras.backend as KB
import tensorflow.keras.callbacks as KC
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

from src.globe import TCONVS
from src.layer import Conv2dLhcf, Conv2dLhcr, HetConv2d


def find_best(ckpt_fold, idx, lower, mode, suffix='.h5'):
    fns = os.listdir(ckpt_fold)
    fns.sort()
    fns = fns[min(len(fns) - 1, lower):]
    names = [fn[:-len(suffix)] for fn in fns]
    values = np.array([fn.split('-')[idx] for fn in names])

    if mode == 'max':
        best_ckpt = fns[int(np.argmax(values))]
    elif mode == 'min':
        best_ckpt = fns[int(np.argmin(values))]
    else:
        raise NotImplemented
    return best_ckpt


def load(kmdl, path):
    # load the parts which have identical names and shapes: std->std; lhc-formable->lhc-formable
    kmdl.load_weights(path, True, True)

    file0 = file = h5py.File(path, 'r')
    if 'layer_names' not in file.attrs and 'model_weights' in file:
        file = file['model_weights']
    from tensorflow.python.keras.saving.hdf5_format import _legacy_weights, load_attributes_from_hdf5_group, \
        preprocess_weights_for_loading

    if 'keras_version' in file.attrs:
        original_keras_version = file.attrs['keras_version']  # .decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in file.attrs:
        original_backend = file.attrs['backend']  # .decode('utf8')
    else:
        original_backend = None

    layer_names = load_attributes_from_hdf5_group(file, 'layer_names')
    index = {}
    for layer in kmdl.layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    # load the remaining parts
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = file[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]

        layer = index.get(name, [])
        if len(layer) == 0:
            continue
        assert len(layer) == 1
        layer = layer[0]

        if type(layer) in (Conv2dLhcf, Conv2dLhcr):
            weight_values = preprocess_weights_for_loading(layer, weight_values, original_keras_version,
                original_backend)
            wdict = dict(zip(weight_names, weight_values))

            symbolic_weights = _legacy_weights(layer)
            symbol_names = [s.name for s in symbolic_weights]
            sdict = dict(zip(symbol_names, symbolic_weights))

            for pname in Conv2dLhcf.VAR_NAMES[:3]:
                # symb = [__ for _, __ in sdict.items() if _[:-2].endswith(pname)]
                symb = [__ for _, __ in sdict.items() if pname in _]
                # wght = [__ for _, __ in wdict.items() if _[:-2].endswith(pname)]
                wght = [__ for _, __ in wdict.items() if pname in _]
                assert len(symb) == 1 and len(wght) <= 1
                if len(wght) == 1:
                    weight_value_tuples.append((symb[0], wght[0]))

    KB.batch_set_value(weight_value_tuples)
    file0.close()


def MyConv2d(tconv, formable=False, dt=0.1, cgi=32, cgo=16):
    def func(filters, kernel_size, strides=(1, 1), padding='same', activation='relu', use_bias=True,
            kernel_initializer='he_normal', bias_initializer='zeros', kernel_regularizer=None,
            bias_regularizer=None, **kwargs):

        if tconv == TCONVS[0]:
            return KL.Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=activation,
                use_bias=use_bias, kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, **kwargs)

        elif tconv == TCONVS[1]:
            return Conv2dLhcf(filters, kernel_size, strides=strides, padding=padding, activation=activation,
                use_bias=use_bias, kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                formable=formable, dt=dt, cgi=cgi, cgo=cgo, **kwargs)

        elif tconv == TCONVS[2]:
            return Conv2dLhcr(filters, kernel_size, strides=strides, padding=padding, activation=activation,
                use_bias=use_bias, kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                formable=formable, dt=dt, cgi=cgi, cgo=cgo, **kwargs)

        elif tconv == TCONVS[3]:
            return HetConv2d(filters, kernel_size, strides=strides, padding=padding, activation=activation,
                use_bias=use_bias, kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, **kwargs)
        else:
            raise NotImplemented

    return func


class ReduceLrOnPlateau(KC.ReduceLROnPlateau):

    def __init__(self, warm_ratio, monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=1e-4,
            cooldown=0, min_lr=0, **kwargs):
        super(ReduceLrOnPlateau, self).__init__(monitor, factor, patience, verbose, mode, min_delta, cooldown, min_lr,
            **kwargs)
        self.warm_ratio = warm_ratio
        self.lr0 = None

    def on_epoch_begin(self, epoch, logs=None):
        super(ReduceLrOnPlateau, self).on_epoch_begin(epoch, logs)
        if epoch == 0:
            self.lr0 = KB.get_value(self.model.optimizer.lr)
            lr_warm = self.lr0 * self.warm_ratio
            KB.set_value(self.model.optimizer.lr, lr_warm)
        elif epoch == 1:
            KB.set_value(self.model.optimizer.lr, self.lr0)
        print("[ReduceLrOnPleatu.on_epoch_end] lr =", KB.get_value(self.model.optimizer.lr))


def build_vgg16(MyConv2d, ishape, nclass, name):
    input = KL.Input(ishape)  # 32

    x = KL.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal',
        name='block1_conv1')(input)
    x = MyConv2d(64, (3, 3), name='block1_conv2')(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)

    x = MyConv2d(128, (3, 3), name='block2_conv1')(x)
    x = MyConv2d(128, (3, 3), name='block2_conv2')(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)

    x = MyConv2d(256, (3, 3), name='block3_conv1')(x)
    x = MyConv2d(256, (3, 3), name='block3_conv2')(x)
    x = MyConv2d(256, (3, 3), name='block3_conv3')(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)

    x = MyConv2d(512, (3, 3), name='block4_conv1')(x)
    x = MyConv2d(512, (3, 3), name='block4_conv2')(x)
    x = MyConv2d(512, (3, 3), name='block4_conv3')(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x)  # 2

    x = MyConv2d(512, (3, 3), name='block5_conv1')(x)
    x = MyConv2d(512, (3, 3), name='block5_conv2')(x)
    x = MyConv2d(512, (3, 3), name='block5_conv3')(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block5_pool')(x)  # 2*2 -> 1*1

    x = KL.GlobalAvgPool2D()(x)

    # output = KL.Dense(nclass, activation='softmax', name='predictions')(x)
    # XXX in case `mixed_precision` being used
    output = KL.Dense(nclass, activation=None, name='predictions')(x)
    output = KL.Activation('softmax', dtype='float32')(output)
    kmdl = KM.Model(input, output, name=name)

    return kmdl
