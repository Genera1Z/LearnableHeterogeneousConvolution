import os

import tensorflow as tf
import tensorflow.keras.backend as KB
import tensorflow.keras.callbacks as KC
import tensorflow.keras.optimizers as KO

from src.datum import Cifar10
from src.globe import TCONVS, STAGES
from src.layer import LhcUpdate
from src.misc import ensure_fold, backup_fold
from src.model import MyConv2d, build_vgg16, ReduceLrOnPlateau, load, find_best


def train(stage, nepoch=100, batch_size=256, lr0=0.01, name='vgg16-cifar10'):
    print(f'\n{"@" * 16} STAGE {stage.upper()} {"@" * (100 - 16 - 8 - len(stage))}\n')

    KB.clear_session()
    tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))

    log_fold = f'res/logs/{name}-{stage}/'
    ckpt_fold = f'res/ckpt/{name}-{stage}/'

    ### init dataset and model

    cifar10 = Cifar10()
    ncate, dshape = cifar10.NCATE, cifar10.DSIZE
    diter_t, diter_v = [cifar10.get(_, nepoch, batch_size) for _ in ('train', 'val')]

    myconv2d = MyConv2d(TCONVS[1], formable=stage == STAGES[0], dt=0.1, cgi=32, cgo=16)
    kmdl = build_vgg16(myconv2d, dshape, ncate, name)

    if stage == STAGES[0]:
        print('Loading pretrained weights...')
        kmdl.load_weights('../../Models/KerasApplications/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
            by_name=True, skip_mismatch=True)
    elif stage == STAGES[1]:
        legacy = ckpt_fold + find_best(ckpt_fold, 2, 50, 'max')
        if os.path.isfile(legacy):
            print('Loading legacy best...', legacy)
            load(kmdl, legacy)
    else:
        raise NotImplemented

    ### train it

    [ensure_fold(p) for p in [log_fold, ckpt_fold]]
    [backup_fold(p) for p in [log_fold, ckpt_fold]]
    ckpt_file = ckpt_fold + '{epoch:03d}-{val_loss:.5f}-{val_acc:.5f}.h5'

    optim = KO.SGD(lr0, 0.9, True)  # , clipvalue=1e99)  # `distribute` not support `clipvalue`

    kmdl.compile(optim, loss='categorical_crossentropy', metrics=['acc'])
    kmdl.summary()

    callbacks = [
        KC.EarlyStopping('val_acc', patience=20, verbose=1, mode='max'),
        KC.ModelCheckpoint(ckpt_file, 'val_acc', 1, False, True, 'max'),
        ReduceLrOnPlateau(0.5, 'val_acc', 0.1, 20, 1, 'max', min_lr=1e-9),
        KC.TensorBoard(log_fold, histogram_freq=0, write_graph=True, write_images=True),
        LhcUpdate(20, 20)
    ]

    history = kmdl.fit(diter_t, None, None, nepoch, 1, callbacks, validation_data=diter_v,
        steps_per_epoch=diter_t.nstep, validation_steps=None,
        workers=16, use_multiprocessing=False)
    print(history)


def main():
    train(STAGES[0], 100, 256, 1e-1, 'vgg16-cifar10')
    train(STAGES[1], 200, 256, 1e-2, 'vgg16-cifar10')


if __name__ == '__main__':
    main()
