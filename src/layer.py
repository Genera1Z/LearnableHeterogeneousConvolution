import numpy as np
import tensorflow as tf
import tensorflow.keras.activations as KA
import tensorflow.keras.callbacks as KC
import tensorflow.keras.layers as KL
import tensorflow.keras.utils as KU


class LhcProbe:
    @staticmethod
    def count_weights(masks):
        nwght_ttl = masks.size  # np.prod(masks.shape)
        nwght_vld = np.sum(masks)
        return nwght_ttl, nwght_vld

    @staticmethod
    def _gen_shape_codes(ksize=3):
        slen = ksize ** 2
        _shapes = np.random.randint(0, 2, (slen, 1000000))
        _scodes = list(np.unique(np.sum([_shapes[_] * 10 ** _ for _ in range(slen)], 0)))
        scodes = list()
        temp = list()
        for i in range(slen + 1):
            for j, scode in enumerate(_scodes):
                if j not in temp and str(scode).count('1') == i:
                    temp.append(j)
                    scodes.append(scode)
        return np.array(scodes)

    SCODES = _gen_shape_codes.__func__()

    @staticmethod
    def count_shapes(masks):
        # 0 make sure elements in masks are binary
        assert len(np.where(masks == 0)[0]) + len(np.where(masks == 1)[0]) == masks.size
        masks = masks.astype(np.uint8)
        # 1 encode shapes into scalars to simplify the counting
        shapes_flat = np.reshape(masks, (-1, *masks.shape[2:4]))  # (9, ci, co)
        shapes_enc = np.sum([shapes_flat[_] * 10 ** _ for _ in range(shapes_flat.shape[0])], 0)  # (ci, co)
        # 2 count the number of shapes using np.unique()
        nshape_tuple = np.unique(shapes_enc, return_counts=True)
        # 3 supplement the shape codes not presenting here
        nshape_dict = dict(zip(LhcProbe.SCODES, np.zeros([len(LhcProbe.SCODES)])))
        nshape_dict.update(dict(zip(*nshape_tuple)))
        # 4 return the number of shapes
        nshape_list = list(nshape_dict.values())
        return nshape_list

    @staticmethod
    def calc_masks(effects, ci, co, cgi, cgo):
        effects = tf.stop_gradient(effects)
        if len(effects.shape) == 4:
            masks = Conv2dLhcf.calc_masks(effects, ci, co, cgi, cgo)
        elif len(effects.shape) == 3:
            masks = Conv2dLhcr.calc_masks(effects, ci, co, cgi, cgo)
        else:
            raise NotImplemented
        return masks.numpy().astype(np.uint8)


class LhcUpdate(KC.Callback):
    """Update variables in all LHC layers."""
    TRATIOS = (0.01, 0.99)

    def __init__(self, nwarm_e, nwarm_r):
        """
        :param nwarm_e: the number of epochs for mask enabling warmup;
            better not greater than the learning rate decay patience
        :param nwarm_r: the number of epochs for mask regularizing warmup;
            better not greater than the early stopping patience
        """
        super(LhcUpdate, self).__init__()
        self.nwarm_e = nwarm_e
        self.delta_e = 1 / nwarm_e  # the incremental quantity per epoch for mask enabling

        self.nwarm_r = nwarm_r
        # `1` is the target loss weight for mask regularization, relative to the task loss
        self.delta_r = 4 / nwarm_r  # the incremental quantity per epoch for mask regularizing  # XXX

        self.loss_r9 = None  # the maximum of mask regularizing
        self.scale0 = 0.0  # to scale the initial mask regularization loss to be equal to the task loss
        self.scale1 = tf.Variable(0, False, dtype=tf.float32)  # scale factor variable for warmup
        # cmpt_dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        # self.scale1 = tf.Variable(0, False, dtype=cmpt_dtype)  # scale factor variable for warmup

        self.layers = list()  # all LHC layers
        self.formable = None  # whether to fix `masks` in LHC layers

    def on_train_begin(self, logs=None):
        self.layers = [_ for _ in self.model.layers if hasattr(_, 'masks')]

        mean_formable = np.mean([_.formable for _ in self.layers])
        assert mean_formable in [0, 1]
        self.formable = mean_formable == 1

        if not self.formable:
            return

        dt_all = [_.dt for _ in self.layers]
        assert np.std(dt_all) < 1e-3  # only support all-same `dt`s
        dt = self.layers[0].dt

        self.loss_r9 = self.TRATIOS[0] + max(abs(1 - dt), dt) * self.TRATIOS[1]
        if dt < 0:
            print('[LhcUpdate.on_train_begin] dt:', dt, ', no `mrglr_loss`!')
            return

        # in case `mixed_precision` being used
        # cmpt_dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype

        # mrglr_func = lambda: tf.cast(self.scale1, cmpt_dtype) * self.mask_regularization_loss(self.layers, dt)
        mrglr_func = lambda: self.scale1 * tf.cast(self.mask_regularization_loss(self.layers, dt), self.scale1.dtype)
        self.model.add_loss(mrglr_func)

    def on_epoch_begin(self, epoch, logs=None):
        if not self.formable:
            return

        if epoch <= self.nwarm_e:  # update mask enabling
            prob = (epoch * self.delta_e) if epoch < self.nwarm_e else (1.0 + 1e-1)  # TODO 1/np.power(1.25,n0-n)
            for layer in self.layers:
                vector = np.where(np.random.rand(1, 1, 1, layer.co) < prob, True, False)
                layer.vector.assign(vector)
            print('[LhcUpdate.on_epoch_begin] prob:', prob, epoch)

        if epoch <= self.nwarm_r:  # update mask regularizing
            factor = epoch * self.delta_r * self.scale0
            self.scale1.assign(factor)
            print('[LhcUpdate.on_epoch_begin] epoch * self.delta_r:', epoch * self.delta_r, self.scale1, epoch)

    def on_epoch_end(self, epoch, logs=None):
        if not self.formable:
            return

        if epoch == 0:
            loss_t = logs['loss']  # the initial task loss
            self.scale0 = loss_t / self.loss_r9
            print('[LhcUpdate.on_epoch_begin] scale0:', self.scale0, 'loss_r9:', self.loss_r9, epoch)

    @staticmethod
    def mask_regularization_loss(layers, dt, ratios=TRATIOS):
        assert 0 < dt < 1
        fvld_all, fttl_all = list(), list()
        dvld_all, dttl_all = list(), list()

        for layer in layers:
            nwght_vld = tf.reduce_sum(layer.masks)
            nwght_ttl = np.prod(layer.masks.shape)

            howo = np.prod(layer.output.shape[1:3])

            fvld_all.append(nwght_vld * howo)
            fttl_all.append(nwght_ttl * howo)

            dvld_all.append(nwght_vld)
            dttl_all.append(nwght_ttl)

        mr_f = sum(fvld_all) / sum(fttl_all)
        mr_d = tf.abs(dt - sum(dvld_all) / sum(dttl_all))
        return mr_f * ratios[0] + mr_d * ratios[1]


class Conv2dLhc(KL.Layer):
    """Learnable Heterogeneous Convolution - Base Class"""
    VAR_NAMES = ('_W_', 'effect', '_b_', 'vector')

    def __init__(self, filters, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=None, use_bias=True,
            kernel_initializer='he_normal', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
            formable=False, dt=-1, cgi=1, cgo=1, **kwargs):
        """
        :param formable: whether to fix `masks` or not
        :param dt: the density target, normalized
        :param cgi: the number of adjacent slices that have the same shape
        :param cgo: the number of adjacent kernels that have the same shape
        """
        super(Conv2dLhc, self).__init__(**kwargs)
        self.ci, self.co = -1, filters
        assert kernel_size[0] == kernel_size[1] and len(kernel_size) == 2

        self.ksize = kernel_size
        self.stride = strides
        self.padd = padding.upper()
        self.act = activation
        self.use_bias = use_bias

        self.kernels, self.kinitr, self.krglrr = None, kernel_initializer, kernel_regularizer
        self.biases, self.binitr, self.brglrr = None, bias_initializer, bias_regularizer

        self.formable = formable
        self.dt = dt
        assert 0 < dt < 1 or dt == -1
        self.cgi, self.cgo = cgi, cgo

        self.masks = None
        self.effects, self.einitr, self.erglrr = None, 'he_normal', None
        self.vector = None

    def build(self, ishape):
        """XXX dynamic graph, invoked once on epoch begin?
        """
        self.ci = int(ishape[3])

        self.kernels = self.add_weight(self.VAR_NAMES[0], (*self.ksize, self.ci, self.co), tf.float32,
            self.kinitr, self.krglrr, True)

        self.effects = self.add_effects(self.VAR_NAMES[1], self, self.einitr, self.erglrr, self.formable)

        if self.use_bias:
            self.biases = self.add_weight(self.VAR_NAMES[2], (self.co,), tf.float32,
                self.binitr, self.brglrr, True)

        super(Conv2dLhc, self).build(ishape)

        if self.formable:
            self.vector = tf.Variable(np.ones([1, 1, 1, self.co], np.bool), trainable=False, name=self.VAR_NAMES[3])

            self._non_trainable_weights.remove(self.vector)
            # for i, _ in enumerate(self._non_trainable_weights):
            #     if _.name == self.vector.name:
            #         self._non_trainable_weights.pop(i)

    def call(self, inputs, **kwargs):
        """XXX static graph, invoked twice on epoch begin?, due to its decoration by `tf.function`!
        """
        self.masks = self.calc_masks(self.effects, self.ci, self.co, self.cgi, self.cgo)

        if self.formable:
            masks_b = tf.ones_like(self.masks)
            self.masks = tf.where(self.vector, self.masks, masks_b)

        kernels_mask = self.kernels * self.masks  # TODO use `self.kernels_mask` to add loss XXXXXXXXXXXXXXXXXXXXXXXXXXX
        output = tf.nn.conv2d(inputs, kernels_mask, [1, *self.stride, 1], self.padd)

        if self.use_bias:
            output += self.biases[None, None, None]

        if self.act:
            output = KA.get(self.act)(output)

        return output

    def compute_output_shape(self, ishape):
        space_new = [KU.conv_utils.conv_output_length(_, self.ksize[0], self.padd.lower(), self.stride[0], 1)
            for _ in ishape[1:-1]]
        return tf.TensorShape([ishape[0]] + space_new + [self.co])

    def get_config(self):
        config = super(Conv2dLhc, self).get_config()
        return config

    @staticmethod
    def add_effects(name, self, initr, rglzr, trainable):
        raise NotImplemented

    @staticmethod
    def calc_masks(effects, ci, co, cgi, cgo):
        raise NotImplemented


class Conv2dLhcf(Conv2dLhc):
    """Learnable Heterogeneous Convolution - Free Shapes"""

    @staticmethod
    def add_effects(name, self, initr, rglzr, trainable):
        ngi, ngo = [int(np.ceil(i / o)) for i, o in ([self.ci, self.cgi], [self.co, self.cgo])]
        shape = (*self.ksize, ngi, ngo)
        effects = self.add_weight(name, shape, tf.float32, initr, rglzr, trainable)
        return effects

    @staticmethod
    def calc_masks(effects, ci, co, cgi, cgo):
        effects_norm_ = Conv2dLhcf.step_func(effects)[:, :, :, None, :, None]  # (3, 3, ci / cgi, 1, co / cgo, 1)
        effects_tile_ = tf.tile(effects_norm_, [1, 1, 1, cgi, 1, cgo])
        a, b, c, d = effects.shape
        effects_tile = tf.reshape(effects_tile_, [a, b, c * cgi, d * cgo])
        effects_norm = effects_tile[:, :, :ci, :co]  # in case `ci % cgi != 0` or `co % cgo != 0`
        masks = effects_norm
        return masks

    @staticmethod
    @tf.custom_gradient
    def step_func(xi):
        assert len(xi.shape) == 4
        xo = tf.where(tf.greater(xi, 0), tf.ones_like(xi), tf.zeros_like(xi))

        def grad(dy):  return tf.where(tf.less(tf.abs(xi), 1), dy, 0.1 * dy)

        return xo, grad


class Conv2dLhcr(Conv2dLhc):  # TODO XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    """Learnable Heterogeneous Convolution - Rigid Shapes"""

    GROUPS = [np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], np.uint8),
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.uint8),
        [np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], np.uint8), np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], np.uint8)],
        np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], np.uint8),
        np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]], np.uint8),
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)]

    SHAPES = np.stack([
        GROUPS[0],
        GROUPS[1],
        GROUPS[2][0], GROUPS[2][1], np.rot90(GROUPS[2][0]), np.rot90(GROUPS[2][1]),
        GROUPS[3], np.rot90(GROUPS[3]), np.rot90(np.rot90(GROUPS[3])), np.rot90(np.rot90(np.rot90(GROUPS[3]))),
        GROUPS[4], np.rot90(GROUPS[4]), np.rot90(np.rot90(GROUPS[4])), np.rot90(np.rot90(np.rot90(GROUPS[4]))),
        GROUPS[5]], axis=0)

    NSHAPE = SHAPES.shape[0]

    @staticmethod
    def add_effects(name, self, initr, rglzr, trainable):
        ci2, co2 = [int(np.ceil(i / o)) for i, o in ([self.ci, self.cgi], [self.co, self.cgo])]
        shape = (Conv2dLhcr.NSHAPE, ci2, co2)
        effects = self.add_weight(name, shape, tf.float32, initr, rglzr, trainable)
        return effects

    @staticmethod
    def calc_masks(effects, ci, co, cgi, cgo):
        effects_norm_ = Conv2dLhcr.step_func(effects)[:, :, None, :, None]  # (15, ci / cgi, 1, co / cgo, 1)
        effects_tile_ = tf.tile(effects_norm_, [1, 1, cgi, 1, cgo])
        a, b, c = effects.shape
        effects_tile = tf.reshape(effects_tile_, [a, b * cgi, c * cgo])
        effects_norm = effects_tile[:, :ci, :co]  # in case `ci % lgi != 0` or `co % lgo != 0`
        masks = tf.reduce_sum(Conv2dLhcr.SHAPES[..., None, None] * effects_norm[:, None, None], 0)
        return masks

    @staticmethod
    @tf.custom_gradient
    def step_func(xi):
        assert len(xi.shape) == 3
        xmax = tf.reduce_max(xi, 0, True)
        xo = tf.where(tf.less(xi, xmax), tf.zeros_like(xi), tf.ones_like(xi))

        def grad(dy): return tf.where(tf.less(tf.abs(xi - tf.reduce_mean(xi, 0, True)), 1), dy, 0.1 * dy)

        return xo, grad


class HetConv2d(KL.Layer):
    """HetConv."""

    def __init__(self, filters, kernel_size=(3, 3),
            strides=(1, 1), padding='valid', activation=None, use_bias=True,
            kernel_initializer='he_normal', bias_initializer='zeros',
            p=2, **kwargs):
        super(HetConv2d, self).__init__(**kwargs)
        self.p = p
        self.ci, self.co = -1, filters
        self.ksize = kernel_size
        self.stride = strides
        self.padd = padding.upper()
        self.act = activation
        self.use_bias = use_bias
        self.kinit, self.binit = kernel_initializer, bias_initializer
        self.kernels = None
        self.biases = None
        self.masks = None

    def build(self, ishape):
        self.ci = ishape[3]
        self.kernels = self.add_weight(name='kernels_pwc', shape=(*self.ksize, self.ci, self.co),
            initializer=self.kinit, regularizer=None, trainable=True)
        self.masks = self.gen_masks(self.ksize, self.ci, self.co, self.p)
        if self.use_bias:
            self.biases = self.add_weight(name='biases', shape=(self.co,),
                initializer=self.binit, regularizer=None, trainable=True)
        super(HetConv2d, self).build(ishape)

    def call(self, inputs, **kwargs):  # (?, hi, wi, ci)
        kernels_masked = self.kernels * self.masks
        output = tf.nn.conv2d(inputs, kernels_masked, [1, *self.stride, 1], self.padd)
        if self.use_bias:
            output += self.biases
        if self.act:
            output = KA.get(self.act)(output)
        return output

    def compute_output_shape(self, ishape):
        space_new = [KU.conv_utils.conv_output_length(_, self.ksize[0], self.padd.lower(), self.stride[0], 1)
            for _ in ishape[1:-1]]
        return tf.TensorShape([ishape[0]] + space_new + [self.co])

    @staticmethod
    def gen_masks(ksize, ci, co, p):
        assert tuple(ksize) == (3, 3)
        assert ci % p == 0 and co % p == 0
        s3x3 = ci // p
        s1x1 = ci - s3x3
        r1x1 = s1x1 // s3x3
        mask = list()
        for _ in range(s3x3):
            mask.append(Conv2dLhcr.SHAPES[0])
            mask.extend([Conv2dLhcr.SHAPES[13]] * r1x1)
        masks = list()
        for _ in range(co):
            masks.append(np.stack(mask, axis=-1))
            HetConv2d.cyclic_shift(mask)
        return np.stack(masks, axis=-1)

    @staticmethod
    def cyclic_shift(mask: list):
        mask.insert(0, mask[-1])
        mask.pop()
