##########################################################
# %%
# Library for tensorflow 2.2.0
##########################################################

import sys, os
import numpy as np

from matplotlib import pyplot as plt
import h5py

# tensorflow
import tensorflow as tf

# keras
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Activation, BatchNormalization, \
    Conv2DTranspose, LeakyReLU, concatenate
from tensorflow.keras.models import Model


##########################################################
# %%
# define common functions
##########################################################

def mosaic(img, num_row, num_col, fig_num, clim, title='', use_transpose=False, use_flipud=False):
    fig = plt.figure(fig_num)
    fig.patch.set_facecolor('black')

    if img.ndim < 3:
        img_res = img
        plt.imshow(img_res)
        plt.gray()
        plt.clim(clim)
    else:
        if img.shape[2] != (num_row * num_col):
            print('sizes do not match')
        else:
            if use_transpose:
                for slc in range(0, img.shape[2]):
                    img[:, :, slc] = np.transpose(img[:, :, slc])

            if use_flipud:
                img = np.flipud(img)

            img_res = np.zeros((img.shape[0] * num_row, img.shape[1] * num_col))
            idx = 0

            for r in range(0, num_row):
                for c in range(0, num_col):
                    img_res[r * img.shape[0]: (r + 1) * img.shape[0], c * img.shape[1]: (c + 1) * img.shape[1]] = img[:,
                                                                                                                  :,
                                                                                                                  idx]
                    idx = idx + 1
        plt.imshow(img_res)
        plt.gray()
        plt.clim(clim)

    plt.suptitle(title, color='white', fontsize=48)


def mvec(data):
    xl = data.size
    res = np.reshape(data, (xl))
    return res


def load_h5py(filename, rmod='r'):
    f = h5py.File(filename, rmod)
    arr = {}
    for k, v in f.items():
        arr[k] = np.transpose(np.array(v))
    return arr


def mfft(x, axis=0):
    y = np.fft.fftshift(np.fft.fft(np.fft.fftshift(x, axes=axis), axis=axis), axes=axis)
    return y


def mifft(x, axis=0):
    y = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)
    return y


def mfft2(x, axes=(0, 1)):
    y = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x, axes=axes), axes=axes), axes=axes)
    return y


def mifft2(x, axes=(0, 1)):
    y = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)
    return y


def msos(img, axis=3):
    return np.sqrt(np.sum(np.abs(img) ** 2, axis=axis))


##########################################################
# %%
# tensorflow functions
##########################################################

c2r = lambda x: tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1)
r2c = lambda x: tf.complex(x[..., 0], x[..., 1])


def nrmse(y_true, y_pred):
    return 100 * (K.sqrt(K.sum(K.square(y_pred - y_true)))) / (K.sqrt(K.sum(K.square(y_true))))

def nmae(y_true, y_pred):
    return 100 * (K.sum(K.abs(y_pred - y_true))) / (K.sum(K.abs(y_true)))

def custom_norm_loss(y_true, y_pred):
    return (nrmse(y_true, y_pred)+nmae(y_true, y_pred))/2.0



class tfft3(Layer):
    def __init__(self, **kwargs):
        super(tfft3, self).__init__(**kwargs)

    def build(self, input_shape):
        super(tfft3, self).build(input_shape)

    def call(self, x):
        xc = r2c(x[0])

        # fft3
        xt = tf.signal.ifftshift(xc, axes=(-3, -2, -1))
        kt = tf.signal.fft3d(xt)
        kt = tf.signal.fftshift(kt, axes=(-3, -2, -1))

        return c2r(kt)


class tifft3(Layer):
    def __init__(self, **kwargs):
        super(tifft3, self).__init__(**kwargs)

    def build(self, input_shape):
        super(tifft3, self).build(input_shape)

    def call(self, x):
        xc = r2c(x[0])

        # ifft3
        xt = tf.signal.ifftshift(xc, axes=(-3, -2, -1))
        kt = tf.signal.ifft3d(xt)
        kt = tf.signal.fftshift(kt, axes=(-3, -2, -1))

        return c2r(kt)


class tfft2(Layer):
    def __init__(self, **kwargs):
        super(tfft2, self).__init__(**kwargs)

    def build(self, input_shape):
        super(tfft2, self).build(input_shape)

    def call(self, x):
        xc = r2c(x[0])

        hx = int(int(xc.shape[1]) / 2)
        hy = int(int(xc.shape[2]) / 2)

        # fft2 over last two dimension
        xt = tf.roll(xc, shift=(hx, hy), axis=(-2, -1))
        kt = tf.signal.fft2d(xt)
        kt = tf.roll(kt, shift=(hx, hy), axis=(-2, -1))

        return c2r(kt)


# needed to be changed to multi-coil
class tifft2(Layer):
    def __init__(self, **kwargs):
        super(tifft2, self).__init__(**kwargs)

    def build(self, input_shape):
        super(tifft2, self).build(input_shape)

    def call(self, x):
        xc = r2c(x[0])

        hx = int(int(xc.shape[1]) / 2)
        hy = int(int(xc.shape[2]) / 2)

        # ifft2 over last two dimension
        it = tf.roll(xc, shift=(-hx, -hy), axis=(-2, -1))
        it = tf.signal.ifft2d(it)
        it = tf.roll(it, shift=(-hx, -hy), axis=(-2, -1))

        return c2r(it)


##########################################################
# %%
# U-net functions
##########################################################


# Conv2D -> Batch Norm -> Nonlinearity
def conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type='relu', USE_BN=True, layer_name='', strides = (1, 1), alpha = 0.3 ):
    with K.name_scope(layer_name):
        x = Conv2D(num_out_chan, kernel_size, activation=None, padding='same', kernel_initializer='truncated_normal', strides=strides)(x)
        if USE_BN:
            x = BatchNormalization()(x)
        if activation_type == 'LeakyReLU':
            return LeakyReLU(alpha=alpha)(x)
        else:
            return Activation(activation_type)(x)


# Conv2Dt -> Batch Norm -> Nonlinearity
def conv2Dt_bn_nonlinear(x, num_out_chan, kernel_size, activation_type='relu', USE_BN=True, layer_name=''):
    with K.name_scope(layer_name):
        x = Conv2DTranspose(num_out_chan, kernel_size, strides=(2, 2), padding='same',
                            kernel_initializer='truncated_normal')(x)
        if USE_BN:
            x = BatchNormalization()(x)

        if activation_type == 'LeakyReLU':
            return LeakyReLU()(x)
        else:
            return Activation(activation_type)(x)


# Conv2D -> Batch Norm -> softmax
def conv2D_bn_softmax(x, num_out_chan, kernel_size, USE_BN=True, layer_name=''):
    with K.name_scope(layer_name):
        x = Conv2D(num_out_chan, kernel_size, activation=None, padding='same', kernel_initializer='truncated_normal')(x)
        if USE_BN:
            x = BatchNormalization()(x)
        return Activation('softmax')(x)


def createOneLevel_UNet2D(x, num_out_chan, kernel_size, depth, num_chan_increase_rate, activation_type, USE_BN):
    if depth > 0:

        # Left
        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type, USE_BN=USE_BN)
        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type, USE_BN=USE_BN)

        x_to_lower_level = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last')(
            x)

        # Lower level
        x_from_lower_level = createOneLevel_UNet2D(x_to_lower_level, int(num_chan_increase_rate * num_out_chan),
                                                   kernel_size, depth - 1, num_chan_increase_rate, activation_type,
                                                   USE_BN)
        x_conv2Dt = conv2Dt_bn_nonlinear(x_from_lower_level, num_out_chan, kernel_size, activation_type=activation_type,
                                         USE_BN=USE_BN)

        # Right
        x = concatenate([x, x_conv2Dt], axis=3)
        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type, USE_BN=USE_BN)
        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type, USE_BN=USE_BN)

    else:
        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type, USE_BN=USE_BN)
        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type, USE_BN=USE_BN)

    return x


def UNet2D_softmax(nx, ny, ns, nc_input=2, kernel_size=(3, 3), num_out_chan_highest_level=64, depth=5,
                   num_chan_increase_rate=2, activation_type='LeakyReLU', USE_BN=True):
    # define the inputs
    input_x = Input(shape=(nx, ny, nc_input), dtype=tf.float32)

    x = conv2D_bn_nonlinear(input_x, num_out_chan_highest_level, kernel_size, activation_type=activation_type,
                            USE_BN=USE_BN)

    temp = createOneLevel_UNet2D(x, num_out_chan_highest_level, kernel_size, depth - 1, num_chan_increase_rate,
                                 activation_type, USE_BN)

    # output_img = conv2D_bn_nonlinear(temp, num_output_chans, kernel_size, activation_type=None, USE_BN=False)
    output_img = conv2D_bn_softmax(temp, ns, kernel_size, USE_BN=False)

    return Model(inputs=input_x, outputs=output_img)


# from keras import Callback
# from keras import backend as K
# class AdamLearningRateTracker(Callback):
#     def on_epoch_end(self, logs={}):
#         beta_1=0.9, beta_2=0.999
#         optimizer = self.model.optimizer
#         if optimizer.decay>0:
#             lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
#         t = K.cast(optimizer.iterations, K.floatx()) + 1
#         lr_t = lr * (K.sqrt(1. - K.pow(beta_2, t)) /(1. - K.pow(beta_1, t)))
#         print('\nLR: {:.6f}\n'.format(lr_t))