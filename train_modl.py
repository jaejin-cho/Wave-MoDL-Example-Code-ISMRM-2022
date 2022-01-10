##########################################################
# %%
# tensorflow version 2.2.0
##########################################################
import os,sys
os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"

# ##########################################################
# %%
# Add/Set paths
##########################################################

path0 = os.path.join(os.getcwd(),'utils')
sys.path.append(path0)

data_train_path     = os.path.join(os.getcwd(),'database/train/')
data_valid_path     = os.path.join(os.getcwd(),'database/valid/')
data_test_path      = os.path.join(os.getcwd(),'database/test/')
result_path         = os.path.join(os.getcwd(),'network/')

##########################################################
# %%
# import some librarie
##########################################################

import  numpy               as  np
import  tensorflow          as  tf

import  library_utils       as  mf
import  matplotlib.pyplot   as  plt

from    tensorflow.keras.optimizers    import  Adam
from    tensorflow.keras.callbacks     import  ModelCheckpoint

# keras
import tensorflow.keras
from tensorflow.keras import backend as K

from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Conv3D, MaxPooling3D, Activation, BatchNormalization, \
    Add, Conv3DTranspose, LeakyReLU, Lambda, concatenate, Average
from tensorflow.keras.models import Model


##########################################################
# %%
# parameters
##########################################################

ry          = 4
rz          = 3

# wave parameter
Tadc        = 5000*1e-6         # sec
FOVy        = 240*1e-3          # m
FOVz        = 192*1e-3          # m

# network size
num_block   = 10
nLayers     = 5
num_filters = 24

# training
N_epochs    = 20
S_batch     = 1
model_name  = os.path.join(result_path,'modl_qalas_w_kykz.hdf5')
hist_name   = os.path.join(result_path,'modl_qalas_w_kykz.npz')


num_slc, nx, ny, nc = 192, 480, 238, 32
# num_slc, nx, ny, nc, ns   = 192, 256, 224, 12, 4

w_contrast  = [3.26, 2.36, 1.57 , 1.12, 1]

##########################################################
# %%
# data filenames
##########################################################

files_train         = os.listdir(data_train_path)
files_train.sort()
files_train_dat     = [];

for fl in range(len(files_train)):
    if files_train[fl].endswith('_img.mat'):
        files_train_dat.append(files_train[fl][:-8])

num_train_subj = len(files_train_dat)


files_valid         = os.listdir(data_valid_path)
files_valid.sort()
files_valid_dat     = [];

for fl in range(len(files_valid)):
    if files_valid[fl].endswith('_img.mat'):
        files_valid_dat.append(files_valid[fl][:-8])

num_valid_subj = len(files_valid_dat)


##########################################################
# %%
# function
##########################################################

def np_c2r(x):
    return np.stack((np.real(x),np.imag(x)),axis=-1)


def np_r2c(x):
    return x[...,0]+1j*x[...,1]


def np_c2r5(x):
    res = np.stack(  ( np.real(x[...,0]),np.imag(x[...,0]), \
                       np.real(x[...,1]),np.imag(x[...,1]), \
                       np.real(x[...,2]),np.imag(x[...,2]), \
                       np.real(x[...,3]),np.imag(x[...,3]), \
                       np.real(x[...,4]),np.imag(x[...,4])  ),   axis = -1  )
    return res

def np_r2c5(x):
    res = np.stack( (  x[...,0]+1j*x[...,1],\
                       x[...,2]+1j*x[...,3],\
                       x[...,4]+1j*x[...,5],\
                       x[...,6]+1j*x[...,7],\
                       x[...,8]+1j*x[...,9] ),    axis = -1  )
    return res


c2r=Lambda(lambda x:tf.stack([tf.math.real(x),tf.math.imag(x)],axis=-1))
r2c=Lambda(lambda x:tf.complex(x[...,0],x[...,1]))


c2r5=Lambda(lambda x:tf.stack([tf.math.real(x[...,0]),tf.math.imag(x[...,0]),\
                               tf.math.real(x[...,1]),tf.math.imag(x[...,1]),\
                               tf.math.real(x[...,2]),tf.math.imag(x[...,2]),\
                               tf.math.real(x[...,3]),tf.math.imag(x[...,3]),\
                               tf.math.real(x[...,4]),tf.math.imag(x[...,4])],axis=-1))

r2c5=Lambda(lambda x:tf.stack([tf.complex(x[...,0],x[...,1]),\
                               tf.complex(x[...,2],x[...,3]),\
                               tf.complex(x[...,4],x[...,5]),\
                               tf.complex(x[...,6],x[...,7]),\
                               tf.complex(x[...,8],x[...,9])], axis=-1))


class tfft3(Layer):
    def __init__(self, **kwargs):
        super(tfft3, self).__init__(**kwargs)

    def build(self, input_shape):
        super(tfft3, self).build(input_shape)

    def call(self, x):
        xc = r2c5(x[0])

        # fft3
        t0 = tf.transpose(xc,[0,4,1,2,3])
        t0 = tf.signal.ifftshift(t0, axes=(2,3,4))
        t1 = tf.signal.fft3d(t0)
        t1 = tf.signal.fftshift(t1, axes=(2,3,4))
        t2 = tf.transpose(t1,[0,2,3,4,1])

        return c2r5(t2)


class tifft3(Layer):
    def __init__(self, **kwargs):
        super(tifft3, self).__init__(**kwargs)

    def build(self, input_shape):
        super(tifft3, self).build(input_shape)

    def call(self, x):
        xc = r2c5(x[0])

        # ifft3
        t0 = tf.transpose(xc,[0,4,1,2,3])
        t0 = tf.signal.ifftshift(t0, axes=(2,3,4))
        t1 = tf.signal.ifft3d(t0)
        t1 = tf.signal.fftshift(t1, axes=(2,3,4))
        t2 = tf.transpose(t1,[0,2,3,4,1])

        return c2r5(t2)


##########################################################
# %%
# tensorflow functions
##########################################################

class Aclass:
    def __init__(self, csm,mask,lam):
        with tf.name_scope('Ainit'):
            self.mask           =   mask
            self.csm            =   csm
            self.lam            =   lam

    def myAtA(self,img):
        with tf.name_scope('AtA'):
            coilImages  =   tf.expand_dims(self.csm,axis=-1)*tf.expand_dims(img,axis=0)
            kspace      =   tf.transpose(tf.signal.fft3d(tf.transpose(coilImages,[0,4,1,2,3])),[0,2,3,4,1])
            temp        =   kspace*self.mask
            coilImgs    =   tf.transpose(tf.signal.ifft3d(tf.transpose(temp,[0,4,1,2,3])),[0,2,3,4,1])
            coilComb    =   tf.reduce_sum(coilImgs*tf.math.conj(tf.expand_dims(self.csm,axis=-1)),axis=0)
            coilComb    =   coilComb+self.lam*img
        return coilComb


def myCG(A,rhs):

    rhs=r2c5(rhs)
    cond=lambda i,rTr,*_: tf.logical_and( tf.less(i,10), rTr>1e-5)
    def body(i,rTr,x,r,p):
        with tf.name_scope('cgBody'):
            Ap      =   A.myAtA(p)
            alpha   =   rTr / tf.cast(tf.reduce_sum(tf.math.conj(p)*Ap),dtype=tf.float32)
            alpha   =   tf.complex(alpha,0.)
            x       =   x + alpha * p
            r       =   r - alpha * Ap
            rTrNew  =   tf.cast( tf.reduce_sum(tf.math.conj(r)*r),dtype=tf.float32)
            beta    =   rTrNew / rTr
            beta    =   tf.complex(beta,0.)
            p       =   r + beta * p
        return i+1,rTrNew,x,r,p

    x       =   tf.zeros_like(rhs)
    i,r,p   =   0,rhs,rhs
    rTr     =   tf.cast( tf.reduce_sum(tf.math.conj(r)*r),dtype=tf.float32)
    loopVar =   i,rTr,x,r,p
    out     =   tf.while_loop(cond,body,loopVar,name='CGwhile',parallel_iterations=1)[2]
    return c2r5(out)


class myDC(Layer):

    def __init__(self, **kwargs):
        super(myDC, self).__init__(**kwargs)

        self.lam1 = self.add_weight(name='lam1', shape=(1,), initializer=tf.constant_initializer(value=0.015),
                                     dtype='float32', trainable=True)
        self.lam2 = self.add_weight(name='lam2', shape=(1,), initializer=tf.constant_initializer(value=0.015),
                                     dtype='float32', trainable=True)

    def build(self, input_shape):
        super(myDC, self).build(input_shape)

    def call(self, x):
        rhs, csm, mask = x
        lam3 = tf.complex(self.lam1 + self.lam2, 0.)

        def fn(tmp):
            c, m, r = tmp
            Aobj = Aclass(c, m, lam3)
            y = myCG(Aobj, r)
            return y

        inp = (csm, mask, rhs)
        # Mapping functions with multi-arity inputs and outputs
        rec = tf.map_fn(fn, inp, dtype=tf.float32, name='mapFn')
        return rec

    def lam_weight1(self, x):
        img = x[0]
        res = (self.lam1+self.lam2) * img
        return res

    def lam_weight2(self, x):
        in0, in1 = x
        res = self.lam1 * in0 + self.lam2 * in1
        return res


class rm_bg(Layer):
    def __init__(self, **kwargs):
        super(rm_bg, self).__init__(**kwargs)

    def build(self, input_shape):
        super(rm_bg, self).build(input_shape)

    def call(self, x):
        img, csm    = x
        rcsm        = tf.expand_dims(tf.reduce_sum(tf.math.abs(csm),axis=1), axis=-1)
        cmask       = tf.math.greater(rcsm,tf.constant(0,dtype=tf.float32))
        rec         = tf.cast(cmask,dtype=tf.float32) * img
        return rec


class rm_bmask(Layer):
    def __init__(self, **kwargs):
        super(rm_bmask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(rm_bmask, self).build(input_shape)

    def call(self, x):
        img, bmask  = x
        rcsm        = tf.expand_dims(bmask, axis=-1)
        cmask       = tf.math.greater(rcsm,tf.constant(0,dtype=tf.float32))
        rec         = tf.cast(cmask,dtype=tf.float32) * img
        return rec

##########################################################
# %%
# network
##########################################################


# Conv3D -> Batch Norm -> Nonlinearity
def conv3D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type='relu', USE_BN=True, layer_name=''):
    with K.name_scope(layer_name):
        x = Conv3D(num_out_chan, kernel_size, activation=None, padding='same', kernel_initializer='truncated_normal')(x)
        if USE_BN:
            x = BatchNormalization()(x)
        if activation_type == 'LeakyReLU':
            return LeakyReLU()(x)
        else:
            return Activation(activation_type)(x)

def RegConvLayers(nx,ny,nz,ne,nLayers,num_filters):

    input_x     = Input(shape=(nx,ny,nz,2*ne), dtype = tf.float32)
    filter_size = (3,3,2)
    bool_USE_BN = True
    AT          = 'LeakyReLU'

    rg_term     = input_x
    for lyr in range(0,nLayers):
        rg_term = conv3D_bn_nonlinear(rg_term, num_filters, filter_size, activation_type=AT, USE_BN=bool_USE_BN, layer_name='')

    # go to image space
    rg_term = conv3D_bn_nonlinear(rg_term, 2*ne, (1,1,1), activation_type='tanh', USE_BN=False, layer_name='')

    # skip connection
    rg_term = Add()([rg_term,input_x])

    return Model(inputs     =   input_x, outputs    =   rg_term)


def create_modl(nx, ny, rz, nc, ne, nLayers, num_block, num_filters = 64):

    # define the inputs
    input_c     = Input(shape=(nc,nx,ny,rz), dtype = tf.complex64,      name = 'input_c')
    input_m     = Input(shape=(nx,ny,rz,ne), dtype = tf.complex64,      name = 'input_m')
    input_Atb   = Input(shape=(nx,ny,rz,ne), dtype = tf.complex64,      name = 'input_a')

    dc_term     = c2r5(input_Atb)

    RegConv_k   = RegConvLayers(nx,ny,rz,ne,nLayers,num_filters)
    RegConv_i   = RegConvLayers(nx,ny,rz,ne,nLayers,num_filters)
    UpdateDC    = myDC()
    rmbg        = rm_bg()

    myFFT       = tfft3()
    myIFFT      = tifft3()

    for blk in range(0,num_block):
        # CNN Regularization
        rg_term_i   = RegConv_i(dc_term)
        rg_term_k   = myIFFT([RegConv_k(myFFT([dc_term]))])
        rg_term     = UpdateDC.lam_weight2([rg_term_i,rg_term_k])
        # AtA update
        rg_term     = Add()([c2r5(input_Atb), rg_term])

        # Update DC
        dc_term     = UpdateDC([rg_term,input_c,input_m])

    out_x = rmbg([dc_term,input_c])

    return Model(inputs     =   [ input_c, input_m, input_Atb],
                 outputs    =   out_x )


##########################################################
# %%
# define modl
##########################################################

model = create_modl(    nx  = nx,
                        ny  = ny,
                        rz  = rz,
                        nc  = nc,
                        ne  = 5,
                        num_block       =   num_block,
                        nLayers         =   nLayers,
                        num_filters     =   num_filters)

# Define an optimizer
adam_opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)

# Compile the model
model.compile(optimizer=adam_opt,loss='mse')

try:
    model.load_weights(model_name)
    print('loading the model...')
except:
    print('without loading...')

##########################################################
# %%
# one epoch
##########################################################

loss_train      = [np.inf]
loss_valid      = [np.inf]
loss_valid_min  = np.inf

for epoch in range(N_epochs):

    loss_train_epoch  = 0
    loss_valid_epoch  = 0

    # training
    data_path   =   data_train_path
    filenames   =   files_train_dat
    np.random.shuffle(filenames)

    for subj in range(num_train_subj):

        # load data
        filename_img = data_path + filenames[subj] + '_img.mat'
        filename_csm = data_path + filenames[subj] + '_csm.mat'

        dt = mf.load_h5py(filename_img)
        subj_img = dt['input_img']
        dt = mf.load_h5py(filename_csm)
        subj_csm = dt['sens']
        del dt

        img     =  subj_img['real'] + 1j*subj_img['imag']
        csm     =  subj_csm['real'] + 1j*subj_csm['imag']
        del subj_img, subj_csm

        # calc parameters
        nx,ny,nz,nc,ne  =   img.shape
        z_skip          =   np.int(nz/rz)

        # suffle
        z_indexes       =   np.arange(z_skip)
        np.random.shuffle(z_indexes)

        for batch in range(z_skip):

            slc         =   z_indexes[batch]

            batch_img   =   img[:,:,slc::z_skip,]

            # hard coding
            batch_img[...,0] = batch_img[...,0] * w_contrast[0]
            batch_img[...,1] = batch_img[...,1] * w_contrast[1]
            batch_img[...,2] = batch_img[...,2] * w_contrast[2]
            batch_img[...,3] = batch_img[...,3] * w_contrast[3]
            batch_img[...,4] = batch_img[...,4] * w_contrast[4]

            in_c        =   csm[:,:,slc::z_skip,]
            out_y       =   np.sum(np.multiply(np.expand_dims(np.conj(in_c),axis=-1),batch_img),3).astype(np.complex64)
            in_m        =   np.zeros((nx,ny,rz,1,ne), dtype=np.complex64)

            # hard coding
            in_m[:,0::ry,0,:,0] = 1
            in_m[:,1::ry,1,:,1] = 1
            in_m[:,2::ry,2,:,2] = 1
            in_m[:,3::ry,0,:,3] = 1
            in_m[:,0::ry,1,:,4] = 1

            kdata       =   np.fft.fftn(batch_img, axes = (0,1,2) )
            kdata_sub   =   np.multiply(kdata, in_m). astype(np.complex64)
            in_x        =   np.fft.ifftn(kdata_sub,   axes = (0,1,2) ).astype(np.complex64)

            del kdata, kdata_sub, batch_img

            in_a        =   np.sum(np.multiply(np.expand_dims(np.conj(in_c),axis=-1),in_x),3).astype(np.complex64)

            # change dimension
            in_x        =   np_c2r5(np.expand_dims(in_x,axis=0))
            in_c        =   np.transpose(np.expand_dims(in_c,axis=0), axes=(0,4,1,2,3))
            in_m        =   np.expand_dims(in_m[...,0,:],axis=0)
            in_a        =   np.expand_dims(in_a,axis=0)
            out_y       =   np_c2r5(np.expand_dims(out_y,axis=0))

            loss_t      =   model.train_on_batch([in_c,in_m,in_a],out_y)
            loss_train_epoch += (loss_t/z_skip/num_train_subj)


    # validation
    data_path   =   data_valid_path
    filenames   =   files_valid_dat

    for subj in range(num_valid_subj):

        # load data
        filename_img = data_path + filenames[subj] + '_img.mat'
        filename_csm = data_path + filenames[subj] + '_csm.mat'

        dt = mf.load_h5py(filename_img)
        subj_img = dt['input_img']
        dt = mf.load_h5py(filename_csm)
        subj_csm = dt['sens']
        del dt

        img     =  subj_img['real'] + 1j*subj_img['imag']
        csm     =  subj_csm['real'] + 1j*subj_csm['imag']
        del subj_img, subj_csm

        # calc parameters
        nx,ny,nz,nc,ne  =   img.shape
        z_skip          =   np.int(nz/rz)

        # suffle
        z_indexes       =   np.arange(z_skip)
        np.random.shuffle(z_indexes)

        for batch in range(z_skip):

            slc         =   z_indexes[batch]

            batch_img   =   img[:,:,slc::z_skip,]

            # hard coding
            batch_img[...,0] = batch_img[...,0] * w_contrast[0]
            batch_img[...,1] = batch_img[...,1] * w_contrast[1]
            batch_img[...,2] = batch_img[...,2] * w_contrast[2]
            batch_img[...,3] = batch_img[...,3] * w_contrast[3]
            batch_img[...,4] = batch_img[...,4] * w_contrast[4]

            in_c        =   csm[:,:,slc::z_skip,]
            out_y       =   np.sum(np.multiply(np.expand_dims(np.conj(in_c),axis=-1),batch_img),3).astype(np.complex64)
            in_m        =   np.zeros((nx,ny,rz,1,ne), dtype=np.complex64)

            # hard coding
            in_m[:,0::ry,0,:,0] = 1
            in_m[:,1::ry,1,:,1] = 1
            in_m[:,2::ry,2,:,2] = 1
            in_m[:,3::ry,0,:,3] = 1
            in_m[:,0::ry,1,:,4] = 1

            kdata       =   np.fft.fftn(batch_img, axes = (0,1,2) )
            kdata_sub   =   np.multiply(kdata, in_m). astype(np.complex64)
            in_x        =   np.fft.ifftn(kdata_sub,   axes = (0,1,2) ).astype(np.complex64)

            del kdata, kdata_sub, batch_img

            in_a        =   np.sum(np.multiply(np.expand_dims(np.conj(in_c),axis=-1),in_x),3).astype(np.complex64)

            # change dimension
            in_x        =   np_c2r5(np.expand_dims(in_x,axis=0))
            in_c        =   np.transpose(np.expand_dims(in_c,axis=0), axes=(0,4,1,2,3))
            in_m        =   np.expand_dims(in_m[...,0,:],axis=0)
            in_a        =   np.expand_dims(in_a,axis=0)
            out_y       =   np_c2r5(np.expand_dims(out_y,axis=0))

            loss_t      =   model.evaluate([in_c,in_m,in_a],out_y)
            loss_valid_epoch += (loss_t/z_skip/num_valid_subj)

    # save model
    if loss_valid_min > loss_valid_epoch:
        model.save_weights(model_name)
        loss_valid_min = loss_valid_epoch

    # save loss values
    loss_train.append(loss_train_epoch)
    loss_valid.append(loss_train_epoch)
    np.savez(hist_name, loss_train=loss_train, loss_valid = loss_valid)
