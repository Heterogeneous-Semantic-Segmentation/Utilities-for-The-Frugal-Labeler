from keras import Model
from keras import layers
from tensorflow.python.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from data import train_generator
import segmentation_models as sm
import numpy as np

from heterogeneous_mask_iterator import DELETED_MASK_IDENTIFIER

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# prep
batch_size = 11
epochs = 400
iterations_per_epoch = 300

data_gen_args = dict(rotation_range=0.3,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     fill_mode='nearest')

train_generator = train_generator(batch_size=batch_size,
                                  train_path='../../data/train_images',
                                  image_folder='ventral_samples',
                                  mask_folders=['ventral_mask_atrium', 'ventral_mask_bulbus', 'ventral_mask_heart'],
                                  heterogeneously_labeled_masks=['ventral_mask_atrium', 'ventral_mask_bulbus',
                                                                 'ventral_mask_heart'],
                                  missing_labels_ratio=0.5,
                                  aug_dict=data_gen_args,
                                  image_color_mode='rgb',
                                  target_size=(256, 256))

# prep end

import time


def adaptive_ca_loss(y_true, y_pred):
    # find maximum value along last axis -> we get ones where a true mask exists.
    # if no true mask exists the only values are 0 and -1
    # after that take the maximum to convert all -1's to 0s -> we get a map for every class where a true mask exists
    available_true_values = tf.math.maximum(tf.reduce_max(y_true, axis=[3]), 0)
    # repeat the mask availability values three times to regain right shape (to be compatible for calculations with y_pred)
    available_true_values = tf.repeat(tf.reshape(available_true_values, y_true.get_shape()[:3] + [1]), 3, axis=3)
    # stack the true calculated aviable true values and y_true flattened on top of each other
    stacked = tf.stack([K.flatten(available_true_values), K.flatten(y_true)])
    # y_true possible values : [-1, 0, 1] ; available_true_values possible values : [0, 1]
    # if we take the product along axis 0 only the combination [1,-1] will yield '-1'
    # so we know that if the product is -1 we have the searched indices:
    # - The true value is given (available_true_values = 1) and
    # - The value does not have a ground truth label (y_true = -1)
    indices = tf.where(tf.math.equal(tf.math.reduce_prod(stacked, axis=0),-1))
    print()
    pass


a = None
for a in train_generator:
    # samples = np.add(a[1],np.random.normal(0,.1,a[1].shape))
    F = adaptive_ca_loss(K.variable(a[1]), K.variable(a[1]))
    print(K.eval(F))
    # print('dice loss: %f'%(time.time()-start))
    start = time.time()
    # F = dice_loss(K.variable(samples), K.variable(samples))
    # print(K.eval(F))
    # print('dice coeff: %f'%(time.time()-start))
