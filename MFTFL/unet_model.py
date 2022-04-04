from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K
import tensorflow as tf
# in jupyter run:
# %env SM_FRAMEWORK=tf.keras
# before calling segmentation_models
import segmentation_models as sm
import numpy as np

"""
    The UNet Model used in this work.
"""


def segmentation_sparse_iou(y_true, y_pred):
    # intersection and union shapes are batch_size * n_classes (values = area in pixels)
    axes = (1, 2)  # W,H axes of each image

    y_true_orig = y_true
    y_true = tf.where(tf.equal(y_true_orig,-1.),1.,y_true)
    y_pred = tf.where(tf.equal(y_true_orig, -1.), 1., y_pred)

    intersection = tf.math.reduce_sum(tf.abs(y_pred * y_true), axis=axes)
    mask_sum = tf.math.reduce_sum(tf.abs(y_true), axis=axes) + tf.math.reduce_sum(tf.abs(y_pred), axis=axes)
    union = mask_sum -intersection

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)

    # define mask to be 0 when no pixels are present in either y_true or y_pred, 1 otherwise
    mask = tf.cast(tf.not_equal(union,0),tf.float32)

    # Drop background class
    iou = iou[:, :-1]
    mask = mask[:,:-1]

    # mean only over non-absent classes
    class_count = tf.math.reduce_sum(mask,axis=0)
    return tf.math.reduce_mean(tf.math.reduce_sum(iou * mask, axis=0)[class_count != 0] / (class_count[class_count != 0]))

def unet(loss,pretrained_weights = None,input_size = (256,256,1),output_filters=1):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Change back None to 'relu'
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    # if we have a single class (output_filters=1) we want to output the value of a sigmoid. If we have more than one
    # class we need a softmax.
    if output_filters == 1:
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        output = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    else:
        output = Conv2D(filters=output_filters, kernel_size=(1, 1))(conv9)
        output = BatchNormalization()(output)
        output = Activation('softmax')(output)

    model = Model(inputs = inputs, outputs = output)
    model.compile(optimizer = Adam(), loss = loss ,  metrics = ['accuracy',segmentation_sparse_iou])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model