from tensorflow.python.keras import backend as K
import tensorflow as tf


smooth = 1.

def adaptive_dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    valid_classes_mask = tf.not_equal(y_true_f, -1)
    y_true_masked = tf.boolean_mask(y_true_f, valid_classes_mask)
    y_pred_masked = tf.boolean_mask(y_pred_f, valid_classes_mask)
    intersection = K.sum(y_true_masked * y_pred_masked)
    coef = (2. * intersection + smooth) / (K.sum(y_true_masked) + K.sum(y_pred_masked) + smooth)
    return 1 - coef

def ca_loss(y_true, y_pred):
    # find maximum value along last axis -> we get ones where a true mask exists.
    # if no true mask exists the only values are 0 and -1
    # after that take the maximum to convert all -1's to 0's -> we get a map for every class where a true mask exists
    available_true_values = tf.math.maximum(tf.reduce_max(y_true, axis=[3]), 0)
    # repeat the mask availability values three times to regain right shape (to be compatible for calculations with y_pred)
    available_true_values = tf.repeat(tf.reshape(available_true_values, y_true.get_shape()[:3] + [1]), 3, axis=3)
    # stack the true calculated available true values and y_true flattened on top of each other
    stacked = tf.stack([K.flatten(available_true_values), K.flatten(y_true)])
    # y_true possible values : [-1, 0, 1] ; available_true_values possible values : [0, 1]
    # if we take the product along axis 0 only the combination [1,-1] will yield '-1'
    # so we know that if the product is -1 we have the searched indices:
    #           - The true value is given (available_true_values = 1) and
    #           - The value does not have a ground truth label (y_true = -1)
    indices = tf.where(tf.math.equal(tf.math.reduce_prod(stacked, axis=0),-1))
    # collect all values of the candidate indices in y_pred
    candidates_y_pred = tf.gather(K.flatten(y_pred),indices)
    # return the average value for all values in the batch
    # the exponential function minus 1 is applied to get values in the range [0,(e-1)] where e is the eulers number ~1.7
    # returns 0 if there are no candidates (so if all classes are given in the batch
    return tf.math.divide_no_nan(K.sum((tf.exp(candidates_y_pred)-1)), len(candidates_y_pred))
