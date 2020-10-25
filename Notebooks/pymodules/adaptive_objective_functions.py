from tensorflow.python.keras import backend as K
import tensorflow as tf

from heterogeneous_mask_iterator import DELETED_MASK_IDENTIFIER

smooth = 1.


def adaptive_dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    valid_classes_mask = tf.not_equal(y_true_f, DELETED_MASK_IDENTIFIER)
    y_true_masked = tf.boolean_mask(y_true_f, valid_classes_mask)
    y_pred_masked = tf.boolean_mask(y_pred_f, valid_classes_mask)
    intersection = K.sum(y_true_masked * y_pred_masked)
    coef = (2. * intersection + smooth) / (K.sum(y_true_masked) + K.sum(y_pred_masked) + smooth)
    return 1 - coef


def adaptive_ca_loss(y_true, y_pred):
    pass
