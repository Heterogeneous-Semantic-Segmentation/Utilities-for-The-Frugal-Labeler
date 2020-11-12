from tensorflow.python.keras import backend as K
import tensorflow as tf

# both loss functions expect unlabeled masks to be labeled as '-1' in all points of the mask.

smooth = 1e-5


def dice_loss_single_channel(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # get boolean mask for valid classes ( all points which are not labeled as -1)
    valid_classes_mask = tf.not_equal(y_true_f, -1)
    # mask both y_true and y_pred to only contain labeled values
    y_true_masked = tf.boolean_mask(y_true_f, valid_classes_mask)
    y_pred_masked = tf.boolean_mask(y_pred_f, valid_classes_mask)
    # calculate the dice loss only with the labeled values
    # normalization is included 'implicitly' since we masked out the unlabeled values.
    intersection = K.sum(y_true_masked * y_pred_masked)
    if intersection == 0:
        return -1.
    else:
        coef = (2. * intersection + smooth) / (K.sum(y_true_masked) + K.sum(y_pred_masked) + smooth)
        return 1 - coef


def dice_loss_single_sample(y_true, y_pred):
    y_true_swapped = tf.einsum("i...j->j...i", y_true)
    y_pred_swapped = tf.einsum("i...j->j...i", y_pred)
    elems = (y_true_swapped, y_pred_swapped)
    preds = tf.map_fn(lambda x: dice_loss_single_channel(x[0], x[1]), elems, dtype=tf.float32)
    preds_filtered_invalid_values = preds[preds >= 0]
    if tf.size(preds_filtered_invalid_values) == 0:
        # if there are no true values return -1 which is a invalid value (later filtered out). possible value
        # lazy solution, a better solution might be to remember the last valid value to not distort the
        # learning process.
        return -1.
    return tf.math.reduce_mean(preds_filtered_invalid_values)


def adaptive_dice_loss(y_true, y_pred):
    elems = (y_true, y_pred)
    preds = tf.map_fn(lambda x: dice_loss_single_sample(x[0], x[1]), elems, dtype=tf.float32)
    preds_filtered_invalid_values = preds[preds >= 0]
    if tf.size(preds_filtered_invalid_values) == 0:
        # If we do not have any true values, assume worst possible value (very lazy solution).
        return tf.constant(1.)
    return tf.math.reduce_mean(preds_filtered_invalid_values)


def ca_loss(y_true, y_pred, background_class_index):
    # remove background class
    y_true = tf.concat([y_true[:, :, :, :background_class_index], y_true[:, :, :, background_class_index + 1:]],
                       axis=-1)
    y_pred = tf.concat([y_pred[:, :, :, :background_class_index], y_pred[:, :, :, background_class_index + 1:]],
                       axis=-1)
    # find maximum value along last axis -> we get ones where a true mask exists.
    # if no true mask exists the only values are 0 and -1
    # after that take the maximum to convert all -1's to 0's -> we get a map for every sample where a true mask exists
    available_true_values = tf.math.maximum(tf.reduce_max(y_true, axis=[3]), 0)
    # repeat the mask availability values three times to regain right shape (to be compatible for calculations with y_pred)
    available_true_values = tf.repeat(tf.reshape(available_true_values, tf.concat([tf.shape(y_true)[:3], [1]], axis=0)),
                                      3, axis=3)
    # stack the true calculated available true values and y_true flattened on top of each other
    stacked = tf.stack([K.flatten(available_true_values), K.flatten(y_true)])
    # y_true possible values : [-1, 0, 1] ; available_true_values possible values : [0, 1]
    # if we take the product along axis 0 only the combination [1,-1] will yield '-1'
    # so we know that if the product is -1 we have the searched indices:
    #           - The true value is given (available_true_values = 1) and
    #           - The value does not have a ground truth label (y_true = -1)
    indices = tf.where(tf.math.equal(tf.math.reduce_prod(stacked, axis=0), -1))
    # collect all values of the candidate indices in y_pred
    candidates_y_pred = tf.gather(K.flatten(y_pred), indices)
    # return the average value for all values in the batch
    return tf.math.divide_no_nan(tf.reduce_sum(candidates_y_pred),
                                 tf.cast(tf.size(candidates_y_pred), dtype=tf.float32))


def combined_loss(background_class_index, alpha):
    def loss(y_true, y_pred):
        ca = ca_loss(y_true, y_pred, background_class_index)
        dice = adaptive_dice_loss(y_true, y_pred)
        if ca == 0:
            return dice
        else:
            return tf.multiply(1 - alpha, dice) + tf.multiply(alpha, ca)

    return loss
