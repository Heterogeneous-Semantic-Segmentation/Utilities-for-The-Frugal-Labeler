from keras import Model
from keras import layers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
import tensorflow as tf
import os
from data import trainGenerator

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def dice_loss_agnostic(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    coef = (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)
    return coef

#prep
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

train_generator = trainGenerator(batch_size,'../../data/train_images','ventral_samples','ventral_mask_combined',data_gen_args,image_color_mode='rgb',color_map=[[33,33,33],[20,20,20],[19,19,19],[0,0,0]],target_size = (256,256),flag_multi_class=True)

for a in train_generator:
    dice_loss_agnostic(a[1], a[1])
    break




#prep end


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    coef = (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)
    return coef

def masked_loss_function(y_true, y_pred, mask_value=0):
    '''
    This model has two target values which are independent of each other.
    We mask the output so that only the value that is used for training
    contributes to the loss.
        mask_value : is the value that is not used for training
    '''
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    return losses.mean_squared_error(y_true * mask, y_pred * mask)



x = [1, 0]
y = [1, 1]
F = masked_loss_function(K.variable(x), K.variable(y), K.variable(0))
assert K.eval(F) == 0