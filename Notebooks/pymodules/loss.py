from keras import Model
from keras import layers
from tensorflow.python.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from unet_model import unet
from data import train_generator
from adaptive_objective_functions import adaptive_dice_loss,ca_loss,combined_loss
import segmentation_models as sm
import numpy as np
from create_one_hot_encoded_map_from_mask import get_one_hot_map

from heterogeneous_mask_iterator import DELETED_MASK_IDENTIFIER

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import os
from PIL import Image

# color map is different on test data
col_map = [[255,255,255],[20,20,20],[19,19,19],[0,0,0]]



X_test = []
for filepath in os.listdir('../../data/test_images/ventral_samples_R0004'):
    image = Image.open('../../data/test_images/ventral_samples_R0004/'+filepath)
    image = image.resize((256, 256))
    # convert image to numpy array
    data = np.asarray(image)
    data = data/255.
    X_test.append(data)
X_test = np.array(X_test)
Y_test = []
for filepath in os.listdir('../../data/test_images/ventral_mask_combined_R0004'):
    image = Image.open('../../data/test_images/ventral_mask_combined_R0004/'+filepath)
    image = image.resize((256, 256))
    Y_test.append(get_one_hot_map(np.asarray(image),col_map))
Y_test = tf.stack(Y_test)



model = unet(combined_loss,input_size = (256,256,3),output_filters=4)

import time

pred = K.variable(model.predict(X_test[0].reshape(1,256,256,3)))
true = K.variable(Y_test[0]).numpy().reshape(1,256,256,4)
true = tf.repeat(true,2,axis=0)
pred = tf.repeat(pred,2,axis=0)

true = true.numpy()
#true[0,:,:,:] = -1
#true[1,:,:,0] = -1
#true[1,:,:,1] = -1
#true[1,:,:,2] = -1
#true[1,:,:,3] = -1
true = K.variable(true)


start = time.time()
adapt = K.eval(adaptive_dice_loss(true[:,:5,:5,:], pred[:,:5,:5,:]))
end_adapt = time.time()-start

ca_loss(true[:,:5,:5,:], pred[:,:5,:5,:])

print(K.eval(combined_loss(true,pred)))

exit()

# prep
batch_size = 1
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



a = None
for a in train_generator:
    # samples = np.add(a[1],np.random.normal(0,.1,a[1].shape))
    F = adaptive_dice_loss(K.variable(a[1]), K.variable(a[1]))
    print(K.eval(F))
    # print('dice loss: %f'%(time.time()-start))
    start = time.time()
    # F = dice_loss(K.variable(samples), K.variable(samples))
    # print(K.eval(F))
    # print('dice coeff: %f'%(time.time()-start))
