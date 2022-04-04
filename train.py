#  Imports
import os
from os import walk
import pickle
from PIL import Image
from numpy import asarray
import segmentation_models as sm
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import  EarlyStopping, ReduceLROnPlateau 
import tensorflow as tf
import segmentation_models as sm
import tensorflow.keras
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from MFTFL.create_one_hot_encoded_map_from_mask import get_one_hot_map
from MFTFL.adaptive_objective_functions import adaptive_dice_loss,ca_loss
from MFTFL.data import train_generator,test_generator
from MFTFL.unet_model import unet


import tensorflow as tf


# color map is different on test data
col_map = [[255,255,255],[20,20,20],[19,19,19],[0,0,0]]

X_test = []
for filepath in os.listdir('data/test_images/ventral_samples_R0004'):
    image = Image.open('data/test_images/ventral_samples_R0004/'+filepath)
    image = image.resize((256, 256))
    # convert image to numpy array
    data = np.asarray(image)
    data = data/255.
    X_test.append(data)
X_test = np.array(X_test)
Y_test = []
for filepath in os.listdir('data/test_images/ventral_mask_combined_R0004'):
    image = Image.open('data/test_images/ventral_mask_combined_R0004/'+filepath)
    image = image.resize((256, 256)) 
    Y_test.append(get_one_hot_map(np.asarray(image),col_map))
Y_test = tf.stack(Y_test)


learned_masks = []
classes_color_dict = {0:[0,150,130],1:[64,64,64],2:[255,255,255],3:[0,0,0]}

class CustomCallback(tensorflow.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        predicted = model.predict(X_test[1].reshape(1,256,256,3))
        learned_masks.append(predicted[0])
        disp_array = np.repeat(np.zeros(list(predicted[0].shape[:2])).reshape(256,256,1),3,axis=2)
        for key in classes_color_dict:
            true_values = np.full(list(predicted[0].shape[:2]) + [3],classes_color_dict.get(key))
            disp_array = np.where(np.repeat((np.argmax(predicted[0],axis=2) == key).reshape(256,256,1),3,axis=2),true_values,disp_array)
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(X_test[1])
        axarr[1].imshow(disp_array.astype(int))
        plt.show()
        
        
save_directory = './missing_labels_test'

batch_size = 11
epochs = 250
iterations_per_epoch = 250
data_gen_args = dict(rotation_range=0.3,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    fill_mode='nearest')

missing_ratio = 0.0
for missing_ratio in [0.0,0.3,0.7]:
    
    # Clear learned masks
    learned_masks = []
    
    print('---------------------')
    print('Missing Ratio: %f'%missing_ratio)
    print('---------------------')

    Path('%s/%s'%(save_directory,missing_ratio)).mkdir(exist_ok=True)

    train_gen = train_generator(batch_size=batch_size,
                                      train_paths=['data/train_images'],
                                      image_folders=['ventral_samples',],
                                      mask_folders=[['ventral_mask_atrium', 'ventral_mask_bulbus', 'ventral_mask_heart']]*1,
                                      heterogeneously_labeled_masks=[['ventral_mask_atrium', 'ventral_mask_bulbus',
                                                                     'ventral_mask_heart']]*1,
                                      missing_labels_ratio=missing_ratio,
                                      sample_weights=[1],
                                      aug_dict=data_gen_args,
                                      image_color_mode='rgb',
                                      target_size=(256, 256))
    val_datagen = ImageDataGenerator()
    val_gen = val_datagen.flow(X_test, Y_test, batch_size=batch_size)
    model = unet(adaptive_dice_loss,input_size = (256,256,3),output_filters=4)
    model_checkpoint = ModelCheckpoint('%s/%s/weights_custom_loss.hdf5'%(save_directory,missing_ratio), monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor="val_loss",verbose = 1,mode='min',patience=30)
    reduce_lr =  ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 10,verbose = 0, mode = "auto", epsilon = 1e-04, cooldown = 0,min_lr = 1e-5)
    history = model.fit_generator(train_gen,steps_per_epoch=iterations_per_epoch,epochs=epochs,callbacks=[model_checkpoint,CustomCallback(),early_stopping,reduce_lr],validation_data=val_gen,verbose=1)

    with open('%s/%s/history'%(save_directory,missing_ratio), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    with open('%s/%s/learned_masks'%(save_directory,missing_ratio), 'wb') as file_pi:
        pickle.dump(learned_masks, file_pi)

