from pathlib import Path

from keras import Model
from keras import layers
#from tensorflow.python.keras import backend as K
#import tensorflow as tf
import matplotlib.pyplot as plt
import os
#from unet_model import unet
#from data import train_generator
#from adaptive_objective_functions import adaptive_dice_loss,ca_loss,combined_loss
import segmentation_models as sm
import numpy as np
#from create_one_hot_encoded_map_from_mask import get_one_hot_map
from keras.callbacks import EarlyStopping, ModelCheckpoint

from data import train_generator,test_generator
import pickle
from unet_model import unet
from adaptive_objective_functions import adaptive_dice_loss,combined_loss,ca_loss

import tensorflow as tf

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


save_directory = './merge_datasets'

batch_size = 11
epochs = 500
iterations_per_epoch = 565 // batch_size
validation_steps  = 165 // batch_size
data_gen_args = dict(rotation_range=0.3,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.1,
                    #horizontal_flip=True,
                    fill_mode='nearest')



for aufteilung in [0.500,0.625,0.750,0.250,0.375]:
    directory = aufteilung


    train_gen = train_generator(batch_size=batch_size,
                                      train_path='../../partial_datasets/data_%.3f/dataset1/train_images'%aufteilung,
                                      image_folder='ventral_samples',
                                      mask_folders=['ventral_mask_atrium'],
                                      train_path2='../../partial_datasets/data_%.3f/dataset2/train_images'%aufteilung,
                                      image_folder2='ventral_samples',
                                      mask_folders2=['ventral_mask_heart'],
                                      heterogeneously_labeled_masks=[],
                                      missing_labels_ratio=0.,
                                      aug_dict=data_gen_args,
                                      image_color_mode='rgb',
                                      target_size=(256,256),
                                      include_background=True)


    test_gen = test_generator(batch_size=batch_size,
                                      test_path='../../partial_datasets/full_data/test_images',
                                      image_folder='ventral_samples',
                                      mask_folders=['ventral_mask_atrium','ventral_mask_heart'],
                                      image_color_mode='rgb',
                                      target_size=(256, 256))

    model = unet([combined_loss(2,0.7)],input_size = (256,256,3),output_filters=3)
    a = combined_loss(2,0.7)
    for t in train_gen:
        a(t[1],model.predict(t[0]))


    #model_checkpoint = ModelCheckpoint('%s/%s/weights_custom_loss.hdf5'%(save_directory,directory), monitor='val_loss', verbose=1, save_best_only=True)
    #early_stopping = EarlyStopping(monitor="val_loss",verbose = 1,mode='min',patience=20)
   #history = model.fit_generator(train_gen,steps_per_epoch=iterations_per_epoch,epochs=epochs,callbacks=[model_checkpoint,early_stopping],validation_data=test_gen,validation_steps=validation_steps,verbose=1)



sums = []
for t in train_gen:
    sums.append(t[1][0][0,0,:]+1)
    print(np.mean(sums,axis=0))
    plt.imshow(t[1][0])
    plt.show()
exit()

cc = combined_loss(None,0.5)



model = unet([combined_loss(None,0.5)],input_size = (512,512,3),output_filters=7)
history = model.fit_generator(train_gen,steps_per_epoch=iterations_per_epoch,epochs=epochs,validation_data=test_gen,validation_steps=validation_steps,verbose=1)



for t in train_gen:
    cc(t[1],model.predict(t[0]))
    print()
    break


exit()

test_gen = test_generator(batch_size=11,
                                  test_path='../../data/test_images',
                                  image_folder='ventral_samples',
                                  mask_folders=['ventral_mask_atrium','ventral_mask_heart'],
                                  image_color_mode='rgb',
                                  target_size=(256, 256))


model = unet([combined_loss(3,0.5)],input_size = (256,256,3),output_filters=4)

model.load_weights('weights_custom_loss.hdf5')

m = tf.keras.metrics.MeanIoU(num_classes=4)
m.update_state([0, 0, 1, 1], [0, 1, 0, 1])


c = combined_loss(2,0.5)

for a in test_gen:
    test = a[1]
    test[0,0,0] = [-1,-1,1]
    c(test,a[1])
    break


print()

exit()

file = open("../missing_labels_test_losses/7_alle_weg_06d/history", 'rb')
f = pickle.load(file)

file = open("../missing_labels_test_losses/9_alle_weg_06d/history", 'rb')
f2 = pickle.load(file)


plt.plot(f2.get('val_loss'))
plt.plot(f2.get('loss'))
plt.show()


exit()

#for i i
#n range(165):
#    i_str = str(i).zfill(5)
#    os.rename(r'../../data/test_images/ventral_mask_heart/color_frame_%d_mask.png'%i,r'../../data/test_images/ventral_mask_heart/%s_mask.png'%(str(i).zfill(5)))



from data import test_generator,train_generator

batch_size = 11

test_gen = test_generator(batch_size=batch_size,
                                  test_path='../../data/test_images',
                                  image_folder='ventral_samples',
                                  mask_folders=['ventral_mask_atrium', 'ventral_mask_bulbus', 'ventral_mask_heart'],
                                  image_color_mode='rgb',
                                  target_size=(256, 256))

i = 0
for a in test_gen:
    plt.imshow(a[0][0])
    plt.show()
    plt.imshow(a[1][0])
    plt.show()
    if i>10:
        break
    i+=1


exit()

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import os
from PIL import Image

# color map is different on test data
col_map = [[255,255,255],[20,20,20],[19,19,19],[0,0,0]]

i = 0.0
his = []
while i<0.6:
    file = open("../heteregenous_data_set_scenario/%.1f/history"%i,'rb')
    his.append(pickle.load(file))
    i+=0.1

for h in his:
    plt.plot(h.get('accuracy'))
    plt.show()

exit()


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

pred = model.predict(X_test[0].reshape(1,256,256,3))

import time

pred = K.variable(model.predict(X_test[0].reshape(1,256,256,3)))
#Y_test = Y_test.numpy()
#Y_test[0][:,:,1] = -1
#Y_test[0][:,:,0] = 1
true = K.variable(Y_test[0]).numpy().reshape(1,256,256,4)
true = tf.repeat(true,2,axis=0)
pred = tf.repeat(pred,2,axis=0)

lozz = combined_loss(3,0.5)

true = true.numpy()
true[0][:,:,:3] = -1
true[1][:,:,:3] = -1
true = K.variable(true)

lozz(true,pred)

true = true.numpy()
#true[0,:,:,:] = -1
#true[1,:,:,0] = -1
#true[1,:,:,1] = -1
#true[1,:,:,2] = -1
#true[1,:,:,3] = -1
true = K.variable(true)


start = time.time()

adapt = K.eval(ca_loss(true, pred,3))
end_adapt = time.time()-start



ca_loss(true[:,:5,:5,:], pred[:,:5,:5,:],3)

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
