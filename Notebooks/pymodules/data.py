from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from create_one_hot_encoded_map_from_mask import get_one_hot_map
import tensorflow as tf
import matplotlib.pyplot as plt
from heterogeneous_mask_iterator import HeteregenousMaskIterator


def adjustImage(img):
    if np.max(img) > 1:
        img = img / 255.
    return img


def adjustMask(mask, color_map, flag_multi_class):
    if flag_multi_class:
        one_hot_mask = []
        for i in range(len(mask)):
            one_hot_mask.append(get_one_hot_map(mask[i],color_map))
        one_hot_mask = tf.stack(one_hot_mask)
        return one_hot_mask
    else:
        # Requires the input image to only contain two colors, otherwise this will not work.
        mask = mask / 255.
        max_value = np.max(mask)
        mask[mask >= max_value] = 1
        mask[mask < max_value] = 0
        return mask


def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "rgb",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,save_to_dir = None,target_size = (256,256),seed = 1,color_map=None):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    hetergenous_mask_generator = HeteregenousMaskIterator(train_path,
                                                          mask_datagen,
                                                          masks= ['ventral_mask_atrium', 'ventral_mask_bulbus', 'ventral_mask_heart'],
                                                          color_map=color_map,
                                                          color_mode = mask_color_mode,
                                                          target_size = target_size,
                                                          batch_size = batch_size,
                                                          save_to_dir = save_to_dir,
                                                          save_prefix  = mask_save_prefix,
                                                          seed = seed)

    train_generator = zip(image_generator, mask_generator,hetergenous_mask_generator)
    for (img,mask,test) in train_generator:

        plt.imshow(img[0])
        plt.show()

        plt.imshow(mask[0])
        plt.show()
        plt.imshow(test[0])
        plt.show()

        adjusted_img = adjustImage(img)
        adjusted_mask = adjustMask(mask,color_map,flag_multi_class)
        yield (adjusted_img, adjusted_mask)


def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, ("%d" % i).zfill(4) + ".tif"),as_gray = as_gray)
        img = img / 255
        # https://github.com/zhixuhao/unet/issues/83
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustImage(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
        print(color_dict[i])
    return img_out / 255

