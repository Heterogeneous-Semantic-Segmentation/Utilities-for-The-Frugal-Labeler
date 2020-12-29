from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from heterogeneous_mask_iterator import HeterogeneousMaskIterator
import segmentation_models as sm
import tensorflow as tf


def adjust_image(img):
    if np.max(img) > 1:
        img = img / 255.
    return img


def train_generator(batch_size,
                    train_path,
                    train_path2,
                    image_folder,
                    image_folder2,
                    mask_folders,
                    mask_folders2,
                    aug_dict,
                    heterogeneously_labeled_masks=None,
                    missing_labels_ratio=0,
                    image_color_mode="rgb",
                    mask_color_mode="rgb",
                    image_save_prefix="image",
                    mask_save_prefix="mask",
                    save_to_dir=None,
                    target_size=(256, 256),
                    include_background=True,
                    seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(train_path,
                                                        classes=[image_folder],
                                                        class_mode=None,
                                                        color_mode=image_color_mode,
                                                        target_size=target_size,
                                                        batch_size=batch_size,
                                                        save_to_dir=save_to_dir,
                                                        save_prefix=image_save_prefix,
                                                        seed=seed)
    heterogeneous_mask_generator = HeterogeneousMaskIterator(directory=train_path,
                                                             image_data_generator=mask_datagen,
                                                             masks=mask_folders,
                                                             heterogeneously_labeled_masks=heterogeneously_labeled_masks,
                                                             missing_labels_ratio=missing_labels_ratio,
                                                             color_mode=mask_color_mode,
                                                             target_size=target_size,
                                                             batch_size=batch_size,
                                                             save_to_dir=save_to_dir,
                                                             save_prefix=mask_save_prefix,
                                                             seed=seed,
                                                             include_background=include_background)
    image_generator2 = image_datagen.flow_from_directory(train_path2,
                                                        classes=[image_folder2],
                                                        class_mode=None,
                                                        color_mode=image_color_mode,
                                                        target_size=target_size,
                                                        batch_size=batch_size,
                                                        save_to_dir=save_to_dir,
                                                        save_prefix=image_save_prefix,
                                                        seed=seed)
    heterogeneous_mask_generator2 = HeterogeneousMaskIterator(directory=train_path2,
                                                              image_data_generator=mask_datagen,
                                                              masks=mask_folders2,
                                                              heterogeneously_labeled_masks=heterogeneously_labeled_masks,
                                                              missing_labels_ratio=missing_labels_ratio,
                                                              color_mode=mask_color_mode,
                                                              target_size=target_size,
                                                              batch_size=batch_size,
                                                              save_to_dir=save_to_dir,
                                                              save_prefix=mask_save_prefix,
                                                              seed=seed,
                                                              include_background=include_background)
    train_generator = zip(image_generator, heterogeneous_mask_generator, image_generator2, heterogeneous_mask_generator2)

    total_batches_seen = 0
    for (img1, mask1,img2, mask2) in train_generator:
        if seed is not None:
            np.random.seed(seed + total_batches_seen)
            total_batches_seen += 1
        # [mask1,mask2,bg]
        mask1 = np.insert(mask1,1,np.full(mask1.shape[:3],-1),axis=3)
        mask2 = np.insert(mask2,0,np.full(mask2.shape[:3],-1),axis=3)
        combined_masks = tf.concat((mask1,mask2),axis=0)
        combined_imgs = tf.concat((img1,img2),axis=0)

        indices = tf.range(start=0, limit=tf.shape(combined_masks)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices,seed=seed)

        yield adjust_image(tf.gather(combined_imgs, shuffled_indices)[:mask1.shape[0]]), tf.gather(combined_masks, shuffled_indices)[:mask1.shape[0]]


def test_generator(batch_size,
                   test_path,
                   image_folder,
                   mask_folders,
                   image_color_mode="rgb",
                   mask_color_mode="rgb",
                   image_save_prefix="image",
                   mask_save_prefix="mask",
                   save_to_dir=None,
                   target_size=(256, 256),
                   seed=1,
                   include_background=True):
    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(test_path,
                                                        classes=[image_folder],
                                                        class_mode=None,
                                                        color_mode=image_color_mode,
                                                        target_size=target_size,
                                                        batch_size=batch_size,
                                                        save_to_dir=save_to_dir,
                                                        save_prefix=image_save_prefix,
                                                        seed=seed)
    heterogeneous_mask_generator = HeterogeneousMaskIterator(directory=test_path,
                                                             image_data_generator=mask_datagen,
                                                             masks=mask_folders,
                                                             heterogeneously_labeled_masks=[],
                                                             missing_labels_ratio=0,
                                                             color_mode=mask_color_mode,
                                                             target_size=target_size,
                                                             batch_size=batch_size,
                                                             save_to_dir=save_to_dir,
                                                             save_prefix=mask_save_prefix,
                                                             seed=seed,
                                                             include_background=include_background)
    test_generator = zip(image_generator, heterogeneous_mask_generator)
    for (img, mask) in test_generator:
        yield adjust_image(img), mask


def gene_train_numpy(image_path,
                     mask_path,
                     flag_multi_class=False,
                     num_class=2,
                     image_prefix="image",
                     mask_prefix="mask",
                     image_as_gray=True,
                     mask_as_gray=True):
    image_name_arr = glob.glob(os.path.join(image_path, "%s*.png" % image_prefix))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix), as_gray=mask_as_gray)
        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjust_image(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr


def label_visualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
        print(color_dict[i])
    return img_out / 255
