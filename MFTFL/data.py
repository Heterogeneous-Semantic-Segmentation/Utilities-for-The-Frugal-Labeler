from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
import segmentation_models as sm
import tensorflow as tf
import matplotlib.pyplot as plt
import random

from MFTFL.heterogeneous_mask_iterator import HeterogeneousMaskIterator

def adjust_image(img):
    if np.max(img) > 1:
        img = img / 255.
    return img


def train_generator(batch_size,
                    train_paths,
                    image_folders,
                    mask_folders,
                    aug_dict,
                    sample_weights,
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
    it = iter([train_paths, image_folders, mask_folders, heterogeneously_labeled_masks, sample_weights])
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        raise ValueError(
            'train_paths, image_folders, mask_folders, heterogeneously_labeled_masks and sample_distribution must all contain the same number of elements!')
    num_of_data_sources = the_len
    num_of_classes = []
    total_num_of_classes = 0
    for mask_folder in mask_folders:
        num_of_classes.append(len(mask_folder))
        total_num_of_classes+=len(mask_folder)
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    train_generators = []
    for i in range(len(train_paths)):
        img_generator = image_datagen.flow_from_directory(train_paths[i],
                                                          classes=[image_folders[i]],
                                                          class_mode=None,
                                                          color_mode=image_color_mode,
                                                          target_size=target_size,
                                                          batch_size=1,
                                                          save_to_dir=save_to_dir,
                                                          save_prefix=image_save_prefix,
                                                          seed=seed)
        mask_iterator = HeterogeneousMaskIterator(directory=train_paths[i],
                                                  image_data_generator=mask_datagen,
                                                  masks=mask_folders[i],
                                                  heterogeneously_labeled_masks=heterogeneously_labeled_masks[i],
                                                  missing_labels_ratio=missing_labels_ratio,
                                                  color_mode=mask_color_mode,
                                                  target_size=target_size,
                                                  batch_size=1,
                                                  save_to_dir=save_to_dir,
                                                  save_prefix=mask_save_prefix,
                                                  seed=seed,
                                                  include_background=include_background)
        train_generators.append(zip(img_generator, mask_iterator))
    random.seed(seed)
    while True:
        batch_masks = np.full((batch_size, target_size[0], target_size[1], total_num_of_classes + include_background), -1)
        batch_imgs = []
        indices = random.choices(list(range(num_of_data_sources)), weights=sample_weights, k=batch_size)
        for j in range(batch_size):
            image, mask = next(train_generators[indices[j]])
            batch_imgs.append(image[0])
            if include_background:
                batch_masks[j][:,:,total_num_of_classes] = mask[:,:,:,num_of_classes[indices[j]]]
                mask = mask[:,:,:,:-1]
            start_index = sum(num_of_classes[:indices[j]])
            num_of_masks = num_of_classes[indices[j]]
            batch_masks[j][:, :, start_index:start_index+num_of_masks] = mask
        yield adjust_image(np.array(batch_imgs)), np.ndarray.astype(batch_masks,np.float32)

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
