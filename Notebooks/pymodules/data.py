from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from heterogeneous_mask_iterator import HeterogeneousMaskIterator
import segmentation_models as sm


def adjust_image(img):
    if np.max(img) > 1:
        img = img / 255.
    return img


def train_generator(batch_size,
                    train_path,
                    image_folder,
                    mask_folders,
                    aug_dict,
                    heterogeneously_labeled_masks=None,
                    missing_labels_ratio=0,
                    image_color_mode="rgb",
                    mask_color_mode="rgb",
                    image_save_prefix="image",
                    mask_save_prefix="mask",
                    save_to_dir=None,
                    target_size=(256, 256),
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
                                                           seed=seed)
    train_generator = zip(image_generator, heterogeneous_mask_generator)
    for (img, mask) in train_generator:
        yield adjust_image(img), mask


def test_generator(test_path, num_image=30, target_size=(256, 256), flag_multi_class=False, as_gray=True):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, ("%d" % i).zfill(4) + ".tif"), as_gray=as_gray)
        img = img / 255
        # https://github.com/zhixuhao/unet/issues/83
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


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
