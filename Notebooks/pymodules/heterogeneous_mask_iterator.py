import numpy as np
import os
from keras_preprocessing.image.iterator import Iterator,BatchFromFilesMixin
from keras_preprocessing.image.utils import (array_to_img,
                    img_to_array,
                    load_img)
from create_one_hot_encoded_map_from_mask import get_one_hot_map
import tensorflow as tf

class HeteregenousMaskIterator(BatchFromFilesMixin, Iterator):

    def __init__(self,
                 directory,
                 image_data_generator,
                 color_map=None,
                 target_size=(256, 256),
                 color_mode='rgb',
                 masks=None,
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 subset=None,
                 interpolation='nearest',
                 dtype='float32'):
        super(HeteregenousMaskIterator, self).set_processing_attrs(image_data_generator,
                                                                   target_size,
                                                                   color_mode,
                                                                   data_format,
                                                                   save_to_dir,
                                                                   save_prefix,
                                                                   save_format,
                                                                   subset,
                                                                   interpolation)
        self.directory = directory
        self.dtype = dtype
        self.color_map = color_map
        # First, count the number of samples and classes.
        self.num_classes = len(masks)
        self.class_indices = dict(zip(masks, range(len(masks))))
        results = []
        self.filenames = []
        # Check if all mask directories contain the same account of mask images
        for dirpath in (os.path.join(directory, subdir) for subdir in masks):
            results.append(os.listdir(dirpath))
        if not self.checkEqual(results):
            print('Mask mismatch!')
            return
        self.samples = len(results[0])
        self._mask_identifiers = results[0]
        self._mask_classes = masks
        super(HeteregenousMaskIterator, self).__init__(self.samples, batch_size, shuffle, seed)
        print('Found %d masks containing %d classes.' %(self.samples, len(masks)))

    def checkEqual(self,iterator):
        iterator = iter(iterator)
        try:
            first = next(iterator)
        except StopIteration:
            return True
        return all(first == rest for rest in iterator)


    @property
    def filepaths(self):
        return self._mask_identifiers

    @property
    def labels(self):
        return self.classes

    @property  # mixin needs this property to work
    def sample_weight(self):
        # no sample weights will be returned
        return None

    def adjustMask(self,mask):
        one_hot_mask = []
        for i in range(len(mask)):
            one_hot_mask.append(get_one_hot_map(mask[i], self.color_map))
        one_hot_mask = tf.stack(one_hot_mask)
        return one_hot_mask


    def adj(self,mask):
        one_hot_mask = []
        for i in range(len(mask)):
            one_hot_mask.append(get_one_hot_map(mask[i]))
        one_hot_mask = tf.stack(one_hot_mask)
        return one_hot_mask

    def get_color_smallest_distance(self,color, color_map):
        '''
        Given a color (first parameter) this function gets the color from the color_map (second parameter) which is the
        closest to the given color.
        '''
        smallest_dist = abs(sum(color) - sum(color_map[0]))
        smallest_index = 0
        for i in range(1, len(color_map)):
            if abs(sum(color) - sum(color_map[i])) < smallest_dist:
                smallest_dist = abs(sum(color) - sum(color_map[i]))
                smallest_index = i
        return color_map[smallest_index]

    def get_one_hot_map(self,mask, class_index, background = [0, 0, 0]):
        '''
        Expects an numpy array of the mask. If no color_map is given, the color_map is generated from all colors in the mask.
        '''
        mask = mask.copy()
        img_width = mask.shape[1]
        img_height = mask.shape[0]
        one_hot_map = np.zeros((img_height, img_width, self.num_classes))
        # if value is ambiguous take the smallest distance to next value in binary color map
        for i in range(img_height):
            for j in range(img_width):
                img_val = mask[i, j, :].tolist()
                if img_val == background:
                    mask[i, j] = np.zeros(self.num_classes,dtype=int).tolist()
                else:
                    mask[i, j] = np.eye(self.num_classes,dtype=int)[class_index].tolist()
        return mask
        palette = self.color_map
        one_hot_map = []
        for colour in palette:
            class_map = tf.reduce_all(tf.equal(mask, colour), axis=-1)
            one_hot_map.append(class_map)
        one_hot_map = tf.stack(one_hot_map, axis=-1)
        one_hot_map = tf.cast(one_hot_map, tf.float32)
        return one_hot_map

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        for i, j in enumerate(index_array):
            # Iterate over all classes
            one_hot_arrays = np.zeros(())
            for k in range(self.num_classes):
                img = load_img(os.path.join(self.directory,self._mask_classes[k],filepaths[j]),
                               color_mode=self.color_mode,
                               target_size=self.target_size,
                               interpolation=self.interpolation)
                x = img_to_array(img, data_format=self.data_format)
                one_hot_arrays.append(self.get_one_hot_map(x,k))
                # Pillow images should be closed after `load_img`,
                # but not PIL images.
                if hasattr(img, 'close'):
                    img.close()
            # Add arrays into one array
            # CODE STILL MISSING
            if self.image_data_generator:
                params = self.image_data_generator.get_random_transform(x.shape)
                x = self.image_data_generator.apply_transform(x, params)
                x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        return batch_x