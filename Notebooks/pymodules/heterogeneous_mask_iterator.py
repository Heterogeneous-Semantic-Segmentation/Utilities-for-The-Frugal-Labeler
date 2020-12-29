import numpy as np
import os
from keras_preprocessing.image.iterator import Iterator, BatchFromFilesMixin
from keras_preprocessing.image.utils import (array_to_img,
                                             img_to_array,
                                             load_img)
import tensorflow as tf

DELETED_MASK_IDENTIFIER = -1


def check_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


class HeterogeneousMaskIterator(BatchFromFilesMixin, Iterator):

    def __init__(self,
                 directory,
                 image_data_generator,
                 target_size=(256, 256),
                 color_mode='rgb',
                 masks=None,
                 heterogeneously_labeled_masks=[],
                 missing_labels_ratio=0,
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 subset=None,
                 interpolation='nearest',
                 dtype='float32',
                 background_color=None,
                 include_background=True,
                 save_dropped_labels=True):
        super(HeterogeneousMaskIterator, self).set_processing_attrs(image_data_generator,
                                                                    target_size,
                                                                    color_mode,
                                                                    data_format,
                                                                    save_to_dir,
                                                                    save_prefix,
                                                                    save_format,
                                                                    subset,
                                                                    interpolation)
        self.include_background = include_background
        if save_dropped_labels:
            self.save_dropped_labels = save_dropped_labels
            self.dropped_labels_memory = {}
        self.background_color = background_color
        self.heterogeneously_labeled_masks = heterogeneously_labeled_masks
        self.missing_labels_ratio = missing_labels_ratio
        self.directory = directory
        self.dtype = dtype
        self.target_size = target_size
        # First, count the number of samples and classes.
        self.num_classes = len(masks)
        self.class_indices = dict(zip(masks, range(len(masks))))
        self.heterogeneous_masks = []
        for mask in heterogeneously_labeled_masks:
            self.heterogeneous_masks.append(self.class_indices.get(mask))
        results = []
        self.filenames = []
        # Check if all mask directories contain the same account of mask images
        for dirpath in (os.path.join(directory, subdir) for subdir in masks):
            results.append(os.listdir(dirpath))
        if not check_equal(results):
            raise ValueError(
                'Directories do not contain the same amount of masks (some directories have missing masks).')
        self.samples = len(results[0])
        self._mask_identifiers = results[0]
        self._mask_classes = masks
        super(HeterogeneousMaskIterator, self).__init__(self.samples, batch_size, shuffle, seed)
        print('Found %d masks containing %d classes.' % (self.samples, len(masks)))
        if (not self.heterogeneously_labeled_masks is None) and (not self.missing_labels_ratio is None):
            print('Removing %d%% of masks of the following class(es): %s' % (
            (self.missing_labels_ratio * 100), self.heterogeneously_labeled_masks))

    @property
    def filepaths(self):
        return self._mask_identifiers

    @property
    def labels(self):
        return self._mask_classes

    @property  # mixin needs this property to work
    def sample_weight(self):
        # no sample weights will be returned
        return None

    def get_one_hot_map(self, mask, class_index,additional_bg_class, background=None):
        if background is None:
            background = [0, 0, 0]
        reduced_mask = tf.reduce_all(tf.equal(mask, background), axis=-1)
        class_indices_mask = tf.where(reduced_mask, self.num_classes + 1, class_index)
        return tf.one_hot(class_indices_mask, self.num_classes + additional_bg_class)

    def remove_masks(self, batch_x):
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        for index in self.heterogeneous_masks:
            masked_batch_x = batch_x[:, :, :, index]
            delete_mask = np.where(np.random.sample(len(masked_batch_x)) <= self.missing_labels_ratio)
            masked_batch_x[delete_mask] = DELETED_MASK_IDENTIFIER
        return batch_x

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples in one-hot-encoded format.
        """
        additional_bg_class = 0
        if self.include_background:
            additional_bg_class = 1
        batch_x = np.zeros((len(index_array),) + (self.target_size[0], self.target_size[1], self.num_classes + additional_bg_class),
                           dtype=self.dtype)
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        known_label_keys = self.dropped_labels_memory.keys()
        known_label_drops = {}
        unset_label_drops = []
        for i, j in enumerate(index_array):
            if j in known_label_keys:
                known_label_drops[i] = j
            else:
                unset_label_drops.append(i)
            one_hot_map = np.zeros((self.target_size[0], self.target_size[1], self.num_classes + additional_bg_class), dtype=np.float32)
            # Iterate over all classes
            params = None
            reserved_pixels = None
            for k in range(self.num_classes):
                filepath = os.path.join(self.directory, self._mask_classes[k], filepaths[j])
                img = load_img(filepath,
                               color_mode=self.color_mode,
                               target_size=self.target_size,
                               interpolation=self.interpolation)
                x = img_to_array(img, data_format=self.data_format)
                # Pillow images should be closed after `load_img`,
                # but not PIL images.
                if hasattr(img, 'close'):
                    img.close()
                if self.image_data_generator:
                    # Params need to be set once for every image (not for every mask)
                    if params is None:
                        params = self.image_data_generator.get_random_transform(x.shape)
                    x = self.image_data_generator.apply_transform(x, params)
                    x = self.image_data_generator.standardize(x)
                if reserved_pixels is None:
                    reserved_pixels = np.zeros(x.shape)
                x = np.where(reserved_pixels, 0, x)
                reserved_pixels += x
                one_hot_map += self.get_one_hot_map(x, k, background=self.background_color,additional_bg_class=additional_bg_class)
            # If one_hot_map has a max value >1 whe have overlapping classes -> prohibited
            one_hot_map = one_hot_map.numpy()
            if one_hot_map.max() > 1:
                raise ValueError('Mask mismatch: classes are not mutually exclusive (multiple class definitions for '
                                 'one pixel).')
            if self.include_background:
                # Background class is everywhere, where the one-hot encoding has only zeros
                one_hot_map = tf.where(tf.repeat(tf.reshape(tf.math.count_nonzero(one_hot_map == tf.zeros(self.num_classes+1), axis=2), [img.height, img.width, 1]), self.num_classes+1,axis=2) == self.num_classes+1, tf.one_hot(tf.constant(self.num_classes, shape=one_hot_map.shape[:2]), self.num_classes+1), one_hot_map)
            batch_x[i] = one_hot_map
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

        # Only where labels are not already known
        if len(self.heterogeneously_labeled_masks) > 0:
            batch_x[unset_label_drops] = self.remove_masks(np.take(batch_x,unset_label_drops,axis=0))

            # Known labels
            for item in known_label_drops.items():
                index_in_batch = item[0]
                index_in_memory = item[1]
                memory_binary_mask = self.dropped_labels_memory.get(index_in_memory)
                batch_x[index_in_batch,:,:,:][memory_binary_mask] = DELETED_MASK_IDENTIFIER

            # Extend Memory of known deletion masks.
            delete_mask = np.where(batch_x==DELETED_MASK_IDENTIFIER,True,False)
            for i in range(len(index_array)):
                self.dropped_labels_memory[index_array[i]] = delete_mask[i,:,:,:]
        return batch_x
