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
                 heterogeneously_labeled_masks=None,
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
                 background_color=None):
        super(HeterogeneousMaskIterator, self).set_processing_attrs(image_data_generator,
                                                                    target_size,
                                                                    color_mode,
                                                                    data_format,
                                                                    save_to_dir,
                                                                    save_prefix,
                                                                    save_format,
                                                                    subset,
                                                                    interpolation)
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
            raise ValueError('Directories do not contain the same amount of masks (some directories have missing masks).')
        self.samples = len(results[0])
        self._mask_identifiers = results[0]
        self._mask_classes = masks
        super(HeterogeneousMaskIterator, self).__init__(self.samples, batch_size, shuffle, seed)
        print('Found %d masks containing %d classes.' % (self.samples, len(masks)))
        if (not self.heterogeneously_labeled_masks is None) and (not self.missing_labels_ratio is None):
            print('Removing %d%% of masks of the following class(es): %s'%((self.missing_labels_ratio*100),self.heterogeneously_labeled_masks))

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

    def get_one_hot_map(self,mask,class_index,background=None):
        if background is None:
            background = [0, 0, 0]
        reduced_mask = tf.reduce_all(tf.equal(mask, background), axis=-1)
        class_indices_mask = tf.where(reduced_mask, self.num_classes + 1, class_index)
        return tf.one_hot(class_indices_mask, self.num_classes)




    def remove_masks(self,batch_x):
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        for index in self.heterogeneous_masks:
            masked_batch_x = batch_x[:, :, :, index]
            delete_mask = np.where(np.random.sample(len(masked_batch_x)) <= self.missing_labels_ratio)
            masked_batch_x[delete_mask] = DELETED_MASK_IDENTIFIER


    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples in one-hot-encoded format.
        """
        batch_x = np.zeros((len(index_array),) + (self.target_size[0], self.target_size[1], self.num_classes),
                           dtype=self.dtype)
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        for i, j in enumerate(index_array):
            one_hot_map = np.zeros((self.target_size[0], self.target_size[1], self.num_classes), dtype=np.float32)
            # Iterate over all classes
            params = None
            for k in range(self.num_classes):
                img = load_img(os.path.join(self.directory, self._mask_classes[k], filepaths[j]),
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
                one_hot_map += self.get_one_hot_map(x, k,self.background_color)
            # If one_hot_map has a max value >1 whe have overlapping classes -> prohibited
            one_hot_map = one_hot_map.numpy()
            if one_hot_map.max() > 1:
                raise ValueError('Mask mismatch: classes are not mutually exclusive (multiple class definitions for '
                                 'one pixel).')
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
        self.remove_masks(batch_x)
        return batch_x
