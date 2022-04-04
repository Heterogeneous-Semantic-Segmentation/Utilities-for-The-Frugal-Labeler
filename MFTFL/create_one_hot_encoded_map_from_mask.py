from PIL import Image
from numpy import asarray
import tensorflow as tf



gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item


def sort_and_deduplicate(l):
    return list(uniq(sorted(l, reverse=True)))

def get_color_smallest_distance(color,color_map):
    '''
    Given a color (first parameter) this function gets the color from the color_map (second parameter) which is the
    closest to the given color.
    '''
    smallest_dist = abs(sum(color) - sum(color_map[0]))
    smallest_index = 0
    for i in range(1,len(color_map)):
        if abs(sum(color) - sum(color_map[i])) < smallest_dist:
            smallest_dist = abs(sum(color) - sum(color_map[i]))
            smallest_index = i
    return color_map[smallest_index]


def get_one_hot_map(mask, color_map=None):
    '''
    Expects an numpy array of the mask. If no color_map is given, the color_map is generated from all colors in the mask.
    '''
    mask = mask.copy()
    img_width = mask.shape[1]
    img_height = mask.shape[0]

    if not color_map:
        rgb_values= []
        for i in range(img_height):
            for j in range(img_width):
                rgb_values.append(mask[i, j, :].tolist())
        # All Color values which appear on the image
        palette = sort_and_deduplicate(rgb_values)
    else:
        palette = color_map
        # if value is ambiguous take the smallest distance to next value in color map
        for i in range(img_height):
            for j in range(img_width):
                img_val = mask[i, j, :].tolist()
                if not (img_val in color_map):
                    mask[i, j] = get_color_smallest_distance(img_val, color_map)

    one_hot_map = []
    for colour in palette:
        class_map = tf.reduce_all(tf.equal(mask, colour), axis=-1)
        one_hot_map.append(class_map)
    one_hot_map = tf.stack(one_hot_map, axis=-1)
    one_hot_map = tf.cast(one_hot_map, tf.float32)
    return one_hot_map
