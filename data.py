import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def _parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    image = tf.image.resize(image, (255, 255))
    image = image / 255.
    return image, label


def make_tf(filename, batch_size):
    filenames = [os.path.join('Dataset/train2017', file)for file in np.loadtxt(filename, dtype=str, usecols=[0], delimiter=',')]
    labels = np.loadtxt(filename, dtype=float, usecols=[1, 2, 3, 4], delimiter=',')
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(batch_size*10).batch(batch_size).repeat()
    train_steps = len(filenames)//batch_size
    return dataset, train_steps

#
# if __name__ == '__main__':
#     iterator = make_tf('filenames.csv', 1)[0].take(1).as_numpy_iterator()
#     for images, bbox in iterator:
#         print(np.shape(images), bbox)
