import os
import numpy as np
import math
import shutil
import tensorflow as tf


def separate():
    data_dir = "data"
    files = os.listdir("train_temp/")

    for file in files:
        category = (file.split('.'))[0]
        desDir = os.path.join(data_dir, category)
        if not os.path.exists(desDir):
            os.makedirs(desDir)
            shutil.move("train_temp/" + file, desDir)
        else:
            shutil.move("train_temp/" + file, desDir)


def get_file(file_dir):
    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            images.append(os.path.join(root, name))
        for name in sub_folders:
            temp.append(os.path.join(root, name))
        # print(files)
    labels = []
    for one_floder in temp:
        n_img = len(os.listdir(one_floder))
        letter = one_floder.split('/')[-1]

        if letter == 'cat':
            labels = np.append(labels, n_img * [0])
        else:
            labels = np.append(labels, n_img * [1])

    # shuffle
    temp = np.array([images, labels])
    temp = temp.transpose()

    # print("---temp shape: ", temp.shape)
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list


def get_batch(image_list, label_list, img_width, img_height, batch_size, capacity):
    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image, img_width, img_height)
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, label_batch


if __name__ == "__main__":
    separate()
#     get_file("data/")
#     get_batch()
