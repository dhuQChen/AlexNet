import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import create_and_read_TFRecord2 as reader2

learning_rate = 1e-4
training_iters = 2000
batch_size = 64
dispaly_step = 5
n_classes = 2
n_fc1 = 4096
n_fc2 = 2048

X_train, y_train = reader2.get_file("./data")
image_batch, label_batch = reader2.get_batch(X_train, y_train, 227, 227, batch_size, 512)

x = tf.placeholder(tf.float32, [None, 227, 227, 3])
y = tf.placeholder(tf.int32, [None, n_classes])

W_conv = {
    'conv1': tf.Variable(tf.truncated_normal([11, 11, 3, 96], stddev=0.01)),
    'conv2': tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.01)),
    'conv3': tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.01)),
    'conv4': tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.01)),
    'conv5': tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.01)),
    'fc1': tf.Variable(tf.truncated_normal([6 * 6 * 256, n_fc1], stddev=0.01)),
    'fc2': tf.Variable(tf.truncated_normal([n_fc1, n_fc2], stddev=0.01)),
    'fc3': tf.Variable(tf.truncated_normal([n_fc2, n_classes], stddev=0.01))
}

b_conv = {
    'conv1': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[96])),
    'conv2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256])),
    'conv3': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384])),
    'conv4': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384])),
    'conv5': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256])),
    'fc1': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc1])),
    'fc2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc2])),
    'fc3': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_classes]))
}

x_image = tf.reshape(x, [-1, 227, 227, 3])

"""
VALID: out_size = ceil((in_size - filter + 2 * padding) / 2 + 1)
"""
# 卷积层1
conv1 = tf.nn.conv2d(x_image, W_conv['conv1'], strides=[1, 4, 4, 1], padding='VALID')
print("---conv1 shape---", conv1.shape)
conv1 = tf.nn.bias_add(conv1, b_conv['conv1'])
conv1 = tf.nn.relu(conv1)
pool1 = tf.nn.avg_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
print("---pool1 shape---", pool1.shape)
norm1 = tf.nn.lrn(pool1, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 卷积层2
"""
SAME: padding = 2, because of 5 % 2 = 1.
"""
conv2 = tf.nn.conv2d(norm1, W_conv['conv2'], strides=[1, 1, 1, 1], padding='SAME')
print("---conv2 shape---", conv2.shape)
conv2 = tf.nn.bias_add(conv2, b_conv['conv2'])
conv2 = tf.nn.relu(conv2)
pool2 = tf.nn.avg_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
print("---pool2 shape---", pool2.shape)
norm2 = tf.nn.lrn(pool2, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 卷积层3
conv3 = tf.nn.conv2d(norm2, W_conv['conv3'], strides=[1, 1, 1, 1], padding='SAME')
print("---conv3 shape---", conv3.shape)
conv3 = tf.nn.bias_add(conv3, b_conv['conv3'])
conv3 = tf.nn.relu(conv3)

# 卷积层4
conv4 = tf.nn.conv2d(conv3, W_conv['conv4'], strides=[1, 1, 1, 1], padding='SAME')
print("---conv4 shape---", conv4.shape)
conv4 = tf.nn.bias_add(conv4, b_conv['conv4'])
conv4 = tf.nn.relu(conv4)

# 卷积层5
conv5 = tf.nn.conv2d(conv4, W_conv['conv5'], strides=[1, 1, 1, 1], padding='SAME')
print("---conv5 shape---", conv5.shape)
conv5 = tf.nn.bias_add(conv5, b_conv['conv5'])
conv5 = tf.nn.relu(conv5)
pool5 = tf.nn.avg_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
print("---pool5 shape---", pool5.shape)

reshape = tf.reshape(pool5, [-1, 6 * 6 * 256])

# fc1
fc1 = tf.add(tf.matmul(reshape, W_conv['fc1']), b_conv['fc1'])
fc1 = tf.nn.relu(fc1)
fc1 = tf.nn.dropout(fc1, 0.5)

# fc2
fc2 = tf.add(tf.matmul(fc1, W_conv['fc2']), b_conv['fc2'])
fc2 = tf.nn.relu(fc2)
fc2 = tf.nn.dropout(fc2, 0.5)

# fc3
fc3 = tf.add(tf.matmul(fc2, W_conv['fc3']), b_conv['fc3'])

# 定义损失
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_pred = tf.equal(tf.argmax(fc3, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


def one_hot(labels):
    '''one-hot 编码'''
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels


def train(epoches):
    with tf.Session() as sess:
        sess.run(init)
        save_model = "./model/AlexNetModel.ckpt"
        train_writer = tf.summary.FileWriter("./log", sess.graph)
        saver = tf.train.Saver()

        c = []
        start_time = time.time()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        step = 0
        for i in range(epoches):
            step = i
            image, label = sess.run([image_batch, label_batch])

            labels = one_hot(label)

            sess.run(optimizer, feed_dict={x: image, y: labels})
            loss_record = sess.run(loss, feed_dict={x: image, y: labels})
            print("now the loss is: %f" % loss_record)

            c.append(loss_record)
            end_time = time.time()
            print("time:", (end_time - start_time))
            start_time = end_time
            print("--------%d epoch is finished-------" % i)
        print("Optimized Finished!")
        saver.save(sess, save_model)
        print("Model Save Finished!")

        coord.request_stop()
        coord.join(threads)

        plt.plot(c)
        plt.xlabel('iters')
        plt.ylabel('loss')
        plt.tight_layout()
        plt.savefig("cnn-tf-AlexNet.png", dpi=200)


from PIL import Image
import os


def Evaluate(testfile):

    count = 0
    sum = 0

    with tf.Session() as sess:
        for root, sub_folders, files in os.walk(testfile):
            for name in files:
                sum += 1
                imagefile = os.path.join(root, name)
                print(imagefile)
                image = Image.open(imagefile)
                image = image.resize([227, 227])
                image_array = np.array(image)

                image = tf.cast(image_array, tf.float32)
                image = tf.image.per_image_standardization(image)
                image = tf.reshape(image, [1, 227, 227, 3])

                saver = tf.train.Saver()

                save_model = tf.train.latest_checkpoint('./model')
                saver.restore(sess,save_model)
                image = tf.reshape(image,[1,227,227,3])
                image = sess.run(image)
                prediction = sess.run(fc3,feed_dict={x: image})

                max_index = np.argmax(prediction)
                if max_index == 0 and name.split('.')[0] == 'cat':
                    count += 1
                if max_index == 1 and name.split('.')[0] == 'dog':
                    count += 1

            print(" The accuracy is: ", count / sum)


if __name__ == '__main__':
    train(training_iters)
