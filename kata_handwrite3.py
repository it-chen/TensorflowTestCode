#coding=utf-8

import tensorflow as tf
import random
import os
import numpy as np
import time
from PIL import Image

tf.app.flags.DEFINE_integer('max_steps', 5000, 'the max training steps ')
tf.app.flags.DEFINE_integer('char_count',46,'识别char的最大值')
tf.app.flags.DEFINE_string('checkpoint_dir', 'C:\\data\\kata1\\checkpoint', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', 'C:\\data\\kata1\\train', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', 'C:\\data\\kata1\\test', 'the test dataset dir')
tf.app.flags.DEFINE_string('mode', 'inference', 'the run mode')

tf.app.flags.DEFINE_boolean('random_flip_up_down', False,"Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_flip_left_right', False,"whether to random flip left and right")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")
tf.app.flags.DEFINE_integer('image_size', 64,"Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whethet to change the rbg to gray")
tf.app.flags.DEFINE_integer('eval_steps', 10, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 100, "the steps to save")
tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('epoch', 1, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('val_batch_size', 128, 'whether to restore from checkpoint')

FLAGS = tf.app.flags.FLAGS

# 画像とlabelsを取得
def get_imagesfile(data_dir):
    filenames = []
    for root, sub_folder, file_list in os.walk(data_dir):
        filenames += [os.path.join(root, file_path) for file_path in file_list]
    labels = [file_name.split('\\')[-2] for file_name in filenames]
    file_labels = [(file, labels[index]) for index, file in enumerate(filenames)]
    random.shuffle(file_labels)
    return file_labels

# 画像の事前処理
def pre_process(images):
    if FLAGS.random_flip_up_down:
        images = tf.image.random_flip_up_down(images)
    if FLAGS.random_flip_left_right:
        images = tf.image.random_flip_left_right(images)
    if FLAGS.random_brightness:
        images = tf.image.random_brightness(images, max_delta=0.3)
    if FLAGS.random_contrast:
        images = tf.image.random_contrast(images, 0.8, 1.2)
    new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
    images = tf.image.resize_images(images, new_size)
    return images

# 画像バチーでtraining、メモリ限りがあるの為
def batch_data(file_labels, sess, batch_size=128):
    image_list = [file_label[0] for file_label in file_labels]
    label_list = [int(file_label[1]) for file_label in file_labels]
    images_tensor = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels_tensor = tf.convert_to_tensor(label_list, dtype=tf.int64)
    input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor])
    labels = input_queue[1]
    images_content = tf.read_file(input_queue[0])
    # dtype:float64⇒float32
    images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
    # 画像の事前処理
    images = pre_process(images)
    # one hot
    labels = tf.one_hot(labels, FLAGS.char_count)
    image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
                                                      min_after_dequeue=10000)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return image_batch, label_batch, coord, threads

#ネットワーク
def network(images, labels=None):
    endpoints = {}

    # conv1_1
    with tf.variable_scope('conv1_1') as scope:
        conv1_1_weight = tf.Variable(tf.truncated_normal([3, 3, 1, 32],stddev=0.1, dtype=tf.float32))
        conv1_1 = tf.nn.conv2d(images, conv1_1_weight, [1, 1, 1, 1], padding='SAME')
        conv1_1_bias = tf.Variable(tf.zeros([32], dtype=tf.float32))
        conv1_1 = tf.nn.relu(tf.nn.bias_add(conv1_1, conv1_1_bias), name=scope.name)

    # conv1_2
    with tf.variable_scope('conv1_2') as scope:
        conv1_2_weight = tf.Variable(tf.truncated_normal([3, 3, 32, 32],stddev=0.1, dtype=tf.float32))
        conv1_2 = tf.nn.conv2d(conv1_1, conv1_2_weight, [1, 1, 1, 1], padding='SAME')
        conv1_2_bias = tf.Variable(tf.zeros([32], dtype=tf.float32))
        conv1_2 = tf.nn.relu(tf.nn.bias_add(conv1_2, conv1_2_bias), name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # conv2_1
    with tf.variable_scope('conv2_1') as scope:
        conv2_1_weight = tf.Variable(tf.truncated_normal([3, 3, 32, 64],stddev=0.1, dtype=tf.float32))
        conv2_1 = tf.nn.conv2d(pool1, conv2_1_weight, [1, 1, 1, 1], padding='SAME')
        conv2_1_bias = tf.Variable(tf.zeros([64], dtype=tf.float32))
        conv2_1 = tf.nn.relu(tf.nn.bias_add(conv2_1, conv2_1_bias), name=scope.name)

    # conv2_2
    with tf.variable_scope('conv2_2') as scope:
        conv2_2_weight = tf.Variable(tf.truncated_normal([3, 3, 64, 64],stddev=0.1, dtype=tf.float32))
        conv2_2 = tf.nn.conv2d(conv2_1, conv2_2_weight, [1, 1, 1, 1], padding='SAME')
        conv2_2_bias = tf.Variable(tf.zeros([64], dtype=tf.float32))
        conv2_2 = tf.nn.relu(tf.nn.bias_add(conv2_2, conv2_2_bias), name=scope.name)

    # pool2
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # fc6
    with tf.variable_scope('fc6') as scope:
        if FLAGS.mode == "inference":
            reshape = tf.reshape(pool2, [1, -1])
        else:
            reshape = tf.reshape(pool2, [FLAGS.val_batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.Variable(tf.truncated_normal([dim, 4096],stddev=0.1, dtype=tf.float32))
        biases = tf.Variable(tf.zeros([4096], dtype=tf.float32))
        fc6 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        if FLAGS.mode == "train":
            fc6 = tf.nn.dropout(fc6, 0.5)

    # fc7
    with tf.variable_scope('fc7') as scope:
        weights = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.1, dtype=tf.float32))
        biases = tf.Variable(tf.zeros([4096], dtype=tf.float32))
        fc7 = tf.nn.relu(tf.matmul(fc6, weights) + biases, name=scope.name)
        if FLAGS.mode == "train":
            fc7 = tf.nn.dropout(fc7, 0.5)

    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(tf.truncated_normal([4096, FLAGS.char_count], stddev=0.1, dtype=tf.float32))
        biases = tf.Variable(tf.zeros([FLAGS.char_count], dtype=tf.float32))
        out = tf.add(tf.matmul(fc7, weights), biases, name=scope.name)

    global_step = tf.Variable(initial_value=0)
    if labels is not None:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels,logits = out))
        train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss, global_step=global_step)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(labels, 1)), tf.float32))
        merged_summary_op = tf.summary.merge_all()

    output_score = tf.nn.softmax(out)
    predict_val_top3, predict_index_top3 = tf.nn.top_k(output_score, k=3)

    endpoints['global_step'] = global_step
    if labels is not None:
        endpoints['labels'] = labels
        endpoints['train_op'] = train_op
        endpoints['loss'] = loss
        endpoints['accuracy'] = accuracy
        endpoints['merged_summary_op'] = merged_summary_op
    endpoints['output_score'] = output_score
    endpoints['predict_val_top3'] = predict_val_top3
    endpoints['predict_index_top3'] = predict_index_top3
    return endpoints

# 検証
def validation(path = FLAGS.test_data_dir):

    sess = tf.Session()

    file_labels = get_imagesfile(path)
    test_size = len(file_labels)

    val_batch_size = FLAGS.val_batch_size
    test_steps = int(test_size / val_batch_size)

    final_predict_val = []
    final_predict_index = []
    groundtruth = []
    file_paths = []

    for i in range(test_steps):
        start = i * val_batch_size
        end = (i + 1) * val_batch_size
        images_batch = []
        labels_batch = []
        labels_max_batch = []

        for j in range(start, end):
            image_path = file_labels[j][0]
            temp_image = Image.open(image_path).convert('L')
            temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
            temp_label = np.zeros([FLAGS.char_count])
            label = int(file_labels[j][1])
            temp_label[label] = 1
            labels_batch.append(temp_label)
            images_batch.append(np.asarray(temp_image) / 255.0)
            labels_max_batch.append(label)
            file_paths.append(image_path)

    images_batch = np.array(images_batch).reshape([-1, 64, 64, 1])
    labels_batch = np.array(labels_batch)

    images = tf.placeholder(dtype=tf.float32, shape=[val_batch_size, 64, 64, 1])
    labels = tf.placeholder(dtype=tf.int32, shape=[val_batch_size, FLAGS.char_count])

    endpoints = network(images, labels)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)

    batch_predict_val, batch_predict_index = sess.run([endpoints['predict_val_top3'],
                                                       endpoints['predict_index_top3']],
                                                      feed_dict={images: images_batch, labels: labels_batch})

    final_predict_val += batch_predict_val.tolist()
    final_predict_index += batch_predict_index.tolist()
    groundtruth += labels_max_batch

    sess.close()
    return final_predict_val, final_predict_index, groundtruth,file_paths

#単一の画像を処理
def inference(image):
    sess = tf.Session()
    temp_image = Image.open(image).convert('L')
    temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
    temp_image = np.asarray(temp_image,dtype=np.float32) / 255.0
    temp_image = temp_image.reshape([-1, 64, 64, 1])
    # temp_image = tf.image.convert_image_dtype(tf.image.decode_png(temp_image, channels=1), tf.float32)

    images = tf.placeholder(dtype=tf.float32, shape=[1, 64, 64, 1])
    endpoints = network(images)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)

        predict_val, predict_index = sess.run([endpoints['predict_val_top3'],
                                                       endpoints['predict_index_top3']],
                                                      feed_dict={images: temp_image})
    sess.close()
    return predict_val, predict_index

# TRAINING
def train():
    try:
        tf.Session()
    except Exception:
        pass

    sess = tf.Session()
    file_labels = get_imagesfile(FLAGS.train_data_dir)
    images, labels, coord, threads = batch_data(file_labels, sess)
    endpoints = network(images, labels)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    start_step = 0

    if FLAGS.restore:
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))

            start_step += int(ckpt.split('-')[-1])

    try:
        while not coord.should_stop():

            start_time = time.time()
            _, loss_val, train_summary, step = sess.run(
                [endpoints['train_op'], endpoints['loss'], endpoints['merged_summary_op'], endpoints['global_step']])
            end_time = time.time()
            print("the step {0} takes {1} loss {2}".format(step, end_time - start_time, loss_val))
            if step > FLAGS.max_steps:
                break
            if step % FLAGS.eval_steps == 1:
                accuracy_val, test_summary, step = sess.run(
                    [endpoints['accuracy'], endpoints['merged_summary_op'], endpoints['global_step']])
                # test_writer.add_summary(test_summary, step)
                print('===============Eval a batch in Train data=======================')
                print('the step {0} accuracy {1}'.format(step, accuracy_val))
                print('===============Eval a batch in Train data=======================')
            if step % FLAGS.save_steps == 1:
                print('Save the ckpt of {0}'.format(step))
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=endpoints['global_step'])
    except tf.errors.OutOfRangeError:
        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=endpoints['global_step'])
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()

def eval_metric(final_predict_index, groundtruth,file_paths):
    assert len(final_predict_index) == len(
        groundtruth), 'final_predict_index, size {0} and groundtruth, size {1} must have the same length'.format(
        len(final_predict_index), len(groundtruth))
    accuracy_cnt = 0
    top3_cnt = 0
    for index, val in enumerate(groundtruth):
        print('val {0}, final_predict_index {1}'.format(val, final_predict_index[index]))
        # HTML
        # print("<tr><td><img src='{0}'></td><td>{1}</td><td>{2}</td><td>{3}</td><td>{4}</td></tr>".format(file_paths[index],dic_kannji[val],dic_kannji[final_predict_index[index][0]],dic_kannji[final_predict_index[index][1]],dic_kannji[final_predict_index[index][2]]))

        lagrest_predict = final_predict_index[index][0]
        if val == lagrest_predict:
            accuracy_cnt += 1
        if val in final_predict_index[index]:
            top3_cnt += 1
    print('eval on test dataset size: {0}'.format(len(groundtruth)))
    print('The accuracy {0}, the top3 accuracy {1}'.format(accuracy_cnt * 1.0 / len(groundtruth),
                                                                 top3_cnt * 1.0 / len(groundtruth)))

def run():
    print(FLAGS.mode)

    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == 'validation':
        final_predict_val, final_predict_index, groundtruth,file_paths = validation()
        result = {}
        result['final_predict_val'] = final_predict_val
        result['final_predict_index'] = final_predict_index
        result['groundtruth'] = groundtruth
        result_file = 'result.dict'

        import pickle
        f = open(result_file, 'wb')
        pickle.dump(result, f)
        f.close()

        eval_metric(final_predict_index, groundtruth,file_paths)
    elif FLAGS.mode == 'inference':
        print('inference')
        # image_file = 'C:\\Users\\lele.chen\\Downloads\\Sample\\Sample\\001\\5.png'
        image_file ='C:\\data\\kata1\\test\\2\\ETL1C_4234_1001_1.png'
        final_predict_val, final_predict_index = inference(image_file)
        print('the result info label {0} predict index {1} predict_val {2}'.format(3, final_predict_index,final_predict_val))
        import char_dic as CHAR_DIC
        print(final_predict_index[0][0])

if __name__ == '__main__':
    print("start time:"+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    run()
    print("end time:"+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))