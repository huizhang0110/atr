import os
import cv2
import time
import yaml
import numpy as np
import tensorflow as tf

from atr.network.res import Res
from atr.network.layers import bilstm, attention_based_decoder
from atr.utils.label_map import LabelMap
from atr.utils.image_utils import load_image

flags = tf.app.flags
flags.DEFINE_string('exp_dir', '/home/zhui/project/atr/experiments/huawei_en', '')
flags.DEFINE_string('image_path', '/home/zhui/project/atr/data/182_slinking_71711.jpg', '')
FLAGS = flags.FLAGS


def main(_):
    config_yaml = os.path.join(FLAGS.exp_dir, "config.yaml")
    print(config_yaml)
    assert os.path.exists(config_yaml), "Config yaml file is not exists"
    with open(config_yaml, "r") as f:
        config = yaml.load(f)
    with open(config["lexicon_file"], "r") as f:
        character_set = [x.strip("\n") for x in f.readlines()]

    # Building inference network
    is_training = False
    label_map_obj = LabelMap(character_set)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    with tf.name_scope("Input"):
        image_placeholder = tf.placeholder(shape=[None, 32, None, 3], dtype=tf.float32)
        # We don't need to use this, just to build a network graph
        ground_truth_placeholder = tf.placeholder(shape=[None, ], dtype=tf.string)
    resnet = Res(istrain=is_training)
    x = resnet(image_placeholder)
    x = tf.squeeze(x, axis=1)
    encoder_output, _ = bilstm("Encoder", x, hidden_units=256)
    _, eval_output = attention_based_decoder(encoder_output,
                                             ground_truth_placeholder,
                                             label_map_obj,
                                             maximum_iterations=200)
    eval_text = eval_output["predict_text"]

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if "moving_mean" in g.name]
    bn_moving_vars += [g for g in g_list if "moving_variance" in g.name]
    var_list += bn_moving_vars
    var_list += [global_step]

    saver = tf.train.Saver(var_list)
    latest_ckpt_file = tf.train.latest_checkpoint(config["log_dir"])
    with tf.Session() as sess:
        sess.run([
            tf.local_variables_initializer(),
            tf.global_variables_initializer(),
            tf.tables_initializer(),
        ])
        saver.restore(sess, latest_ckpt_file)
        print("Loading weights from {} finished, step {}".format(
            latest_ckpt_file, sess.run(global_step)))

        demo_image = load_image(FLAGS.image_path, dynamic=True).reshape([1, 32, -1, 3])
        pred_ = sess.run(eval_text, feed_dict={
            image_placeholder: demo_image})
        print(pred_)


if __name__ == "__main__":
    tf.app.run()
