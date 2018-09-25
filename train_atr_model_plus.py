import os
import cv2
import time
import yaml
import numpy as np
import tensorflow as tf 

# from atr.utils.input_data import get_batch_data
from atr.utils.input_data_from_txt import Dataset
from atr.network.res import Res
from atr.network.layers import bilstm, attention_based_decoder 
from atr.utils.label_map import LabelMap


flags = tf.app.flags
flags.DEFINE_string('exp_dir', '/home/zhui/project/atr/experiments/huawei_en_txt', 
        'experiment model save directory')
FLAGS = flags.FLAGS


def main(_):
    # Loading config
    config_yaml = os.path.join(FLAGS.exp_dir, "config.yaml")
    print(config_yaml)
    assert os.path.exists(config_yaml), "Config yaml file is not exists!!"
    with open(config_yaml, "r") as f:
        config = yaml.load(f)
    with open(config["lexicon_file"], "r") as f:
        character_set = [x.strip("\n") for x in f.readlines()]

    # IO pipeline
    dataset = Dataset(
            config["train_tags_file"],
            config["cache_file"],
            config["train_batch_size"])
    dataset_iterator = dataset.data_generator()

    # Build network
    is_training = True
    label_map_obj = LabelMap(character_set)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    with tf.name_scope("Input"):
        image_placeholder = tf.placeholder(shape=[None, 32, None, 3], dtype=tf.float32)
        groundtruth_text_placeholder = tf.placeholder(shape=[None, ], dtype=tf.string)
        tf.summary.image("InputImage", image_placeholder, 2)
    resnet = Res(istrain=is_training)
    x = resnet(image_placeholder)
    x = tf.squeeze(x, axis=1)
    encoder_output, _ = bilstm("Encoder", x, hidden_units=config["encoder_lstm_hidden_units"])
    train_output, eval_output = attention_based_decoder(
        encoder_output,
        groundtruth_text_placeholder,
        label_map_obj,
        maximum_iterations=200)
    loss_tensor = train_output["loss"]
    tf.summary.scalar("loss", loss_tensor)
    train_text = train_output["predict_text"]
    eval_text = eval_output["predict_text"]

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdadeltaOptimizer(learning_rate=config["learning_rate"]).minimize(
                loss_tensor, global_step)

    # Saver
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if "moving_mean" in g.name]
    bn_moving_vars += [g for g in g_list if "moving_variance" in g.name]
    var_list += bn_moving_vars
    var_list += [global_step]
    train_saver = tf.train.Saver(var_list=var_list)

    sess = tf.Session()
    summary_op = tf.summary.merge_all()    
    train_log_writer = tf.summary.FileWriter(
            os.path.join(config["log_dir"], "train"),
            sess.graph)

    sess.run([
        tf.global_variables_initializer(),
        tf.local_variables_initializer(),
        tf.tables_initializer()
    ])  # run init

    ## Restore weights from ckpt file
    begin_iter = 0
    ckpt_dir = os.path.join(config["log_dir"], "model.ckpt")
    if os.path.exists(os.path.join(config["log_dir"], "checkpoint")):
        latest_ckpt_file = tf.train.latest_checkpoint(config["log_dir"])
        train_saver.restore(sess, save_path=latest_ckpt_file)
        begin_iter = sess.run(global_step)
        print("Loading weights from {} finished, start_iter: {}".format(
            latest_ckpt_file, sess.run(global_step)))
    
    # Training progress
    print("Start training")
    step_change_to_middle_data = 30 * len(dataset) / config["train_batch_size"]  # After 30 epoch, change to middle text data
    step_change_to_long_data = 40 * len(dataset) / config["train_batch_size"]  # After 40 epoch, change to long text data
    step_change_to_random_data = 60 * len(dataset) / config["train_batch_size"]  # After 50 epoch, range select data batch

    for step in range(begin_iter, config["end_iter"]):
        if step < step_change_to_middle_data:
            images, groundtruth_text = next(dataset_iterator)
        elif step < step_change_to_long_data:
            images, groundtruth_text = dataset.get_middle_batch()
        elif step < step_change_to_random_data:
            images, groundtruth_text = dataset.get_long_batch()
        else:
            images, groundtruth_text = dataset.random_get_batch()

        train_feed_dict = {
            image_placeholder: images,
            groundtruth_text_placeholder: groundtruth_text
        }
        _, summary = sess.run([train_op, summary_op], feed_dict=train_feed_dict)
        train_log_writer.add_summary(summary, step)

        if step % 100 == 0:
            loss_ = sess.run(loss_tensor, train_feed_dict) 
            train_text_ = sess.run(train_text, train_feed_dict)
            print("Step {}, loss {}".format(step, loss_))
            print("gts: ", groundtruth_text[:5])
            print("train_texts: ", train_text_[:5])
            print("Eval text: ", sess.run(eval_text, feed_dict={
                image_placeholder: images})[:5])
            print()

        if step % config["ckpt_freq"] == 0:
            train_saver.save(sess, 
                             save_path=ckpt_dir,
                             global_step=global_step)
            print("Saving ckpt file, step {}".format(step))
    sess.close()


if __name__ == "__main__":
    tf.app.run()
