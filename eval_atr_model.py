import cv2
import tensorflow as tf 
from model_aon import get_init_op, inference
import os
from res import Res

flags = tf.app.flags
flags.DEFINE_string('exp_dir', './model/test_huawei', '')
flags.DEFINE_string('image_path', './168_supremacy_76394.jpg', '')
FLAGS = flags.FLAGS


def load_image(image_path):
  image = cv2.imread(image_path, 1)
  image = cv2.resize(
    image, 
    (100, 32),  # new_width, new_height
  )
  image = image[:, :, ::-1]  # From BGR to RGB format
  print(image.shape)
  image = image / 128.0 - 1
  return image


def main(_):
    global_step = tf.Variable(0, name="global_step", trainable=False)
    image_placeholder = tf.placeholder(shape=[None, 32, 100, 3], dtype=tf.float32)
    ground_truth_placeholder = tf.placeholder(shape=[None, ], dtype=tf.string)

    is_training = False
    resnet = Res(istrain=is_training)
    x = resnet(image_placeholder)
    train_output, eval_output = inference(
            x, ground_truth_placeholder)
    eval_text = eval_output["predict_text"]
    
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if "moving_mean" in g.name]
    bn_moving_vars += [g for g in g_list if "moving_variance" in g.name]
    var_list += bn_moving_vars

    saver = tf.train.Saver(var_list)
    latest_ckpt_file = tf.train.latest_checkpoint(FLAGS.exp_dir)
    with tf.Session() as sess:
        sess.run([
            tf.local_variables_initializer(),
            tf.global_variables_initializer(),
            tf.tables_initializer()
        ])
        saver.restore(sess, latest_ckpt_file)
        print("Loading weights from {} finished, step {}".format(
            latest_ckpt_file, sess.run(global_step)))
        
        demo_image = load_image(FLAGS.image_path).reshape([1, 32, 100, 3])
        pred_ = sess.run(eval_text, feed_dict={
            image_placeholder: demo_image})

        print(pred_)


if __name__ == "__main__":
    tf.app.run()
