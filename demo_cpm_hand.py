# For single hand and no body part in the picture
# ======================================================

import tensorflow as tf
import cpm_hand
import numpy as np
from utils import *
import cv2
import os

"""Parameters
"""
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_path',
                           default_value='./models/weights/2018_05_25_10/cpm_hand-15500',
                           #default_value='cpm_hand.pkl',
                           docstring='Your model')
tf.app.flags.DEFINE_string('test_path',
                           default_value='./data/blouse/blouse_test.txt',
                           docstring='test_data')
tf.app.flags.DEFINE_string('test_num',
                           default_value=20,
                           docstring='number of data to test')
tf.app.flags.DEFINE_string('result_path',
                           default_value='./result/blouse',
                           docstring='results saved')
tf.app.flags.DEFINE_integer('input_size',
                            default_value=368,
                            docstring='Input image size')
tf.app.flags.DEFINE_integer('hmap_size',
                            default_value=46,
                            docstring='Output heatmap size')
tf.app.flags.DEFINE_integer('cmap_radius',
                            default_value=21,
                            docstring='Center map gaussian variance')
tf.app.flags.DEFINE_integer('joints',
                            default_value=13,
                            docstring='Number of joints')
tf.app.flags.DEFINE_integer('stages',
                            default_value=3,
                            docstring='How many CPM stages')

# Set color for each finger
joint_color_code = [[139, 53, 255],
                    [0, 56, 255],
                    [43, 140, 237],
                    [37, 168, 36],
                    [147, 147, 0],
                    [70, 17, 145]]

    
def main(argv):

    if not os.path.exists(FLAGS.result_path):
        os.makedirs(FLAGS.result_path)
    
    
    tf_device = '/gpu:0'
    with tf.device(tf_device):
        """Build graph
        """
        input_data=tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.input_size, FLAGS.input_size, 3],
                                  name='input_image')
    
        center_map = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.input_size, FLAGS.input_size, 1],
                                    name='center_map')

        model = cpm_hand.CPM_Model(FLAGS.input_size, FLAGS.hmap_size,FLAGS.stages, FLAGS.joints )

    saver = tf.train.Saver()

    """Create session and restore weights
    """
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    saver.restore(sess, FLAGS.model_path)
    #model.load_weights_from_file(FLAGS.model_path, sess, finetune=True)

    test_center_map = gaussian_img(FLAGS.input_size, FLAGS.input_size, FLAGS.input_size / 2,
                                   FLAGS.input_size / 2,
                                   FLAGS.cmap_radius)
    test_center_map = np.reshape(test_center_map, [1, FLAGS.input_size, FLAGS.input_size, 1])

    # Check weights
    for variable in tf.trainable_variables():
        with tf.variable_scope('', reuse=True):
            var = tf.get_variable(variable.name.split(':0')[0])
            print(variable.name, np.mean(sess.run(var)))


    with tf.device(tf_device):
        test_img = chose_img_test(FLAGS.test_path, FLAGS.test_num)
        for img in test_img:
            img_name=img.strip('\n').split('/')[-1]
            file_path='/data/yuwei/research/Tianchi/Project/data/train_data/' + img
            #file_path = '1.jpg'
            test_img =read_image(file_path, FLAGS.input_size, 'IMAGE')
            test_img_resize = cv2.resize(test_img, (FLAGS.input_size, FLAGS.input_size))
            
            test_img_input = test_img_resize / 256.0 - 0.5
            test_img_input = np.expand_dims(test_img_input, axis=0)


            # Inference
            predict_heatmap, stage_heatmap_np = sess.run([model.current_heatmap,
                                                          model.stage_heatmap,
                                                          ],
                                                         feed_dict={model.input_images: test_img_input,
                                                                    model.cmap_placeholder: test_center_map})

            # Show visualized image
            demo_img, joint_coord_set = visualize_result(test_img, stage_heatmap_np, FLAGS.joints, FLAGS.hmap_size,joint_color_code)
            
            save_path= FLAGS.result_path + '/' + img_name
            cv2.imwrite(save_path, demo_img.astype(np.uint8))
            #cv2.imwrite('1_.jpg', demo_img.astype(np.uint8))
            print(img_name + ' tested:', joint_coord_set)
        print('test done')
           


if __name__ == '__main__':
    tf.app.run()
