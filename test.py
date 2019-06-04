import os
import tensorflow as tf
from net.network import SemanticRegenerationNet, HRSemanticRegenerationNet

from options.test_options import TestOptions
import subprocess
import numpy as np
import cv2
from net.ops import Margin
import time
import glob

os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
        "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]
        ))


def generate_mask(im_shapes, mask_shapes, rand=True):
    mask = np.zeros((im_shapes[0], im_shapes[1])).astype(np.float32)
    if rand:
        of0 = np.random.randint(0, im_shapes[0]-mask_shapes[0])
        of1 = np.random.randint(0, im_shapes[1]-mask_shapes[1])
    else:
        if im_shapes[1] == 512 or im_shapes[1] == 1024:
            of0 = 0
            of1 = (im_shapes[1] - mask_shapes[1]) // 2
        elif im_shapes[1] == 128:
            of0 = 0
            of1 = 0
        else:
            of0 = (im_shapes[0]-mask_shapes[0])//2
            of1 = (im_shapes[1]-mask_shapes[1])//2
    mask[of0:of0+mask_shapes[0], of1:of1+mask_shapes[1]] = 1
    mask = np.expand_dims(mask, axis=2)
    margin = Margin(top=of0, left=of1, bottom=im_shapes[0]-mask_shapes[0]-of0,
                    right=im_shapes[1]-mask_shapes[1]-of1)
    return mask, margin

config = TestOptions().parse()

if os.path.isfile(config.dataset_path):
    pathfile = open(config.dataset_path, 'rt').read().splitlines()
elif os.path.isdir(config.dataset_path):
    pathfile = glob.glob(os.path.join(config.dataset_path, '*.png'))
else:
    print('Invalid testing data file/folder path.')
    exit(1)
total_number = len(pathfile)
test_num = total_number if config.test_num == -1 else min(total_number, config.test_num)
print('The total number of testing images is {}, and we take {} for test.'.format(total_number, test_num))

if config.model == 'srn':
    model = SemanticRegenerationNet()
elif config.model == 'srn-hr':
    model = HRSemanticRegenerationNet()
else:
    print('unknown model types.')
    exit(1)

reuse = False
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = False
with tf.Session(config=sess_config) as sess:
    input_image_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 3])
    mask_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 1])

    top_tf = tf.placeholder(dtype=tf.int32, shape=[])
    left_tf = tf.placeholder(dtype=tf.int32, shape=[])
    bottom_tf = tf.placeholder(dtype=tf.int32, shape=[])
    right_tf = tf.placeholder(dtype=tf.int32, shape=[])
    margin_tf = Margin(top_tf, left_tf, bottom_tf, right_tf)

    output, _ = model.evaluate(input_image_tf, mask_tf, margin_tf, config=config, reuse=reuse)

    output = tf.cast(tf.clip_by_value((output + 1.) * 127.5, 0, 255)[:, :, :, ::-1], tf.uint8)

    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = list(map(lambda x: tf.assign(x, tf.contrib.framework.load_variable(config.load_model_dir, x.name)),
                          vars_list))
    sess.run(assign_ops)
    print('Model loaded.')
    total_time = 0

    if config.random_mask:
        np.random.seed(config.seed)

    for i in range(test_num):
        mask, margin = generate_mask(config.img_shapes, config.mask_shapes, config.random_mask)
        config.margin = margin
        image = cv2.imread(pathfile[i])

        if config.random_crop is False:
            image = cv2.resize(image, (config.img_shapes[1], config.img_shapes[0]))
        else:
            h, w = image.shape[:2]
            if h >= config.img_shapes[0] and w >= config.img_shapes[1]:
                h_start = (h-config.img_shapes[0]) // 2
                w_start = (w-config.img_shapes[1]) // 2
                image = image[h_start: h_start+config.img_shapes[0], w_start: w_start+config.img_shapes[1], :]
            else:
                image = cv2.resize(image, (config.mask_shapes[1], config.mask_shapes[0]))
                h, w = image.shape[:2]
                assert h == config.mask_shapes[0] and w == config.mask_shapes[1]
                image_t = np.zeros(config.img_shapes, dtype=np.uint8)
                h_start = (config.img_shapes[0]-h)//2
                w_start = (config.img_shapes[1]-w)//2
                image_t[h_start: h_start+h, w_start: w_start+w, :] = image
                image = image_t

        image_input = image * mask + 255 * (1-mask)
        cv2.imwrite(os.path.join(config.saving_path, 'input_{:03d}.png'.format(i)), image_input.astype(np.uint8))

        assert image.shape[:2] == mask.shape[:2]

        h, w, _ = image.shape
        grid = 4
        image = image[:h // grid * grid, :w // grid * grid, :]
        mask = mask[:h // grid * grid, :w // grid * grid, :]
        print('{} / {}'.format(i, test_num))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)

        start_t = time.time()
        result = sess.run(output, feed_dict={input_image_tf: image * 1.0, mask_tf: mask,
                                             top_tf: margin.top, left_tf: margin.left,
                                             bottom_tf: margin.bottom, right_tf: margin.right})
        duration_t = time.time() - start_t
        total_time += duration_t
        cv2.imwrite(os.path.join(config.saving_path, '{:03d}.png'.format(i)), result[0][:, :, ::-1])
        if reuse is False:
            reuse = True
    print('total time > {}s, average time > {}s'.format(total_time, total_time/test_num))
