# -*- coding: utf-8 -*-
"""
following code use the crfasrnn sample code (http://crfasrnn.torr.vision)
"""

import sys
import numpy as np
from PIL import Image as PILImage
import socket
import simplejson
from random import choice
import re
import os
socket.setdefaulttimeout(3)
sys.path.append('/opt/caffe/python/')
import caffe
import re
_FOLDER = './'

def make_deploy_prototxt():
    head_data ='''
name: "deeplab-large-fov-suyog-binary"
input: 'data'
input_dim: 1
input_dim: 3
input_dim: 513
input_dim: 513'''
    base_dir = os.getcwd()
    template_file = open(base_dir + '/test_template.prototxt').readlines()[25:428]

    test_file_path = base_dir + '/deploy.prototxt'
    test_file = open(test_file_path, 'w')
    for line in head_data.split("\n"):
        line = line.strip()
        if not line:
            continue
        test_file.write(line+'\n')

    for line in template_file:
        test_file.write(line)
    test_file.close()



MODEL_FILE = _FOLDER + 'deploy.prototxt'
if not os.path.exists(MODEL_FILE):
    make_deploy_prototxt()

PRETRAINED = _FOLDER + 'pixel_objectness.caffemodel'
net = caffe.Net(MODEL_FILE, PRETRAINED)
net.set_mode_gpu()
net.set_device(1)
_SIZE = 513


def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"


def random_str(size=8):
    ran_seq = ""
    for i in range(size):
        ran_seq += choice(
            "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )
    if size > 1 and (ran_seq.isdigit() or not re.findall("\d", ran_seq)):
        return random_str(size)
    return ran_seq


def getpallete(num_cls):
    # this function is to get the colormap for visualizing the segmentation mask
    n = num_cls
    pallete = [0] * (n * 3)
    for j in xrange(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete


def resizeImage(image):
    width = image.shape[0]
    height = image.shape[1]
    maxDim = max(width, height)
    if maxDim > _SIZE:
        if height > width:
            ratio = float(_SIZE / height)
        else:
            ratio = float(_SIZE / width)
        image = PILImage.fromarray(np.uint8(image))
        image = image.resize((int(height * ratio), int(width * ratio)), resample=PILImage.BILINEAR)
        image = np.array(image)
    return image

def segmentor(inputs):
    input_ = np.zeros((len(inputs),
                       _SIZE, _SIZE, inputs[0].shape[2]),
                      dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        input_[ix] = in_

    # Segment
    caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                        dtype=np.float32)
    for ix, in_ in enumerate(input_):
        caffe_in[ix] = in_.transpose((2, 0, 1))
    tic()
    out = net.forward_all(**{net.inputs[0]: caffe_in})
    toc()
    predictions = out[net.outputs[0]]
    return predictions[0]


def run_pixelobjectness(inputfile, outputfile=''):
    input_image = 255 * caffe.io.load_image(inputfile)
    input_image = resizeImage(input_image)

    image = PILImage.fromarray(np.uint8(input_image))
    image = np.array(image)

    pallete = getpallete(256)

    mean_vec = np.array([104.008, 116.669, 122.675], dtype=np.float32)
    reshaped_mean_vec = mean_vec.reshape(1, 1, 3);

    # Rearrange channels to form BGR
    im = image[:, :, ::-1]
    # Subtract mean
    im = im - reshaped_mean_vec

    # Pad as necessary
    cur_h, cur_w, cur_c = im.shape
    pad_h = _SIZE - cur_h
    pad_w = _SIZE - cur_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    # Get predictions
    segmentation = segmentor([im])

    segmentation2 = segmentation[0:cur_h, 0:cur_w]
    if outputfile:
        output_im = PILImage.fromarray(segmentation2.argmax(axis=0).astype(np.uint8))
        output_im.putpalette(pallete)
        output_im.convert('RGB').save(outputfile)

    return simplejson.dumps(segmentation2.tolist())

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print "python predict.py inputimage outputimage"
        sys.exit(1)
    input_img = sys.argv[1]
    output_img = sys.argv[2]
    run_pixelobjectness(inputfile=input_img, outputfile=output_img)