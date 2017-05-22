from __future__ import print_function
import sys
sys.path.insert(0, 'src')
import transform, numpy as np, vgg, pdb, os
import scipy.misc
import tensorflow as tf
from utils import save_img, get_img, exists, list_files
from argparse import ArgumentParser
from collections import defaultdict
import time
import json
import subprocess
import numpy

BATCH_SIZE = 1
DEVICE = '/gpu:0'

def main():
    dev = "/dev/video0"

    checkpoints = ["/kubenfs/lmb/training/style20/model.ckpt-6518",
        "/kubenfs/lmb/training/style9/model.ckpt-9147",
        "/kubenfs/lmb/training/muse/model.ckpt-8145",
        "/kubenfs/lmb/training/udnie/model.ckpt-8193"]
    #checkpoints = ["/kubenfs/lmb/training/style9/model.ckpt-5413",
    #    "/kubenfs/lmb/training/style9/model.ckpt-6499",
    #    "/kubenfs/lmb/training/style9/model.ckpt-7475",
    #    "/kubenfs/lmb/training/style9/model.ckpt-8246"]
    currentStyle = 0
    command = ["ffprobe",
               '-v', "quiet",
               '-print_format', 'json',
               '-show_streams', dev]

    info = json.loads(str(subprocess.check_output(command)))
    width = int(info["streams"][0]["width"])
    height = int(info["streams"][0]["height"])
    fps = round(eval(info["streams"][0]["r_frame_rate"]))
    print("%d - %d %s" % (width,height,str(fps)))
    command = ["ffmpeg",
               '-loglevel', "quiet",
               '-i', dev,
                '-vf',"fps=fps=4,scale=1280:1024",#,scale=1280:1024
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-tune','zerolatency',
               '-crf', '18',
               #'-hwaccel_device', '1',
               '-vcodec', 'rawvideo', '-']
    width = 1280
    height = 1024
    #fps = 24
    pipe_in = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=0, stdin=None, stderr=None)

    out = "test.mp4"

    command = ["ffplay",
               '-loglevel', "info",
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-s', str(width) + 'x' + str(height),  # size of one frame
               '-pix_fmt', 'rgb24',
               #'-tune','zerolatency',
               '-an',  # Tells FFMPEG not to expect any audio
               '-']

    pipe_out = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=None, stderr=None)
    #run --rm -it --name vide-demo --device /dev/video0:/dev/video0 -v /kubenfs:/kubenfs tensorflow/tensorflow:latest-gpu /bin/sh -c 'cd /kubenfs/vide-demo;python piper.py'


    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    soft_config.gpu_options.allow_growth = True

    with g.as_default(), g.device(DEVICE), \
         tf.Session(config=soft_config) as sess:
        batch_shape = (BATCH_SIZE, height, width, 3)
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')
        preds = transform.net(img_placeholder)
        #with tf.device("/cpu:0"):
        saver = tf.train.Saver()
        saver.restore(sess,checkpoints[currentStyle])
        X = np.zeros(batch_shape, dtype=np.float32)
        read_input = True
        last = False
        nbytes = 3 * width * height
        total_count = 0
        first = True
        while read_input:
            count = 0
            while count < BATCH_SIZE:
                if first:
                    skip_count = 0
                    while skip_count < 100:
                        raw_image = pipe_in.stdout.read(width * height * 3)
                        skip_count += 1
                    first = False
                raw_image = pipe_in.stdout.read(width * height * 3)
                if len(raw_image) != nbytes:
                    if count == 0:
                        read_input = False
                    else:
                        last = True
                        X = X[:count]
                        batch_shape = (count, height, width, 3)
                        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                                     name='img_placeholder')
                        preds = transform.net(img_placeholder)
                    break

                image = numpy.fromstring(raw_image, dtype='uint8')
                image = image.reshape((height, width, 3))
                X[count] = image
                count += 1

            if read_input:
                if total_count>20:
                    currentStyle += 1
                    total_count = 0
                    if currentStyle >= len(checkpoints):
                        currentStyle = 0
                    saver.restore(sess,checkpoints[currentStyle])
                if last:
                    read_input = False
                _preds = sess.run(preds, feed_dict={img_placeholder: X})

                for i in range(0, batch_shape[0]):
                    total_count += 1
                    img = np.clip(_preds[i], 0, 255).astype(np.uint8)
                    pipe_out.stdin.write(img)


        pipe_out.terminate()
        pipe_in.terminate()
        pipe_out.stdin.close()
        pipe_in.stdout.close()
        del pipe_in
        del pipe_out

if __name__ == '__main__':
    main()
