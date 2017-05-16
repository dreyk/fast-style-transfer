from __future__ import print_function
import sys, os, pdb
sys.path.insert(0, 'src')
import numpy as np, scipy.misc
from optimize import optimize
from argparse import ArgumentParser
from utils import save_img, get_img, exists, list_files
import tensorflow as tf
import evaluate

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2

LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_ITERATIONS = 2000
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = 'data/train2014'
BATCH_SIZE = 4
DEVICE = '/gpu:0'
FRAC_GPU = 1


PS_HOSTS = 'localhost:2222'
WORKER_HOSTS = 'localhost:2222'
JOB_NAME = 'worker'
TASK_INDEX = 0

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--train_dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        metavar='CHECKPOINT_DIR', required=True)

    parser.add_argument('--style', type=str,
                        dest='style', help='style image path',
                        metavar='STYLE', required=True)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--test', type=str,
                        dest='test', help='test image path',
                        metavar='TEST', default=False)

    parser.add_argument('--test-dir', type=str,
                        dest='test_dir', help='test image save dir',
                        metavar='TEST_DIR', default=False)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS',
                        default=CHECKPOINT_ITERATIONS)

    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)

    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--ps_hosts', type=str,
                        dest='ps_hosts',
                        help='comma-separated list of hostname:port pairs',
                        metavar='PS_HOSTS', default='localhost:2222')
    parser.add_argument('--worker_hosts', type=str,
                        dest='worker_hosts',
                        help='comma-separated list of hostname:port pairs',
                        metavar='WORKER_HOSTS', default='localhost:3333')
    parser.add_argument('--job_name', type=str,
                        dest='job_name',
                        help='job name: worker or ps',
                        metavar='JOB_NAME', default='worker')
    parser.add_argument('--task_index', type=int,
                        dest='task_index',
                        help="Worker task index, should be >= 0. task_index=0 is "
                        "the master worker task the performs the variable "
                        "initialization ",
                        metavar='TASK_INDEX', default=0)
    parser.add_argument('--num_gpus', type=int,
                        dest='num_gpus',
                        help='Total GPU Number',
                        metavar='NUM_GPU', default=0)

    return parser

def check_opts(opts):
    exists(opts.checkpoint_dir, "checkpoint dir not found!")
    exists(opts.style, "style path not found!")
    exists(opts.train_path, "train path not found!")
    if opts.test or opts.test_dir:
        exists(opts.test, "test img not found!")
        if not os.path.exists(opts.test_dir):
            os.makedirs(opts.test_dir)
    exists(opts.vgg_path, "vgg network data not found!")
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.checkpoint_iterations > 0
    assert os.path.exists(opts.vgg_path)
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0
    assert opts.task_index >=0

def _get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]


def main():
    parser = build_parser()
    options = parser.parse_args()

    ps_spec = options.ps_hosts.split(",")
    worker_spec = options.worker_hosts.split(",")

    # Get the number of workers.
    num_workers = len(worker_spec)

    cluster = tf.train.ClusterSpec({
        "ps": ps_spec,
        "worker": worker_spec})
    if options.job_name == "ps":
        print("Start parameter server %d" % (options.task_index))
        server = tf.train.Server(
            cluster, job_name=options.job_name, task_index=options.task_index)
        server.join()
        return

    check_opts(options)

    style_target = get_img(options.style)

    content_targets = _get_files(options.train_path)

    kwargs = {
        "epochs":options.epochs,
        "print_iterations":options.checkpoint_iterations,
        "batch_size":options.batch_size,
        "save_path":options.checkpoint_dir,
        "learning_rate":options.learning_rate
    }

    args = [
        cluster,
        options.task_index,
        options.num_gpus,
        content_targets,
        style_target,
        options.content_weight,
        options.style_weight,
        options.tv_weight,
        options.vgg_path
    ]

    preds, losses, i, epoch = optimize(*args, **kwargs)
    style_loss, content_loss, tv_loss, loss = losses

    print('Epoch %d, Iteration: %d, Loss: %s' % (epoch, i, loss))
    to_print = (style_loss, content_loss, tv_loss)
    print('style: %s, content:%s, tv: %s' % to_print)
    if options.test and (options.task_index == 0):
        assert options.test_dir != False
        preds_path = '%s/%s_%s.png' % (options.test_dir,epoch,i)
        ckpt_dir = os.path.dirname(options.checkpoint_dir)
        evaluate.ffwd_to_img(options.test,preds_path,
                                 options.checkpoint_dir)
    ckpt_dir = options.checkpoint_dir
    cmd_text = 'python evaluate.py --checkpoint %s ...' % ckpt_dir
    print("Training complete. For evaluation:\n    `%s`" % cmd_text)

if __name__ == '__main__':
    main()
