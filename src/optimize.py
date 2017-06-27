from __future__ import print_function
import functools
import vgg, pdb, time
import tensorflow as tf, numpy as np, os
import transform
from utils import get_img

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
DEVICES = 'CUDA_VISIBLE_DEVICES'

# np arr, np arr
def optimize(cluster,task_index,num_gpus,limit,content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1000,
             batch_size=4, save_path='saver',
             learning_rate=1e-3, debug=False):
    if limit >0 :
        print("Limit train set %d" % limit)
        content_targets = content_targets[0:limit]
    mod = len(content_targets) % batch_size
    if mod > 0:
        print("Train set has been trimmed slightly..")
        content_targets = content_targets[:-mod]

    style_features = {}

    batch_shape = (batch_size,256,256,3)
    style_shape = (1,) + style_target.shape
    print(style_shape)
    is_chief = (task_index == 0)
    # precompute style features
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session():
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(vgg_path, style_image_pre)
        style_pre = np.array([style_target])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image:style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    server = tf.train.Server(
            cluster, job_name="worker", task_index=task_index)
    if num_gpus>0:
        worker_device = "/job:worker/task:%d/gpu:0" % (task_index)
    else:
        worker_device = "/job:worker/task:%d/cpu:0" % (task_index)

    time_begin = time.time()
    print("Training begins @ %f" % time_begin)
    with tf.device(
            tf.train.replica_device_setter(
                worker_device=worker_device,
                ps_device="/job:ps/cpu:0",
                cluster=cluster)),tf.Session() as sess:
        global_step = tf.Variable(0, name="global_step", trainable=False)
        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = vgg.preprocess(X_content)

        # precompute content features
        content_features = {}
        content_net = vgg.net(vgg_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        preds = transform.net(X_content/255.0)
        preds_pre = vgg.preprocess(preds)

        net = vgg.net(vgg_path, preds_pre)

        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
        )

        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0,2,1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size

        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size

        loss = content_loss + style_loss + tv_loss

        # overall loss
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

        init_op = tf.global_variables_initializer()

        sv = tf.train.Supervisor(
                is_chief=is_chief,
                logdir=save_path,
                init_op=init_op,
                recovery_wait_secs=1,
                global_step=global_step)

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=["/job:ps", "/job:worker/task:%d" % task_index])

        import random
        uid = random.randint(1, 100)
        print("UID: %s" % uid)
        local_step = 0
        num_examples = len(content_targets)
        iterations = 0
        num_samples = num_examples / batch_size
        num_global =  num_samples * epochs
        print("Number of iterations %d" % num_global)
        step = 0
        with sv.managed_session(server.target, config=sess_config) as sess:
            while step < num_global and not sv.should_stop():
                curr = iterations * batch_size
                step = curr + batch_size
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                for j, img_p in enumerate(content_targets[curr:step]):
                   X_batch[j] = get_img(img_p, (256,256,3)).astype(np.float32)

                iterations += 1
                if iterations == num_samples:
                    iterations = 0
                assert X_batch.shape[0] == batch_size

                feed_dict = {
                   X_content:X_batch
                }
                _, step = sess.run([train_step, global_step], feed_dict=feed_dict)
                local_step += 1
                print("Worker %d: training step %d done (global step: %d)" %
                    (task_index, local_step, step))
            to_get = [style_loss, content_loss, tv_loss, loss, preds]
            test_feed_dict = {
               X_content:X_batch
            }
            tup = sess.run(to_get, feed_dict = test_feed_dict)
            _style_loss,_content_loss,_tv_loss,_loss,_preds = tup
            losses = (_style_loss, _content_loss, _tv_loss, _loss)
            time_end = time.time()
            print("Training ends @ %f" % time_end)
            training_time = time_end - time_begin
            print("Training elapsed time: %f s" % training_time)
        return (_preds, losses, iterations, epochs)

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
