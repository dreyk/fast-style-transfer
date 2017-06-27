import tensorflow as tf
import sys
sys.path.insert(0, 'src')
import transform, numpy as np, vgg, pdb, os

with tf.Session() as sess:
    batch_shape = (1,256,256, 3)
    images= tf.placeholder(tf.float32, shape=batch_shape,name='images')
    preds = transform.net(images/255.0)
    tensor_info_images = tf.saved_model.utils.build_tensor_info(images)
    tensor_info_preds = tf.saved_model.utils.build_tensor_info(preds)
    prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={'images': tensor_info_images},
          outputs={'preds': tensor_info_preds},
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    saver = tf.train.Saver()
    saver.restore(sess,"./models/muse/model.ckpt-8145")
    export_path = './models/muse/1'
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature
        })
    builder.save()
