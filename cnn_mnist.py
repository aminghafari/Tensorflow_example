from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, y, mode):
    # input layer
    input_layer = tf.reshape(features, [-1, 28,28, 1])
    
    # Conv 1
    conv1 = tf.layers.conv2d(inputs = input_layer, filters = 32, kernel_size = [5, 5], padding = "same", activation = tf.nn.relu)
    
    # Pooling 1
    pool1 = tf.layers.max_pooling2d( inputs = conv1, pool_size = [2, 2], strides = 2)
    
    # Conv2
    conv2 = tf.layers.conv2d(inputs = pool1, filters = 64, kernel_size = [5, 5], padding = "same", activation = tf.nn.relu)
    
    # Pooling 2
    pool2 = tf.layers.max_pooling2d( inputs = conv2, pool_size = [2, 2], strides = 2)
    
    # Dens LayersF
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    dens1 = tf.layers.dense( inputs = pool2_flat, units = 1024, activation = tf.nn.relu)
    
    dropout1 = tf.layers.dropout( inputs = dens1, rate = 0.4, training = mode ==learn.ModeKeys.TRAIN)
    
    # Logits
    dens2 = tf.layers.dense( inputs = dropout1, units = 10)
    
    
    loss = None
    train_op = None
    # Loss
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot( indices = tf.cast( y, tf.int32), depth = 10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits = dens2)
        
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(loss = loss, global_step=tf.contrib.framework.get_global_step(), learning_rate=0.001, optimizer="SGD")
        
        
    # Prediction
    predictions = {
      "classes": tf.argmax(
          input=dens2, axis=1),
      "probabilities": tf.nn.softmax(
          dens2, name="softmax_tensor")}
    
    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):
    mnist = learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.array(mnist.train.labels, dtype = np.int32)
    eval_data = mnist.test.images
    eval_labels = np.array(mnist.test.labels, dtype = np.int32)
    
    
    mnist_classifier = learn.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
    
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    
    
    mnist_classifier.fit(x=train_data, y=train_labels,batch_size=100,steps=20000,monitors=[logging_hook])
    
    metrics = {"accuracy":learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes"),}
    
    eval_results = mnist_classifier.evaluate(x=eval_data, y=eval_labels, metrics=metrics)
    print(eval_results)
    
    

if __name__ == "__main__":
  tf.app.run()
