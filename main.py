#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import numpy as np 
import sys 
import cv2
import scipy
import pandas as pd 
import argparse

global epochs_num
global batchsize

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))



def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()
    w_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    w_layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    w_layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    w_layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return w_input, keep_prob, w_layer3, w_layer4, w_layer7

tests.test_load_vgg(load_vgg, tf)

# Set initializer and regularizer
def regularizer(scale):
    return slim.l2_regularizer(scale)

def initializer():
    return tf.truncated_normal_initializer(stddev=0.01)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # 1x1 conv layer
    l3_conv1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, (1,1), (1,1), kernel_initializer=initializer(), kernel_regularizer=regularizer(0.001), name='l3_conv1x1')
    l4_conv1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, (1,1), (1,1), kernel_initializer=initializer(), kernel_regularizer=regularizer(0.001), name='l4_conv1x1')
    l7_conv1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, (1,1), (1,1), kernel_initializer=initializer(), kernel_regularizer=regularizer(0.001), name='l7_conv1x1')
    
    # Upsample
    deconv1 = tf.layers.conv2d_transpose(l7_conv1x1, num_classes, (4,4), (2,2), padding='SAME', kernel_initializer=initializer(), kernel_regularizer=regularizer(0.001), name='deconv1')

    # Connect 1 
    dc1_l4b = tf.add(deconv1, l4_conv1x1, name='dc1_l4b')
    deconv2 = tf.layers.conv2d_transpose(dc1_l4b, num_classes, (4,4), (2,2), padding='SAME', kernel_initializer=initializer(), kernel_regularizer=regularizer(0.001), name='deconv2')

    # Connect 2 
    dc2_l3b = tf.add(deconv2, l3_conv1x1, name='dc2_l3b')
    deconv3 = tf.layers.conv2d_transpose(dc2_l3b, num_classes, (16,16), (8,8), padding='SAME', kernel_initializer=initializer(), kernel_regularizer=regularizer(0.001), name='deconv3')

    return deconv3

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # Get the classification loss
    classification_loss = slim.losses.softmax_cross_entropy(logits, labels)

    # Add the generalization loss to the original loss
    total_loss = slim.losses.get_total_loss(add_regularization_losses=True)

    # Use Adam optimizer for optimization
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # To minimize the total loss rather than classification loss
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    return logits, train_op, total_loss, classification_loss
    
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, total_loss, classification_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param total_loss: TF Tensor for the total amount of loss
    :param classification_loss: TF Tensor for the cross entropy loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    total_loss_plot = []
    classification_loss_plot = []
    samples_plot = []
    sample = 0

    for epoch in tqdm(range(epochs)):
        batch_num = 0
        for img, img_gt in get_batches_fn(batch_size):
            _, closs, tloss = sess.run([train_op, classification_loss, total_loss], feed_dict = {
                input_image: img,
                correct_label: img_gt,
                keep_prob: 0.5,
                learning_rate: 1e-4
            })
            samples_plot.append(sample)
            classification_loss_plot.append(closs)
            total_loss_plot.append(tloss)
            sample += batch_size
            print("Batch%4d (%7d): c_loss(%.2f), t_loss(%.2f)" %(batch_num, sample, closs, tloss))
            batch_num += 1
            
    # Restore the classification loss and total loss to visulize the whole training process
    data_frame = pd.DataFrame(data={'sample':samples_plot, 'closs':classification_loss_plot, 'tloss': total_loss_plot})
    data_frame.to_csv('train_information.csv')
    print('Train information saved.')

 
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        correct_label = tf.placeholder(tf.int32)
        learning_rate = tf.placeholder(tf.float32)
        w_input, keep_prob, w_layer3, w_layer4, w_layer7 = load_vgg(sess, vgg_path)
        layer_output = layers( w_layer3, w_layer4, w_layer7, num_classes)
        logits, train_op, total_loss, classification_loss = optimize(layer_output, correct_label, learning_rate, num_classes)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        train_nn(sess, epoch_num, batchsize, get_batches_fn, train_op, total_loss, classification_loss, w_input, correct_label, keep_prob, learning_rate)
        save_path = saver.save(sess, "./runs/aug_model_E%04d-B%04d.ckpt"%(epoch_num, batchsize))
        print("Model saved in file: %s" % save_path)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, w_input)#, run_label=run_label)



if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_subparsers('--epochs', type=int, default=1, help='epoch number of the training')
    #parser.add_subparsers('--batchsize', type=int,default=1, help='batch size of the training')
    #inf = parser.parse_args()
    #epoch_num = inf.epochs
    #batchsize = inf.batchsize
    if len(sys.argv) != 3:
        print('Please specify the epochs and batch size for training.')
        exit()
    epoch_num = int(sys.argv[1])
    batchsize = int(sys.argv[2])
    run()
