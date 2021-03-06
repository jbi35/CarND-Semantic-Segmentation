import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time
import numpy as np


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
    # load the vgg model and weights
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # get graph
    default_graph = tf.get_default_graph()

    # get layers from graph
    vgg_input = default_graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob  = default_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out = default_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out = default_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out = default_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # add 1x1 convolutional layer
    l7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding="same",
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                kernel_initializer =tf.truncated_normal_initializer(stddev=0.01))

     # add first deconvolution layer
    deconv1 = tf.layers.conv2d_transpose(l7_conv_1x1, num_classes, 4, 2, padding="same",
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                        kernel_initializer =tf.truncated_normal_initializer(stddev=0.01))

     # 1x1 convolution of layer 4
    l4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding="same",
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                kernel_initializer =tf.truncated_normal_initializer(stddev=0.01))

    # add skip connection between layer 4 and first deconvolutional layer
    skip1 = tf.add(l4_conv_1x1, deconv1)

    # second deconvolutional layer
    deconv2 = tf.layers.conv2d_transpose(skip1, num_classes, 4, 2, padding="same",
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                        kernel_initializer =tf.truncated_normal_initializer(stddev=0.01))

    # 1x1 convolution of layer 3
    l3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding="same",
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                kernel_initializer =tf.truncated_normal_initializer(stddev=0.01))

    # add skip connection with layer 3 and second deconvolutional layer
    skip2 = tf.add(l3_conv_1x1, deconv2)

      # add deconvolution back to original size
    output_layer = tf.layers.conv2d_transpose(skip2, num_classes, 16, 8,padding="same",
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                        kernel_initializer =tf.truncated_normal_initializer(stddev=0.01))

    return output_layer

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
    # reshape last layer
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    # loss op
    loss_op = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
    cross_entropy_loss = tf.reduce_mean(loss_op)

    # training op
    # use adam optimizer for simplicity
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param input_image: TF Placeholder for input images
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # initialize
    init = tf.global_variables_initializer()
    sess.run(init)

    # record training time
    t_start = int(time.time())

    # keep prob for dropout
    k_prob = 0.5
    # learning rate
    l_rate = 0.001

    # start training
    print("Starting training...")
    for i in range(epochs):
        training_loss = 0.0
        for batch_x, batch_y in get_batches_fn(batch_size):
            my_feed_dict = {input_image: batch_x,
                            correct_label: batch_y,
                            keep_prob: k_prob,
                            learning_rate: l_rate}

            _ , loss = sess.run((train_op,cross_entropy_loss), feed_dict=my_feed_dict)
            training_loss += (loss * len(batch_x))

        elapsed_time = time.time()
        # print some meaningful information
        if i == 0:
            print("EPOCH  Training time   Training Loss ")
        print("{:3d}   {:.3f}         {:.6f} ".format(i+1, elapsed_time - t_start, training_loss))

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    epochs = 50
    batch_size = 10

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # load vgg
        vgg_input, vgg_keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = load_vgg(sess, vgg_path)

        # define network vgg layers
        network = layers(vgg_layer3, vgg_layer4, vgg_layer7, num_classes)

        # set labels placeholder
        correct_label = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], num_classes])

        # define learning rate placeholder
        learning_rate = tf.placeholder(tf.float32, None)

        # set tensorflow operations for training and loss
        logits, training_op, cross_entropy_loss = optimize(network, correct_label, learning_rate, num_classes)

        # train network
        train_nn(sess, epochs, batch_size, get_batches_fn, training_op, cross_entropy_loss, vgg_input,
                 correct_label, vgg_keep_prob, learning_rate)

        # save test samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)


if __name__ == '__main__':
    run()
