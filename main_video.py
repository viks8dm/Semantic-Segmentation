##################################################################
# import necessary modules
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import scipy
import numpy as np

# use only with video
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import *

##################################################################
# define parameters for segmentation
L2_REG = 1e-3
STD_DEV = 1e-3
KEEP_PROB = 0.5
LEARN_RATE = 1e-4
EPOCHS = 5
BATCH_SIZE = 8
NUM_CLASSES = 2

##################################################################
# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

##################################################################
# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

##################################################################
# Load VGG Model
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # addition
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return w1, keep, layer3, layer4, layer7

print("Testing Load VGG Model ... ... ...")
tests.test_load_vgg(load_vgg, tf)
print("... ... DONE")

##################################################################
# Test Layers
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # function for creating the architecture - decoder
    # use 1x1 convolution to preserve spatial information
    conv_1x1_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                kernel_initializer=tf.random_normal_initializer(stddev=STD_DEV), kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    conv_1x1_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                      kernel_initializer=tf.random_normal_initializer(stddev=STD_DEV), kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    conv_1x1_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                      kernel_initializer=tf.random_normal_initializer(stddev=STD_DEV), kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))

    # perform deconvolution / unsample by 2
    output = tf.layers.conv2d_transpose(conv_1x1_layer7, num_classes, 4, 2, padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    # add skip connection and upsample by 2
    output = tf.add(output, conv_1x1_layer4)
    output = tf.layers.conv2d_transpose(output, num_classes, 4, 2, padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    # add skip connection and upsample by 8
    output = tf.add(output, conv_1x1_layer3)
    output = tf.layers.conv2d_transpose(output, num_classes, 16, 8, padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))

    return output

print("Testing Layers ... ... ...")
tests.test_layers(layers)
print("... ... DONE")


##################################################################
# define optimization function
def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    # FCN-8 section on classification loss
    # image flattening

    # define logits and labels
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    # copute cross-entropy-loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    # train model using adam-optimizer
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

print("TEsting Optimization function ... ... ...")
tests.test_optimize(optimize)
print("... ... DONE")


##################################################################
# train model
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    # returns image and label pair based on batch size
    for epoch in range(epochs):
        for i, (image, label) in enumerate(get_batches_fn(batch_size)):
            # Training

            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label,
                                          keep_prob: KEEP_PROB, learning_rate: LEARN_RATE})

            print("epoch: {}, batch: {}, loss: {}".format(epoch + 1, i, loss))

#print("Testing Train model function ... ... ...")
#tests.test_train_nn(train_nn)
#print("... ... DONE")


##################################################################
def run():
    num_classes = NUM_CLASSES
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    #################
    tf.reset_default_graph()
    model_file = './runs/sem_seg_model.ckpt'
    #################

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        labels = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))
        learning_rate = tf.placeholder(tf.float32)
        logits, train_op, cross_entropy_loss = optimize(layer_output, labels, learning_rate, num_classes)
        
        # to save the trained model
        # saver = tf.train.Saver()

        # # TODO: Train NN using the train_nn function
        # sess.run(tf.global_variables_initializer())
        # train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss,
        #          input_image, labels, keep_prob, learning_rate)
        #
        # # TODO: Save inference data using helper.save_inference_samples
        # helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video
        saver = tf.train.Saver()
        saver.restore(sess, model_file)

        # function for video frame processing
        def process_image(img):
            # process image per original function requirements
            # resize image
            vid_img_shape = img.shape[0:2]
            vid_image = scipy.misc.imresize(img, image_shape)
            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 0.5, input_image: [vid_image]})

            im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            mask = scipy.misc.imresize(mask, vid_img_shape)
            mask = scipy.misc.toimage(mask, mode="RGBA")
            new_image = scipy.misc.toimage(img)
            new_image.paste(mask, box=None, mask=mask)
            return np.array(new_image)




        # video_file = "./data/videos/challenge.mp4"
        # video_out = "./runs/challenge_segmented.mp4"
        video_file = "./data/videos/solidYellowLeft.mp4"
        video_out = "./runs/solidYellowLeft_segmented.mp4"
        clip = VideoFileClip(video_file)
        segmented_video = clip.fl_image(process_image)
        segmented_video.write_videofile(video_out, audio=False)


        # save model
        # saver.save(sess, './runs/sem_seg_model.ckpt')


if __name__ == '__main__':
    # pass
    run()