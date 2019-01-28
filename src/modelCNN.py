import tensorflow as tf
import numpy as np
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"



class Model():
    def __init__(self, image, is_training=True):
        """

        :param image:
        """
        self.image = image
        self.is_training = is_training

    def faceNet(self):
        """
        image entrée: [None, 48, 48, 1]
        :return:
        """
        conv1 = self.conv2dBn1(self.image, 5, 1, 64, pad='SAME')  #[None,96,96,24]
        conv1 = self.maxPooling(conv1, 2)  #[None, 48,48,24]
        conv2 = self.conv2dBn1(conv1, 5, 64, 128, pad='SAME')  #[None, 46,46,36]
        conv2 = self.maxPooling(conv2, 2)  # [None, 22,22,36]
        conv3 = self.conv2dBn1(conv2, 5, 128, 256, pad='SAME')  #[None,
        conv3 = self.maxPooling(conv3, 2)
        flatten = self.flatten(conv3, 9216)
        fc0 = self.fullyConnected1(flatten, 9216, 4096)
        fc1 = self.fullyConnected1(fc0, 4096, 1024)
        fc2 = self.fullyConnected1(fc1, 1024, 256)
        fc3 = self.fullyConnected1(fc2, 256, 7)
        return fc3

    def leNet5(self):
        """

        :param image: [None,32,32,3]
        :return:
        """
        # Layer 1
        conv1 = self.conv2dBn1(self.image, 5, 3, 6)
        conv1 = self.maxPooling(conv1, 2)
        # Layer 2
        conv2 = self.conv2dBn1(conv1, 5, 6, 16)
        conv2 = self.maxPooling(conv2, 2)
        # flatten
        flatten = self.flatten(conv2, 400)
        # Layer 3
        fc1 = self.fullyConnected1(flatten, 400, 120)
        fc2 = self.fullyConnected1(fc1, 120, 84)
        fc3 = self.fullyConnected1(fc2, 84, 10)
        return fc3

    def conv2dBn1(self, input, sizeFiltre, numChannelPre, numChannelCur, strides=1, pad='VALID'):
        """
        Co the Initialize cho cac variable in BN khi muon transfer learning
        :param input:
        :param sizeFiltre:
        :param numChannelPre:
        :param numChannelCur:
        :param strides:
        :param pad:
        :return:
        """
        shape = [sizeFiltre, sizeFiltre, numChannelPre, numChannelCur]
        # Layer with BN
        w = self.initWeight(shape)
        xNorm = tf.layers.batch_normalization(input, training=self.is_training)
        z = tf.nn.conv2d(xNorm, w, strides=[1, strides, strides, 1], padding=pad)  # tich chap convolution
        output = self.relu(z)  # Non linear activation function
        return output

    def conv2dBn2(self, input, sizeFiltre, numChannelPre, numChannelCur, strides=1, pad='VALID'):
        """
        pad='SAME': adding zeros to have the same shape
            'VALID': no-adding

        (nxn) convolved with (fxf) filter/kernel and padding p, stride s,
        it give us ((n+2p-f)/s + 1,(n+2p-f)/s + 1) matrix

        :param input: image entrée
        :param sizeFiltre:
        :param numChannelPre: number of filtres of the layer precedent (or channel of image)
        :param numChannelCur: number of filtres for this layer
        :param strides: stride in conv
        :param pad: 'SAME' or 'VALID' in conv
        :return:
        """
        shape = [sizeFiltre, sizeFiltre, numChannelPre, numChannelCur]
        # Layer with BN
        w = self.initWeight(shape)
        ############## Batch normalisation ##############
        # Cach 1:
        # mean, var = tf.nn.moments(input, [0])  # calculate mean and var of weights
        # epsilon = tf.constant(1e-3)
        # beta = tf.Variable(tf.zeros_like(var))
        # gamma = tf.Variable(tf.ones_like(var))
        # xNorm = tf.nn.batch_normalization(input, mean, var, beta, gamma, epsilon)  # TODO is_training with this case
        # Cach 2:
        # xNorm = tf.contrib.layers.batch_norm(input, center=True, scale=True, is_training=self.is_training)
        # Cach 3:
        # xNorm = (input - mean)/tf.sqrt(var + epsilon)
        # Scale and shift to obtain the final output of the batch normalization
        # this value is fed into the activation function (here a sigmoid)
        # xHat = tf.matmul(xNorm, gamma) + beta
        ############## ################### ##############
        # xNorm = self.batch_norm_wrapper(input, is_training)
        gamma = tf.Variable(tf.ones([numChannelPre]), trainable=True)
        beta = tf.Variable(tf.zeros([numChannelPre]), trainable=True)

        popMean = tf.Variable(tf.zeros([numChannelPre]), trainable=False)  # TODO Khong dung nhu trong ly thuyet Phai la
        popVariance = tf.Variable(tf.ones([numChannelPre]), trainable=False)  # TODO trung tinh cua mean, variance trong tap data training

        epsilon = tf.constant(1e-3)
        def BNTraining():
            decay = tf.constant(0.9)
            movingMean, movingVariance = tf.nn.moments(input, [0, 1, 2], keep_dims=False)
            trainMean = tf.assign(popMean, popMean * decay + movingMean * (1 - decay))
            trainVariance = tf.assign(popVariance, popVariance * decay + movingVariance * (1 - decay))

            with tf.control_dependencies([trainMean, trainVariance]):
                return tf.nn.batch_normalization(input, movingMean, movingVariance, beta, gamma, epsilon)

        def BNTesting():
            return tf.nn.batch_normalization(input, popMean, popVariance, beta, gamma, epsilon)
        is_training = tf.constant(self.is_training, dtype=tf.bool)
        xNorm = tf.cond(is_training, BNTraining, BNTesting)
        z = tf.nn.conv2d(xNorm, w, strides=[1, strides, strides, 1], padding=pad)  # tich chap convolution

        output = self.relu(z)  # Non linear activation function

        return output

    def maxPooling(self, input, ksize, pad='VALID'):
        """

        :param input:
        :param ksize:
        :param pad:
        :return:
        """
        output = tf.nn.max_pool(input, ksize=[1, ksize, ksize, 1], strides=[1, ksize, ksize, 1], padding=pad)
        return output

    def fullyConnected1(self, input, numNeuronsPre, numNeuronsCur):
        shape = [numNeuronsPre, numNeuronsCur]
        w = self.initWeight(shape)
        output = tf.matmul(input, w)
        xNorm = tf.layers.batch_normalization(output, training=self.is_training)
        output = self.relu(xNorm)  # Non linear activation function
        return output


    def fullyConnected2(self, input, numNeuronsPre, numNeuronsCur):
        """

        :param input:
        :param numNeurons:
        :return:
        """
        # Etape 1: fully connected ###
        # Cach 1
        shape = [numNeuronsPre, numNeuronsCur]
        w = self.initWeight(shape)
        output = tf.matmul(input, w)
        # Cach 2
        # output = tf.contrib.layers.fully_connected(input, numNeuronsCur, activation_fn=None)
        # Cach 3
        # output = tf.layers.dense(input, numNeuronsCur, use_bias=False, activation=None)

        # Etape 2: Add batch normalisation ###
        gamma = tf.Variable(tf.ones([numNeuronsCur]))
        beta = tf.Variable(tf.zeros([numNeuronsCur]))
        popMean = tf.Variable(tf.zeros([numNeuronsCur]), trainable=False)
        popVariance = tf.Variable(tf.ones([numNeuronsCur]), trainable=False)
        epsilon = tf.constant(1e-3)

        def BNTraining():
            decay = tf.constant(0.9)
            batchMean, batchVariance = tf.nn.moments(output, [0])
            trainMean = tf.assign(popMean, popMean * decay + batchMean * (1 - decay))
            trainVariance = tf.assign(popVariance, popVariance * decay + batchVariance * (1 - decay))
            with tf.control_dependencies([trainMean, trainVariance]):
                return tf.nn.batch_normalization(output, batchMean, batchVariance, beta, gamma, epsilon)

        def BNTesting():
            return tf.nn.batch_normalization(output, popMean, popVariance, beta, gamma, epsilon)
        is_training = tf.constant(self.is_training, dtype=tf.bool)
        output = tf.cond(is_training, BNTraining, BNTesting)

        # Etape 3: Activation function
        output = self.relu(output)
        return output

    def flatten(self, input, numNeurons):
        """

        :param input:
        :param numNeurons: calculated from precedent layer
        :return:
        """
        output = tf.reshape(input, shape=[-1, numNeurons])
        return output

    def relu(self, input):
        output = tf.nn.relu(input)
        return output

    def sofmax(self, input):
        output = tf.nn.softmax(input)
        return output

    @staticmethod
    def initWeight(shape):
        """
        initialize the value of w
        :return:
        """
        # Works OK for small networks but it going to zero in deeper networks.
        # and the gradient will vanish sooner in deep networks.
        # w = 0.01 * np.random.rand(shape=shape)
        # # Works OK for small networks but The network will explode with big numbers
        # with deeper networks
        # w = 1 * np.random.rand(shap=shape)
        # Xavier initialization, It breaks when you are using RELU.
        # He initialization (Solution for the RELU issue)
        w = tf.Variable(tf.truncated_normal(shape=shape, mean=0.0, stddev=0.01), dtype=np.float32)
        tf.summary.histogram("Weights", w)
        return w

    @staticmethod
    def initBias(shape):
        """
        initialize the value of b
        :return:
        """
        b = tf.Variable(tf.constant(0.1, shape=shape), dtype=np.float32)
        tf.summary.histogram('Bias', b)
        return b

    def trainModel(self):
        """

        :return:
        """

    def inferenceModel(self):
        """

        :return:
        """

    def saveModel(self):
        """

        :return:
        """

    def loadModel(self):
        """

        :return:
        """

    def __str__(self):
        #use print to show this message when call this class
        return "This is the neural network that we will use to train our data"

if __name__ == '__main__':
    test = 0
    if test: 
        image = tf.random_normal(shape=[5, 48, 48, 1])
        #image = tf.TensorArray(image)
        a = Model(image, is_training=False)
        b = tf.shape(a.faceNet())
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # print(sess.run(b))
        print(sess.run(b))
        sess.close()