import modelCNN 
import tensorflow as tf
import numpy as np
import os
import time
import pandas as pd 
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from config import *
from sklearn.metrics import confusion_matrix
file_dir = os.path.dirname(os.path.abspath("/home/minh/Documents/Artificial-Intelligence/computer_vision/face_detect_opencv_python/data"))
package_dir_a = os.path.join(file_dir, 'data')
sys.path.append(package_dir_a) #sys.path.insert(0, package_dir_a)

def _main():
    """

    :return:
    """

    with open(package_dir_a + "/fer2013.csv") as f: # don't need to f.close()
        dataframe = pd.read_csv(f)

    emotion1 = np.array(dataframe["emotion"])
    emotion = np.zeros((35887, 7))
    emotion[np.arange(len(dataframe["pixels"])), emotion1] = 1
    emotion = emotion.reshape([35887, 7])
    pixels = [np.reshape(np.fromstring(dataframe["pixels"][i], dtype=int, sep=' ')/255, [48, 48, 1]) for i in range(len(dataframe["pixels"]))]  
    typeUsage = dataframe["Usage"]

    test_size, seed = [0.99, 7] # 0.7 for data set
    # train_test_split: shuffle:(default=True)
    XTrain, XTest, YTrain, YTest = model_selection.train_test_split(pixels, emotion, test_size=test_size, random_state=seed)


    cwd = "/home/minh/Documents/Artificial-Intelligence/computer_vision/face_detect_opencv_python/src"
    pathLogs = cwd + '/logs2'
    # Load training set
    #XTrain, YTrain = loadData()
    numImage = YTrain.shape[0]

    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Start running operations on the Graph.
        # STEP 1: Input data #########
        X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name='Input')  # for image_data
        Y = tf.placeholder(tf.float32, shape=[None, numClass])
        is_training = tf.placeholder(tf.bool)
        # Reshape images for visualization
        tf.summary.image("Input", X)
        # STEP 2: Building the graph #########
        #  Building the graph
        # Generate output tensor targets for filtered bounding boxes.
        YPred = modelCNN.Model(X).faceNet()

        with tf.name_scope("Loss"):
            loss = tf.reduce_mean(tf.pow(YPred-Y, 2))
            tf.summary.scalar("Loss", loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # TODO Batch normalisation to know
        with tf.control_dependencies(update_ops):  # TODO
            # with tf.name_scope("Optimizer"):
            learningRate = 0.1
            optimizer = tf.train.AdamOptimizer(0.01).minimize(loss, global_step=global_step)
        # STEP 3: Build the evaluation step ########
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        with tf.Session(config=config, graph=graph) as sess:
            # Merges all summaries collected in the default graph
            summary_op = tf.summary.merge_all()
            # Summary Writers
            # tensorboard --logdir='./graphs/' --port 6005
            trainSummaryWriter = tf.summary.FileWriter(pathLogs + '/graphs/train', sess.graph)
            validationSummaryWriter = tf.summary.FileWriter(pathLogs + '/graphs/validation', sess.graph)

            # summaryWriter = tf.summary.FileWriter('./graphs', sess.graph)
            sess.run(tf.global_variables_initializer())
            # If you want to continue training from check point
            # checkpoint = "/home/minh/PycharmProjects/yolo3/save_model/SAVER_MODEL_boatM/model.ckpt-" + "1"
            # saver.restore(sess, checkpoint)
            for epoch in range(epochs):
                startTime = time.time()
                ## Training#########
                meanLossTrain = []
                for start in (range(0, numImage, batchSize)):
                    startTrain = time.time()
                    end = start + batchSize
                    summaryTrain, lossTrain, _ = sess.run([summary_op, loss, optimizer],
                                                            feed_dict={X: XTrain[start:end],
                                                                       Y: YTrain[start:end]})
                    trainSummaryWriter.add_summary(summaryTrain, epoch)
                    # Flushes the event file to disk
                    trainSummaryWriter.flush()
                    meanLossTrain.append(lossTrain)
                    finTrain = time.time() - startTrain
                    print("Batch[%s:%s], epoch: %s\tloss: %s\ttime: %s s" % (start, end, epoch + 1, lossTrain, finTrain))
                    checkpoint_path = pathLogs + "/model.ckpt"
                    saver.save(sess, checkpoint_path, global_step=epoch)
                # meanLossTrain = np.mean(meanLossTrain)
                # finTime = time.time() - startTime
                # meanLossValid = []
                for start in (range(0, 40, batchSize)):
                    end = start + batchSize
                    end = start + batchSize
                    # Run summaries and measure accuracy on validation set
                    YPred_resul, loss_valid = sess.run([YPred, loss],
                                                         feed_dict={X: XTrain[start:end],
                                                                    Y: YTrain[start:end]})
                    #validationSummaryWriter.add_summary(summary_valid, epoch)
                    # print(YPred_resul[:10])
                    aa = YPred_resul[:20]
                    bb = YTrain[:20]
                    Y_pred_integer = [np.argmax(aa[i]) for i in range(20)] 
                    Y_Test = [np.argmax(bb[i]) for i in range(20)] 
                    print(Y_pred_integer)
                    print(Y_Test)
                    # Compute confusion matrix
                    cnf_matrix = confusion_matrix(Y_Test[:20], Y_pred_integer[:20])
                    print(cnf_matrix)
                    # np.set_printoptions(precision=2)

                    # Plot non-normalized confusion matrix
                    # plt.figure()
                    # plot_confusion_matrix(cnf_matrix, title='Confusion matrix, without normalization')

                    # plt.show()
                #     # Flushes the event file to disk
                #     validationSummaryWriter.flush()
                #     meanLossValid.append(loss_valid)
                # mean_loss_valid = np.mean(meanLossValid)
                # print("epoch %s / %s \ttrain_loss: %s,\tvalid_loss: %s" % (epoch + 1, epochs, mean_loss_train, mean_loss_valid))
                #
                # print(finTime)

            # train_summary_writer.close()
            # validation_summary_writer.close()
            # summary_writer.close()


if __name__ == "__main__":
    
    _main()
