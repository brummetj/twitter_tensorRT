import numpy as np
import tensorflow as tf
import os
import sys
import pickle
import datetime
from random import randint
from tweetRT.lstm.constants import hyper_parameter, cluster_specification
from tweetRT.utils.logger import Logger
from tweetRT.utils.clock import Profiler

logger = Logger("Model")


class LSTM(Profiler):
    def __init__(self):
        """
        Initialize data.
        """
        super().__init__()
        self.accuracy = None
        self.prediction = None
        self.last = None
        self.value = None
        self.bias = None
        self.weight = None
        self.correct_prediction = None
        self.input_data = None
        self.saver = None
        self.labels = None
        self.session = None
        self.path = None
        self.word_vectors = None
        self.words_list = None
        self.loss = None
        self.word_ids = None
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.lstm_cell = None

    def create_graph_run(self):
        """
        create graph from data. This needs to be called before anything else to work.
        :return: N/A
        """
        try:
            logger.getLogger().info("Starting create_graph timer")
            logger.getLogger().info("Creating pre-trained rnn graph on device:"
                                    " {}".format(tf.test.gpu_device_name()))
            super().start()
            logger.getLogger().debug("Creating features")
            self.labels = tf.placeholder(tf.float32, [hyper_parameter['BATCH_SIZE'],
                                                      hyper_parameter['NUM_CLASSES']])

            logger.getLogger().debug("Creating input data")
            self.input_data = tf.placeholder(tf.int32, [hyper_parameter['BATCH_SIZE'],
                                                        hyper_parameter['MAX_SEQ_LENGTH']])
            data = tf.Variable(tf.zeros([hyper_parameter['BATCH_SIZE'],
                                         hyper_parameter['MAX_SEQ_LENGTH'],
                                         hyper_parameter['NUM_DIMENSIONS']]), dtype=tf.float32)

            logger.getLogger().debug("Defining data from word vector and list of words")
            data = tf.nn.embedding_lookup(self.word_vectors, self.input_data)
            # getting data from list and word vectors.

            logger.getLogger().debug("Building a LSTM neural network with "
                                     + str(hyper_parameter['LSTM_UNITS']) + " Units")
            self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(hyper_parameter['LSTM_UNITS'])

            logger.getLogger().debug("Opening the gateway for the LSTM so data can flow through")
            self.lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=self.lstm_cell, output_keep_prob=0.25)

            # get the value of our rnn
            logger.getLogger().debug("Getting values from our RNN LSTM")
            value, _ = tf.nn.dynamic_rnn(self.lstm_cell, data, dtype=tf.float32)

            # Get the weight of each variable.
            logger.getLogger().debug("Getting weight")
            self.weight = tf.Variable(tf.truncated_normal([hyper_parameter['LSTM_UNITS'],
                                                           hyper_parameter['NUM_CLASSES']]))

            logger.getLogger().debug("Setting bias")
            self.bias = tf.Variable(tf.constant(0.1, shape=[hyper_parameter['NUM_CLASSES']]))

            # transpose values between 0 - 2
            value = tf.transpose(value, [1, 0, 2])

            self.last = tf.gather(value, int(value.get_shape()[0]) - 1)

            # Lets build our prediction output.
            logger.getLogger().debug("Getting prediction value")
            self.prediction = (tf.matmul(self.last, self.weight) + self.bias)

            # Correct prediction for accuracy.
            self.correct_prediction = tf.equal(tf.argmax(self.prediction, 1),
                                               tf.argmax(self.labels, 1))

            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            logger.getLogger().debug("Getting new session")

            self.session = tf.InteractiveSession()

            logger.getLogger().debug("Restoring saved model")

            saver = tf.train.Saver(tf.global_variables())

            saver.restore(self.session, tf.train.latest_checkpoint(self.path + '/models'))

            super().end()
            logger.getLogger().info("Ending create_graph timer")
        except Exception as E:
            logger.getLogger().error(E)

    def restore_graph(self):
        """
        This will restore a graph that was saved into the '/models' directory
        :return:
        """
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.session, tf.train.latest_checkpoint(self.path + '/models'))

    def reset_graph(self):
        """
        Reset tensorflow graph
        :return:
        """
        tf.reset_default_graph()

    def close_session(self):
        """
        close session for tensorflow.
        :return:
        """
        self.session.close()

    def load_data(self):
        """
        Here we load data from the pkl file. Needs to be a pkl file of the static name below.

        :return:
        """
        try:
            logger.getLogger().info("Retrieving word vector data")
            with open(self.path + '/data/glove.pkl', 'rb') as f:
                self.word_vectors = pickle.load(f)

            self.words_list = []
            for i in self.word_vectors:
                self.words_list.append(i)

            word_vectors = []
            for i in self.words_list:
                word_vectors.append(self.word_vectors[i])

            self.word_vectors = word_vectors
            logger.getLogger().debug("Finished loading word vector and converting to word list")
        except Exception as e:
            return e

    def load_model(self, file):
        try:
            print(file)
            f = open(self.path + '/data/' + file, 'r')
            model = {}
            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                embedding = np.array([float(val) for val in splitLine[1:]])
                model[word] = embedding
            print("Done.", len(model), " words loaded!")
            with open(self.path + '/data/glove.pkl', 'wb') as output:
                pickle.dump(model, output)
        except Exception as e:
            return e

    def load_npy(self):
        self.words_list = np.load(self.path + '/data/wordsList.npy')
        self.words_list.tolist()
        self.words_list =  [word.decode('UTF-8') for word in self.words_list]
        self.word_vectors = np.load(self.path + '/data/wordVectors.npy')
        self.word_ids = np.load(self.path + '/data/idsMatrix.npy')
        logger.getLogger().debug("Shape of word vector " + str(self.word_vectors.shape))
        logger.getLogger().debug("Shape of word ids" + str(self.word_ids.shape))

    def start_parameter_server(self, task_index):
        """
        This function starts the head of the cluster workers, there can be more then one.
        :param task_index: index of the head
        :return: a cluster
        """
        cluster_spec = tf.train.ClusterSpec(cluster_specification)
        server = tf.train.Server(cluster_spec, job_name='ps', task_index=task_index, config=self.config)
        server.join()

    def get_train_batch(self):
        """
        This will get the batch for training data
        :return: array of data, labels for features, 1 or 0
        """
        labels = []
        arr = np.zeros([hyper_parameter['BATCH_SIZE'], hyper_parameter['MAX_SEQ_LENGTH']])
        for i in range(hyper_parameter['BATCH_SIZE']):
            if (i % 2 == 0):
                num = randint(1, 11499)
                labels.append([1, 0])
            else:
                num = randint(13499, 24999)
                labels.append([0, 1])
            arr[i] = self.word_ids[num - 1:num]
        return arr, labels

    def get_test_batch(self):
        """
        This will test a batch in the word_vector for finding a set a of words.
        :return: a array and features
        """
        labels = []
        arr = np.zeros([hyper_parameter['BATCH_SIZE'], hyper_parameter['MAX_SEQ_LENGTH']])
        for i in range(hyper_parameter['BATCH_SIZE']):
            num = randint(11499, 13499)
            if num <= 12499:
                labels.append([1, 0])
            else:
                labels.append([0, 1])
            arr[i] = self.word_ids[num - 1: num]
        return arr, labels

    def start_worker(self, task_index):
        """
        This function will start a worker for a distributed parallel training process
        :param task_index: index of worker
        :return: trained data set
        """
        logger.getLogger().info("Setting worker device and specifications")
        cluster_spec = tf.train.ClusterSpec(cluster_specification)
        server = tf.train.Server(cluster_spec, job_name="worker", task_index=task_index, config=self.config)
        worker_device = "/job:worker/task:{}".format(task_index)

        logger.getLogger().info("Starting rnn with worker: {}".format(task_index))
        with tf.device(tf.train.replica_device_setter(worker_device=worker_device,
                                                      cluster=cluster_spec)):

            tf.reset_default_graph()
            logger.getLogger().debug("Creating features - Worker : " + str(task_index))
            labels = tf.placeholder(tf.float32, [hyper_parameter['BATCH_SIZE'],
                                                 hyper_parameter['NUM_CLASSES']])

            logger.getLogger().debug("Creating input data - Worker : " + str(task_index))
            self.input_data = tf.placeholder(tf.int32, [hyper_parameter['BATCH_SIZE'],
                                                        hyper_parameter['MAX_SEQ_LENGTH']])

            logger.getLogger().debug("Defining data from word vector and list of words  - Worker: " + str(task_index))
            data = tf.Variable(tf.zeros([hyper_parameter['BATCH_SIZE'],
                                         hyper_parameter['MAX_SEQ_LENGTH'],
                                         hyper_parameter['NUM_DIMENSIONS']]), dtype=tf.float32)

            # self.word_vectors = tf.cast(self.word_vectors,tf.float32)
            data = tf.nn.embedding_lookup(self.word_vectors, self.input_data)

            logger.getLogger().debug("Building a LSTM neural network with "
                                     + str(hyper_parameter['LSTM_UNITS']) + " Units - Worker: " + str(task_index))

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(hyper_parameter['LSTM_UNITS'])

            logger.getLogger().debug("Opening the gateway for the LSTM so data can flow through - Worker: " + str(task_index))
            lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)

            # get the value of our rnn
            logger.getLogger().debug("Getting values from our RNN LSTM - Worker: " + str(task_index))
            value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

            # Get the weight of each variable.
            logger.getLogger().debug("Getting weight - Worker: " + str(task_index))
            self.weight = tf.Variable(tf.truncated_normal([hyper_parameter['LSTM_UNITS'],
                                                           hyper_parameter['NUM_CLASSES']]))

            logger.getLogger().debug("Setting bias  - Worker: " + str(task_index))
            self.bias = tf.Variable(tf.constant(0.1, shape=[hyper_parameter['NUM_CLASSES']]))

            # transpose values
            self.value = tf.transpose(value, [1, 0, 2])

            self.last = tf.gather(self.value, int(self.value.get_shape()[0]) - 1)

            # Lets build our prediction output.
            logger.getLogger().debug("Getting prediction value  - Worker: " + str(task_index))
            self.prediction = (tf.matmul(self.last, self.weight) + self.bias)

            # Correct prediction for accuracy.
            self.correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            logger.getLogger().debug("Getting loss")

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction,
                                                                                   labels=labels))

            global_step = tf.contrib.framework.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer().minimize(self.loss, global_step=global_step)

            tf.summary.scalar('Loss', self.loss)
            tf.summary.scalar('Accuracy', self.accuracy)

            logger.getLogger().debug("Starting to train the graph - Worker: " + str(task_index))

            local_step_value = 0
            hooks = [tf.train.StopAtStepHook(last_step=hyper_parameter['ITERATIONS'])]
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=task_index == 0,
                                                   checkpoint_dir=self.path + "/models/checkpoints",
                                                   hooks=hooks) as sess:

                while not sess.should_stop():
                    next_batch, next_batch_labels = self.get_train_batch()
                    _, glob_step, loss, accuracy = sess.run([optimizer, global_step, self.loss, self.accuracy],
                                                            {self.input_data: next_batch, labels: next_batch_labels})
                    local_step_value += 1
                    if local_step_value % 100 == 0:
                        print("Local Step %d, Global Step %d (Loss: %.2f) (Accuracy: %.2f)" % (
                               local_step_value, glob_step, loss, accuracy))
                self.session = sess

            logger.getLogger().debug("Finished")

