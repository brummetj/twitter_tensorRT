""" TensorRT Research Project with Twitter sentiment Analysis

Author: Joshua Brummet
Email: Joshua.brummet@gmail.com, Joshua.brummet@ucdenver.edu

Usage:
    tweetRT start stream --filter=<filter> --port=<port>
    tweetRT start server [ps] [worker] [--index=<index>]
    tweetRT get reviews
    tweetRT run --port=<port>
    tweetRT download txt

Options:
    -h --help           show this screen.
    start               start command
    stream              opens a port on port number, and type of filter for tweets.
    create server       create cluster server
    ps                  parameter server
    run seq             run the program sequential
    worker              create a new cluster worker


"""
import docopt
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from tweetRT.stream import StreamManager
from tweetRT.utils.logger import Logger
from tweetRT.tweetread import sendData
from tweetRT.utils.colors import bcolors
from tweetRT.lstm.lstm_model import LSTM
from tweetRT.lstm.pre_process import PreProcess

logger = Logger("Main")

def read_tweet(model, process, sentiment):
    """
    :param model: Takes in the model object built from lstm
    :param process: process is the pre-process object that strips text
    :param sentiment: sentiment is the text string we are getting an analysis on.
    :return: output of sentiment analysis
    """
    matrix = process.get_sentence_matrix(sentiment, model.words_list)
    predicted_sentiment = model.session.run(model.prediction, {model.input_data: matrix})[0]
    if predicted_sentiment[0] > predicted_sentiment[1]:
            print("--------------")
            print("{} : ".format(sentiment) + bcolors.OKGREEN + " Positive sentiment" + bcolors.ENDC)
            print("--------------")
    else:
            print("--------------")
            print("{} : ".format(sentiment) + bcolors.FAIL + "Negative sentiment" + bcolors.ENDC)
            print("--------------")


def print_rdd(model, process, rdd):
        count = rdd.count()
        if count > 0:
            for text in rdd.collect():
                read_tweet(model, process, text)
        else:
            print("nothing yet")


def main():
    """
    Main is the runner program. It will parse arguments and determine the use case of tweetRT
    :return: sys.exit()
    """
    args = docopt.docopt(__doc__)

    try:
        # Open socket for tweet stream.
        if args['start'] and args['stream']:
                print("Starting twitter stream. Enter the following command to see tweets in a new terminal.")
                print(bcolors.OKBLUE + "tweetRT run --port={}".format(args['--port']) + bcolors.ENDC)
                stream_manager = StreamManager()
                stream_manager.start_listener(int(args['--port']))
                sendData(stream_manager.connection, args['--filter'])

        model = LSTM()

        if args['run']:
            # Connecting spark to a stream for analysis
            try:
                logger.getLogger().info("Running through and grabbing a pre trained")
                model.load_npy()
                model.reset_graph()
                model.create_graph_run()
                print("Time elapsed to create graph:  {}".format(model.duration))
                process = PreProcess()

                logger.getLogger().debug("Setting spark context")
                sc = SparkContext()
                sc.setLogLevel("ERROR")
                ssc = StreamingContext(sc, 5)

                logger.getLogger().debug("Getting tweets from localhost:{}".format(args['--port']))
                tweets = ssc.socketTextStream("localhost", int(args['--port']))

                tokens = tweets.map(lambda tweet: tweet.strip('\n'))
                tokens.foreachRDD(lambda x: print_rdd(model, process, x))

                logger.getLogger().debug("Starting spark")
                ssc.start()
                ssc.awaitTermination()
            except Exception as e:
                logger.getLogger().error(e)

        # create a server PS
        if args['start'] and args['server'] and args['ps']:
            model.start_parameter_server(int(args['--index']))

        # Create a server worker
        if args['start'] and args['server'] and args['worker']:
            try:

                model.load_npy()
                model.start()
                model.start_worker(int(args['--index']))
                model.end()
                print("Time to train data: {}".format(model.duration))
                model.close_session()
            except Exception as e:
                logger.getLogger().error(e)

        # Download a txt file as a pkl. Not used very much.
        if args['download'] and ['txt'] and args['--file']:
            try:
                print(bcolors.OKBLUE + "Saving file as pkl: " + bcolors.ENDC)
                file = args['--file']
                model.load_model(file)
                print(bcolors.OKGREEN + "File Saved :D" + bcolors.ENDC)
            except Exception as e:
                logger.getLogger().error(e)

    except Exception as e:
        logger.getLogger().error("Exception: " + str(e))


if __name__ == "__main__":
    main()