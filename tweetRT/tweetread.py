import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
from tweetRT.utils.logger import Logger
import socket
import json

logger = Logger("Tweetread")

# Set up your credentials
# add your credentials between ''
consumer_key='8j9sH3OIrJ6VcWlcC8iKnYowS'
consumer_secret='Q6px5qXmQgmbxRBKbZ2lbMhqbrwOd4v5ub80tgv65KaiwxhHXK'
access_token ='950452072633090048-X658atEgVTnU2xh2dEbiUDPUsmhCY8D'
access_secret='4x7a6LFKOgEDzfiNIM86V3zHtcxEE3O1JQ80VOdLguSP7'

class TweetsListener(StreamListener):

  def __init__(self, csocket):
      super().__init__()
      self.client_socket = csocket

  def on_data(self, data):
      try:
          msg = json.loads( data )
          if msg['lang'] == 'en':
              print(msg['text'].encode('utf-8'))
              data = msg['text'] + "\n"
              self.client_socket.send(data.encode('utf-8'))
              return True
      except BaseException as e:
          print("Error on_data: %s" % str(e))
      return True

  def on_error(self, status):
      print(status)
      return True


def sendData(c_socket, filter):
    """
    Controller Function to handle twitter data and filtering.
    :param c_socket: connection to socket
    :param filter: type of filter to do sentiment analaysis on.
    :return: N/A
    """
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    twitter_stream = Stream(auth, TweetsListener(c_socket))
    twitter_stream.filter(track=[filter])

