import socket


class StreamManager():

    def __init__(self):
        self.socket = None
        self.connection = None

    def start_listener(self, port):
        """
        Start listening on port 5555 for tweets.
        :return:
        """
        self.socket = socket.socket()  # Create a socket object
        host = "127.0.0.1"  # Get local machine name
        port = int(port)
        # Reserve a port for your service.
        self.socket.bind((host, port))  # Bind to the port

        self.socket.listen(5)  # Now wait for client connection.
        self.connection, addr = self.socket.accept()  # Establish connection with client.

        print("Received request from: " + str(addr))


    def close_listener(self):
        """
        Close the socket
        """
        self.socket.shutdown(socket.SHUT_RDWR)

