import datetime

class Profiler(object):
    def __init__(self):
        self.start_time = 0
        self.duration = 0

    def start(self):
        self.start_time = datetime.datetime.now()

    def end(self):
        self.duration = datetime.datetime.now() - self.start_time