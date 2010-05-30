#!/usr/bin/env python
# encoding: utf-8
"""
vecsearch.py

Created by mikepk on 2010-05-30.
Copyright (c) 2010 Michael Kowalchik. All rights reserved.
"""

import sys
import os
import unittest

import getopt
from Daemon import Daemon

import signal
import socket
import threading
import SocketServer
import time

# standard python logging
import logging
import logging.handlers

#sys.path.append( os.path.dirname( os.path.realpath( __file__ ) ) )

sys.argv[0] = 'vecsearch'


class Logger:
    def __init__(self,log_file = '/tmp/vecsearch.log'):

        LOG_FILENAME = log_file

        # Set up a specific logger with our desired output level
        self.my_logger = logging.getLogger('VecSearch')
        self.my_logger.setLevel(logging.DEBUG)

        # Add the log message handler to the logger
        handler = logging.handlers.RotatingFileHandler(
                      LOG_FILENAME, 
                      maxBytes=1024*1024*20, 
                      backupCount=10)

        # create formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        # add formatter to handler
        handler.setFormatter(formatter)
        self.my_logger.addHandler(handler)
        self.write("Logging Started")
        
    def write(self,msg):
        if msg == '\n':
            return
        # msg = msg.rstrip()
        # self.my_logger.debug(repr(msg))
        self.my_logger.debug(msg)



class ThreadedTCPRequestHandler(SocketServer.StreamRequestHandler):
        
    def handle(self):
        # data = self.request.recv(1024)
        data = ''
        while not data:
            self.wfile.write("READY: ")
            data = self.rfile.readline().strip()

        self.server.logger.write("Received %s" % data)

        # execute index, query or compare command

        response = "%s\n" % data

        self.wfile.write(response)
        

class ThreadedTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    pass



class Query():
    '''Execute queries.'''
    def __init__(self,logger):
        self.command_lock = threading.Lock()
        self.logger = logger
        self.command_queue = []
        
    def work(self):
        self.logger.write("Starting query thread.\n")
        while 1:
            try:
                data = ''
                self.command_lock.acquire()
                if self.command_queue:
                    data = self.command_queue.pop(0)
                self.command_lock.release()
                if data == 'q':
                    self.logger.write("Running Command: %s\n" % str(data))
                elif data == 'email':
                    self.logger.write("Running Command: %s\n" % str(data))
            except Exception,err:
                self.logger.write("An error occured\n")
                #self.logger.flush()                
                self.logger.write('%s\nError occured\n-------------------\n%s\n' % (str(sys.exc_info()),str(err)))
                #self.logger.flush()
            time.sleep(1)


class VecSearchServer(Daemon):
    '''The Vector Search server.'''
    def __init__(self, pidfile, stdin='/dev/null', stdout='/dev/null', stderr='/dev/null', log=None):
        Daemon.__init__(self, pidfile, stdin, stdout, stderr)
        self.logger = log

    def run(self):
        HOST, PORT = "localhost", 7500

        self.logger.write("Starting the search server\n=====================\n")

        server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
        
        ip, port = server.server_address

        server.logger = self.logger
        #server.worker = Query(self.logger)

        #worker_thread = threading.Thread(target=server.worker.work)
        #worker_thread.start()

        signal.signal(signal.SIGHUP,  lambda *args: self.restart())
        signal.signal(signal.SIGTERM,  lambda *args: self.stop())
        signal.signal(signal.SIGQUIT, lambda *args: self.stop())

        # Start a thread with the server -- that thread will then start one
        # more thread for each request
        # server_thread = threading.Thread(target=server.serve_forever)
        server.serve_forever()
        # Exit the server thread when the main thread terminates
        # server_thread.setDaemon(True)
        # server_thread.start()


def main(argv=None):
    logfile = Logger()
    vs = VecSearchServer('/var/run/vecsearch.pid',stdout='/var/tmp/log1', stderr='/var/tmp/log2', log=logfile)
    vs.start()

if __name__ == "__main__":
    sys.exit(main())
