
"""
This module provides an interface for interacting with long lasting calculations via a TCP socket.
"""

# source: http://stackoverflow.com/questions/23828264/
# how-to-make-a-simple-multithreaded-socket-server-in-python-that-remembers-client


import socket
import threading
import time
import Queue
from log import logging

# for data
msgqueue = Queue.Queue()
running = False


# Colloct all known messages here to avoid confusion
class MessageContainer(object):
    def __init__(self):
        self.lmshell_inner = "lmshell_inner"
        self.lmshell_outer = "lmshell_outer"
        self.plot_reslist = "plot_reslist"
        self.change_x = "change_x"
        self.run_ivp = "run_ivp"

messages = MessageContainer()


class ThreadedServer(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        for i in range(5):
            try:
                self.sock.bind((self.host, self.port+i))
            except socket.error as err:
                logging.warn("port {} already in use, increasing by 1.".format(self.port+i))
                continue



    def listen(self):
        self.sock.listen(5)
        while True:
            logging.info("listening")
            # wait for an incomming connection
            client, address = self.sock.accept()
            client.settimeout(None)
            sublistener = threading.Thread(target=self.listentoclient, args=(client, address))

            # end this thread if the main thread finishes
            sublistener.daemon = True
            sublistener.start()

    def listentoclient(self, client, address):
        size = 1024
        while True:
            try:
                data = client.recv(size)
                if data:
                    msgqueue.put(data)
                else:
                    logging.info('Client disconnected')
                    client.close()
            except IOError:
                client.close()
                return False


def listen_for_connections(port):
    listener = threading.Thread(target=ThreadedServer('', port).listen)
    listener.daemon = True
    listener.start()

    # TODO: implement that flag without global keyword
    global running
    running = True


def has_message(txt):
    """
    Non-matching Messages ar put back into the queue

    :param txt: message to look for
    :return: True or False
    """
    assert running
    if msgqueue.empty():
        return False

    msg = msgqueue.get()

    if txt in msg:
        return True
    else:
        msgqueue.put(msg)


def process_queue():
    """"simulate to perform some work (for testing)"""
    while True:
        if msgqueue.empty():
            logging.debug("empty queue")
        else:
            msg = msgqueue.get()
            msgqueue.task_done()
            logging.info("tcp-msg: %s" % str(msg))
            if "exit" in msg:
                break
        time.sleep(1)

    logging.info("finished")

if __name__ == "__main__":
    PORT = input("Port? ")
    listen_for_connections(PORT)

    process_queue()

