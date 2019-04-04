
"""
This module provides an interface for interacting with long lasting calculations via a TCP socket.
"""

# source: http://stackoverflow.com/questions/23828264/
# how-to-make-a-simple-multithreaded-socket-server-in-python-that-remembers-client


import socket
import threading
import time
import queue
import platform
from .log import logging

from ipydex import IPS

# for data
msgqueue = queue.Queue()
ctrlqueue = queue.Queue()
running = False
listener = None


# Colloct all known messages here to avoid confusion
class MessageContainer(object):
    def __init__(self):
        self.lmshell_inner = "lmshell_inner"
        self.lmshell_outer = "lmshell_outer"
        self.plot_reslist = "plot_reslist"
        self.change_x = "change_x"
        # change the weight matrix
        self.change_w = "change_w"
        self.run_ivp = "run_ivp"

messages = MessageContainer()

server = []
client_list = []
threads = []


class ThreadedServer(object):
    def __init__(self, host, port):

        server.append(self)
        self.host = host
        self.port = port
        confirmflag = False
        for i in range(500):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.sock.bind((self.host, self.port+i))
                break
            except socket.error as err:
                confirmflag = True
                logging.warn("port {} already in use, increasing by 1.".format(self.port+i))
                continue
        logging.debug("Connected to localhost:{}".format(self.port + i))
        if confirmflag:
            input("Press Enter.")

    def listen(self):
        self.sock.listen(5)
        while True:
            logging.info("listening")
            # wait for an incomming connection
            client, address = self.sock.accept()
            if not ctrlqueue.empty():
                msg = ctrlqueue.get()
                ctrlqueue.task_done()
                if "exit" in msg:
                    break

            client.settimeout(None)
            client_list.append(client)
            sublistener = threading.Thread(target=self.listentoclient, args=(client, address))

            threads.append(sublistener)

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


def start_stopable_thread(callable, dt=0.1, name=None):
    """
    This function produces a function that starts a thread,
    and then waits for a message to terminate

    This contruction (with a parent thread that polls a queue)
    allows to savely stop threads which perform blocking operations

    :param callable:    callable which will be the thread
    :param dt:          waiting time in seconds
    """

    def thrdfnc():
        thr = threading.Thread(target=callable)
        if name is not None:
            thr.name = name
        threads.append(thr)
        thr.daemon = True
        thr.start()

        while True:
            if not ctrlqueue.empty():
                msg = ctrlqueue.get()
                ctrlqueue.task_done()
                if "exit" in msg:
                    break
            time.sleep(dt)

        print("finish threads")

    return thrdfnc


def listen_for_connections(port):

    target = ThreadedServer('', port).listen
    thrdfnc = start_stopable_thread(target, name="listen-thread")

    thr = threading.Thread(target=thrdfnc)
    threads.append(thr)
    thr.daemon = True
    thr.start()

    # TODO: implement that flag without global keyword
    global running
    running = True


def stop_listening():

    ctrlqueue.put("exit")
    # time.sleep(2)
    # IPS()
    # server[-1].sock.close()
    # time.sleep(2)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    address = server[-1].sock.getsockname()

    if platform.system() == "Windows":
        host, port = address

        # Connecting to the any-address 0.0.0.0 is not allowed on Windows
        # https://stackoverflow.com/questions/11982562/socket-connect-to-0-0-0-0-windows-vs-mac
        if host == '0.0.0.0':
            address = ('127.0.0.1', port)

    sock.connect(address)

    sock.close()
    server[-1].sock.close()

    # TODO: implement that flag without global keyword
    global running
    running = False

def has_message(txt):
    """
    Ask the server if a specific message has arrived.
    Non-matching Messages are put back into the queue

    :param txt: message to look for
    :return: True or False
    """
    if not running:
        return False

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
    PORT = eval(input("Port? "))
    listen_for_connections(PORT)

    process_queue()

