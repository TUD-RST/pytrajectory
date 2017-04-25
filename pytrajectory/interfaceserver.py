# source: http://stackoverflow.com/questions/23828264/how-to-make-a-simple-multithreaded-socket-server-in-python-that-remembers-client


import socket
import threading
import time
import Queue

# for data
msgqueue = Queue.Queue()


class ThreadedServer(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))

    def listen(self):
        self.sock.listen(5)
        while True:
            print("listening")
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
                    msgqueue.put("processed: " + data)
                else:
                    raise ValueError('Client disconnected')
            except IOError:
                client.close()
                return False


def listen_for_connections():
    ThreadedServer('', PORT).listen()


def process_queue():
    """"Here the actual work is done"""
    while True:
        if msgqueue.empty():
            print "empty queue"
        else:
            x = msgqueue.get()
            msgqueue.task_done()
            print x
            if "exit" in x:
                break
        time.sleep(1)

    print("finished")

if __name__ == "__main__":
    PORT = input("Port? ")

    # wait for incomming connections from clients
    listener = threading.Thread(target=ThreadedServer('', PORT).listen)
    listener.daemon = True
    listener.start()

    process_queue()

