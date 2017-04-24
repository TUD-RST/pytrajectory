# source: http://stackoverflow.com/questions/23828264/how-to-make-a-simple-multithreaded-socket-server-in-python-that-remembers-client


import socket
import threading
import time
import Queue

# for data
msgqueue = Queue.Queue()

# for control flow information
ctrlqueue = Queue.Queue()


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
            if not ctrlqueue.empty():
                break

            # wait for an incomming connection
            client, address = self.sock.accept()
            client.settimeout(60)
            listener = threading.Thread(target=self.listenToClient, args=(client, address))

            # end this thread if the main thread finishes
            listener.daemon = True
            listener.start()

    def listenToClient(self, client, address):
        size = 1024
        while True:
            try:
                data = client.recv(size)
                if data:
                    msgqueue.put("processed: " + data)
                else:
                    raise ValueError('Client disconnected')
            except:
                client.close()
                return False


def finish_server():
    """
    connect to the server and thus trigger the lookup of the ctrlqueue
    :return:  None
    """
    ctrlqueue.put("exit")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('localhost', PORT))


def worker():
    "Here the actual work is done"
    while True:

        if msgqueue.empty():
            print "empty queue"
        else:
            x = msgqueue.get()
            msgqueue.task_done()
            print x
            if "exit" in x:
                finish_server()
                break
        time.sleep(1)

    print("finished")

if __name__ == "__main__":
    PORT = port_num = input("Port? ")

    threading.Thread(target=worker).start()

    # wait for incomming connections from clients
    ThreadedServer('', port_num).listen()