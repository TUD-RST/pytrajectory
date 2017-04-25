import socket
import time


from ipHelp import IPS

HOST, PORT = "localhost", 5006


# Create a socket (SOCK_STREAM means a TCP socket)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
IPS()
sock.connect((HOST, PORT))

sock.send("abc\n")
print "abc"
time.sleep(2)
sock.send("xyz\n")
print "xyz"
time.sleep(2)
sock.send("exit\n")
print "exit"


sock.close()

if 0:
    try:
        # Connect to server and send data
        sock.sendall(data + "\n")

        # Receive data from the server and shut down
        received = sock.recv(1024)
    finally:
        sock.close()