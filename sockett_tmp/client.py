import socket
import time
import sys


from ipHelp import IPS

from pytrajectory import interfaceserver as ifs

HOST, PORT = "localhost", 5006


# Create a socket (SOCK_STREAM means a TCP socket)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))


if len(sys.argv) > 1:
    msg = sys.argv[1]
    assert msg in dir(ifs.messages)
    sock.send(msg)

else:
    sock.send(ifs.messages.change_x)
    #sock.send(ifs.messages.plot_reslist)


# sock.send("exit\n")
# print "exit"

sock.close()
