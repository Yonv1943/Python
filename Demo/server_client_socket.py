import sys
import time

import socket
import pickle

"""
Source: Yonv1943 2019-05-04
https://github.com/Yonv1943/Python/upload/master/Demo/server_client_socket.py

Python send and receive objects through Sockets - Sudheesh Singanamalla
https://stackoverflow.com/a/47396267/9293137
import socket, pickle

Pickle EOFError: Ran out of input when recv from a socket - Antti Haapala
https://stackoverflow.com/a/24727097/9293137
from multiprocessing.connection import Client
"""


def run_client(host, port):
	data = ['any', 'object']  # the data you wanna send

	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.connect((host, port))

	while True:
		data_bytes = pickle.dumps(data)
		s.send(data_bytes)
		print('Send:', type(data), sys.getsizeof(data_bytes))

		time.sleep(0.5)  # 0.5 second
	# s.close()
	pass


def run_server(host, port):
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind((host, port))
	s.listen(1)
	print('Server Listening')

	conn, addr = s.accept()
	print('Server connected by:', addr)

	while True:
		data_bytes = conn.recv(1024)  # [bufsize=1024] >= [sys.getsizeof(data_bytes)]
		data = pickle.loads(data_bytes)  # Pickle EOFError: Ran out of input, when data_bytes is too large.
		print('Received:', type(data), sys.getsizeof(data_bytes))
	# conn.close()
	# s.close()
	pass


if __name__ == '__main__':
	server_host = 'x.x.x.x'  # host = 'localhost'
	server_port = 32928  # if [Address already in use], use another port


	def get_ip_address(remote_server="8.8.8.8"):
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		s.connect((remote_server, 80))
		return s.getsockname()[0]


	if get_ip_address() == server_host:
		run_server(server_host, server_port)  # first, run this function only in server
	else:
		run_client(server_host, server_port)  # then, run this function only in client
