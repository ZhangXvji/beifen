import time
import socket
import struct

def udp():
    udp_addr = ("", 8063)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(udp_addr)
    while True:
        data, addr = udp_socket.recvfrom(1024)
        received_data = struct.unpack("????", data)
        print(received_data)
        time.sleep(1)

if __name__ == '__main__':
    udp()