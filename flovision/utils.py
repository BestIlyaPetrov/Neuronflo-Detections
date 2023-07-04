import os
import traceback

import socket
from zeroconf import ServiceBrowser, Zeroconf

class IPListener:
    def __init__(self, target_service_name):
        self.target_service_name = target_service_name
        self.found_ip = None

    def remove_service(self, zeroconf, type, name):
        pass

    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info:
            addresses = [socket.inet_ntoa(addr) for addr in info.addresses]
            print(f"Discovered service: {name}, IP addresses: {', '.join(addresses)}")
            
        if info and name == self.target_service_name:
            address = socket.inet_ntoa(info.addresses[0])
            self.found_ip = address
            print(f"Found IP address of {name}: {address}")

    def update_service(self, zeroconf, type, name):
        pass


def findLocalServer(target_service_name = "neuronflo-server._workstation._tcp.local."):
    #Find the IP of the windows server
    listener = IPListener(target_service_name)
    zeroconf = Zeroconf()

    try:
        print("Looking for server IP...")
        browser = ServiceBrowser(zeroconf, "_workstation._tcp.local.", listener)
        while True:
            if listener.found_ip is not None:
                break
    except KeyboardInterrupt:
        pass
    finally:
        zeroconf.close()

    # Continue with more lines of code after the IP address has been found
    print(f"IP address found: {listener.found_ip}, continuing with the rest of the script...")
    return listener.found_ip



#returns the highest image index in a directory
def get_highest_index(folder_path):
    max_index = -1

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            try:
                index = int(filename.split("_")[2].split(".")[0])
                if index > max_index:
                    max_index = index
            except Exception as e:#(ValueError, IndexError):
                print("Couldn't parse the image index in: ", folder_path)
                print(e)
                traceback.print_exc()
                pass

    return max_index