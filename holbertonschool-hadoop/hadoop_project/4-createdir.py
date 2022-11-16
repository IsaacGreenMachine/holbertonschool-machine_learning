"""module for createdir function"""
from snakebite.client import Client


def createdir(l):
    """
    creates the directories listed on l within HDFS
    """
    client = Client("localhost", 9000, use_trash=False)
    for i in l:
        for j in client.mkdir([i]):
            pass


l = ["/Betty", "/Betty/Holberton"]
createdir(l)
