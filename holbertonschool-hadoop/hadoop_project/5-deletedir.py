"""module for deletedir function"""
from snakebite.client import Client


def deletedir(l):
    """
    removes the directories listed on l within HDFS
    """
    client = Client("localhost", 9000, use_trash=False)
    for i in l[::-1]:
        for j in client.rmdir([i]):
            pass


l = ["/Betty", "/Betty/Holberton"]
deletedir(l)
