"""module for download function"""
from snakebite.client import Client
import os


def download(l):
    """
    retrieves from the HDFS files listed in l and
    store them in the home /tmp folder of the user
    """
    try:
        client = Client("localhost", 9000, use_trash=False)
        filepath = os.path.dirname(os.path.abspath(__file__))
        os.mkdir(filepath + '/tmp')
    except Exception:
        pass
    for i in l:
        for j in client.cat([i]):
            for k in j:
                with open('/tmp/' + i.split('/')[-1], 'w') as f:
                    f.write(k)


l = ["/holbies/input/lao.txt"]
download(l)
lao = open("/tmp/lao.txt", "r")
print(lao.read())
lao.close()
