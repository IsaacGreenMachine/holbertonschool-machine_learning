"""module for reducer function"""
from snakebite.client import Client
from mapper import printIDs


def reducer():
    """
    takes mapper.py output and top prints top ten salaries
    sorted by totalyearlycompensation.
    The mapper and reducer run on the Hadoop environment with the
    mapred command.
    """
    client = Client("localhost", 9000, use_trash=False)
    lst = printIDs()
    lst.sort(key=firstItem, reverse=True)
    print("id\tSalary\tcompany")
    for row in lst[:10]:
        print("{0}\t{1}\t{2}".format(row[0], row[3], row[1]))


def firstItem(l):
    try:
        return int(l[3])
    except Exception:
        return 0


if __name__ == "__main__":
    reducer()
