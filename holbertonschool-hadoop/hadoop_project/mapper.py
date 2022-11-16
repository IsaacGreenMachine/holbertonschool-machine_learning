"""module for printIDs function"""
from snakebite.client import Client


def printIDs():
    """
    takes the rows of the salaries.csv and print:
        the id,
        the company
        totalyearlycompensation items.
    Id and company will be separated by a tabulation
    company and totalyearlycompensation will be separated by a comma
    """
    finallist = []
    client = Client("localhost", 9000, use_trash=False)
    for i, val in enumerate(client.cat(['/salaries.csv'])):
        if i == 0:
            for group in list(val):
                for row in group.split('\n'):
                    spt = row.split(',')
                    finallist.append(spt)
            return finallist


if __name__ == "__main__":
    for row in printIDs():
        try:
            print("{0}\t{1},{2}".format(row[0], row[1], row[3]))
        except Exception:
            print(row)
