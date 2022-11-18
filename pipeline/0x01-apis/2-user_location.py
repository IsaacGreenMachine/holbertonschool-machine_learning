#!/usr/bin/env python3
"""
prints the location of a specific user by using the GitHub API
args[1] : GitHub user with the full API URL
    example: ./2-user_location.py https://api.github.com/users/holbertonschool
    If the user doesn’t exist, print Not found
    If the status code is 403, print "R"eset in X min"
    where X is minutes from now and the value of X-Ratelimit-Reset
MAKE SURE FILE IS EXECUTABLE
"""
import sys
import requests
import datetime
if __name__ == '__main__':
    useraddress = sys.argv[1]
    r = requests.get(useraddress)
    if r.status_code == 403:
        dt = datetime.datetime.fromtimestamp(
            int(r.headers['X-RateLimit-Reset']))
        now = datetime.datetime.now()
        if dt.minute < now.minute:
            remain = dt.minute + 60 - now.minute
        else:
            remain = dt.minute - now.minute
        print("Reset in {} min".format(remain))
    elif r.status_code == 404:
        print("Not found")
    else:
        print(r.json()['location'])
