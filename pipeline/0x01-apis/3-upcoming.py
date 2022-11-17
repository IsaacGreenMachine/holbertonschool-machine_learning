#!/usr/bin/env python3
"""
using the (unofficial) SpaceX API,
displays the upcoming launch with these information:
  Name of the launch
  The date (in local time)
  The rocket name
  The name (with the locality) of the launchpad

The “upcoming launch” is the one which is the soonest from now,
in UTC (we encourage you to use the date_unix for sorting it)
and if 2 launches have the same date, use the first one in the API result.
"""
import requests
import time
import datetime
if __name__ == '__main__':
    nearest_launch = requests.get(
        'https://api.spacexdata.com/latest/launches/next'
        ).json()
    '''
    current_time = time.mktime(datetime.datetime.today().timetuple())
    nearest = 2000000000
    r = requests.get('https://api.spacexdata.com/latest/launches')
    for i in r.json():
        if i['date_unix'] > current_time and i['date_unix'] < nearest:
            nearest = i['date_unix']
            nearest_launch = i
    '''
    launchname = nearest_launch['name']
    date = nearest_launch['date_local']
    rkt = requests.get(
        'https://api.spacexdata.com/v4/rockets/' + nearest_launch['rocket'])
    rocketname = rkt.json()['name']
    lp = requests.get(
        'https://api.spacexdata.com/latest/launchpads/' +
        nearest_launch['launchpad'])
    padname = lp.json()['name']
    locality = lp.json()['locality']
    print("{} ({}) {} - {} ({})".format(launchname,
          date, rocketname, padname, locality))
