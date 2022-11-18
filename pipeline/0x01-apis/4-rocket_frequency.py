#!/usr/bin/env python3
"""
using the (unofficial) SpaceX API,
displays the number of launches per rocket
All launches should be taking in consideration
Each line should contain the rocket name and the number of launches
Order the result by the number launches (descending)
If multiple rockets have the same amount of launches,
order them by alphabetic order (A to Z)
"""
import requests
if __name__ == '__main__':
    r = requests.get('https://api.spacexdata.com/latest/launches')
    launch_dict = {}
    for i in r.json():
        rocketname = requests.get(
            'https://api.spacexdata.com/latest/rockets/' +
            i['rocket']).json()['name']
        if launch_dict.get(rocketname) is not None:
            launch_dict[rocketname] += 1
        else:
            launch_dict[rocketname] = 0
    while len(launch_dict) > 0:
        max = 0
        maxrocket = None
        for i in launch_dict.copy():
            if launch_dict[i] > max:
                maxrocket = i
                max = launch_dict[i]
                rocketlist = [i]
            elif launch_dict[i] == max:
                rocketlist.append(i)
        rocketlist.sort()
        for i in rocketlist:
            launch_dict.pop(i)
        for i in rocketlist:
            print("{}: {}".format(i, max))
