#!/usr/bin/env python3
""" module for availableShips function """
import requests


def availableShips(passengerCount):
    """
    returns a list of ships that can hold a given number
    of passengers from SWAPI API
    Dont forget the pagination
    If no ship available, return an empty list.
    """
    ships = []
    shiplink = 'https://swapi-api.hbtn.io/api/starships/'
    # print(r.status_code)
    # print(r.encoding)
    # print(r.headers)
    # print(r.links)
    # print(r.content)
    # print(r.text)
    while shiplink:
        req = requests.get(shiplink)
        ships += req.json()['results']
        shiplink = req.json()['next']
        # print(shiplink)
    # print(ships)
    goodships = []
    for ship in ships:
        if (
            ship['passengers'] not in ['n/a', 'unknown'] and
            int(ship['passengers'].replace(',', '')) >= passengerCount
                ):
            goodships.append(ship['name'])
    return goodships
