#!/usr/bin/env python3
""" module for sentientPlanets function """
import requests


def sentientPlanets():
    '''
    returns a list of names of the home planets
    of all sentient species from SWAPI AI
    Dont forget the pagination
    '''
    species = []
    specieslink = 'https://swapi-api.hbtn.io/api/species/'
    while specieslink:
        req = requests.get(specieslink)
        species += req.json()['results']
        specieslink = req.json()['next']
    sentients = []
    for specie in species:
        if specie['designation'] == 'sentient':
            try:
                sentients.append(requests.get(
                    specie['homeworld']).json()['name'])
            except Exception:
                pass
    return sentients
