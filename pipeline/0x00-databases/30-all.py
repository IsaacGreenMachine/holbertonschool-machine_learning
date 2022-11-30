#!/usr/bin/env python3
""" module for list_all function """
import pymongo


def list_all(mongo_collection):
    '''
    lists all documents in a collection
    Return an empty list if no document in the collection
    mongo_collection will be the pymongo collection object
    '''
    return mongo_collection.find()
