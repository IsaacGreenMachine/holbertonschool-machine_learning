#!/usr/bin/env python3
""" module for insert_school function """
import pymongo


def insert_school(mongo_collection, **kwargs):
    """
    inserts a new document in a collection based on kwargs
    mongo_collection is a pymongo collection object
    Returns the new _id
    """
    print(mongo_collection.insert_one(kwargs))
