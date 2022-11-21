"""
1. utils functions
"""
import time 
import datetime
import numpy as np

def getTime():
    """
    return accurate time point in format: Year-Month-Day-Hour:Minute:Second.unique_labels
    """ 
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S.%f')

def combine_dict_to_list(dicts):
    ## dicts is list of dicts
    if isinstance(dicts,dict):
        return dicts
    if len(dicts)==0:
        return {}
    if len(dicts)==1:
        return dicts[0]

    keys=[]
    for db in dicts: 
        keys += db.keys()
   
    dictReturn = {}
    for key in keys:
        dictReturn[key] = []
        
    for db in dicts:
        for key in dictReturn:
            if key in db:
                dictReturn[key].append(db[key])
            else:
                dictReturn[key].append(None)
    return dictReturn