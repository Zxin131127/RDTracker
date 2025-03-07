# from TransformerTrack.pytracking.evaluation import Tracker, get_dataset, trackerlist
from pytracking.evaluation import Tracker, get_dataset, trackerlist


def RDTracker_SV248():
    trackers = trackerlist('rdtracker', 'rdtracker', range(1))
    dataset = get_dataset('sv248')
    return trackers, dataset

def RDTracker_VISO():
    trackers = trackerlist('rdtracker', 'rdtracker', range(1))
    dataset = get_dataset('viso')
    return trackers, dataset


def RDTracker_SatSOT():
    trackers = trackerlist('rdtracker', 'rdtracker', range(1))
    dataset = get_dataset('satsot')
    return trackers, dataset



