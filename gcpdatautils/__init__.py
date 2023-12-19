from importlib.resources import files

def resource_filename(pkg, filename):
    return str(files(pkg)/filename)

ROTTEN_EGGS = resource_filename('gcpdatautils.resources', 'rotteneggs.txt')

from gcpdatautils.load_data import *

