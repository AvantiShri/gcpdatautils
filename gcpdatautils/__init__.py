from importlib.resources import files
from load_data import *

def resource_filename(pkg, filename):
    return str(files(pkg)/filename)

ROTTEN_EGGS = resource_filename('gcpdatautils.resources', 'rotteneggs.txt')


