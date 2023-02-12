#import required libraries
import sys
import pickle

#other modules
from indexer import createIndex

#call indexer passing command line argument
index = createIndex(sys.argv[1])

#save to file
pickle.dump(index, open("indexFile.pickle", "wb"))

#read from file
index = pickle.load(open("indexFile.pickle", "rb"))