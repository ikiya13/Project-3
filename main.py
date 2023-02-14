#import required libraries
import sys
import pickle
import logging

#other modules
from indexer import createIndex

#create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

#call indexer passing command line argument
logging.info("Creating index.")
index = createIndex()

#save to file
logging.info("Saving to file.")
pickle.dump(index, open("indexFile.pickle", "wb"))

#read from file
logging.info("Reading from file.")
index = pickle.load(open("indexFile.pickle", "rb"))