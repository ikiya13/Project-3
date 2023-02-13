#import required libraries
import sys
import pickle
import logging

#create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

#other modules
from indexer import createIndex

#call indexer passing command line argument
logging.info("Creating index.")
index = createIndex()

#save to file
logging.info("Saving to file.")
pickle.dump(index, open("indexFile.pickle", "wb"))

#read from file
logging.info("Reading from file.")
index = pickle.load(open("indexFile.pickle", "rb"))

# logging.info("Looping and printing.")
# for entry in index:
#     print(str(entry) + ": " + str(index[entry]) + "\n\n")