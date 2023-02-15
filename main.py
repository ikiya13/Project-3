#import required libraries
import sys
import pickle
import logging
from pathlib import Path

#other modules
from indexer import createIndex
from indexer import searchIndex

#create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

#call indexer passing command line argument
path = Path("indexFile.pickle")
if not path.is_file():

    logging.info("Creating index.")
    index = createIndex()

    #save to file
    logging.info("Saving to file.")
    pickle.dump(index, open("indexFile.pickle", "wb"))
else:
    #read from file
    logging.info("Reading from file.")
    index = pickle.load(open("indexFile.pickle", "rb"))
    #print(index)


# input search, split 1 or 2 keywords, call searchIndex with list of tokens (1or2)
search = input("Search 1 or 2 keywords: ")
search = search.split(" ")
if len(search) > 2:
    search = input("Error: Too many keywords - Search 1 or 2 keywords: ")
    search = search.split(" ")

searchIndex(search, index)