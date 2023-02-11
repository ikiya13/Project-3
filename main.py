#import required libraries
import sys

#other modules
from indexer import createIndex

#call indexer passing command line argument
createIndex(sys.argv[1])