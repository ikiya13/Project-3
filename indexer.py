import json
from bs4 import BeautifulSoup
import cbor

def createIndex():
    #variables
    allTokens = []
    parentDirectory = r"C:\Users\farme\Downloads\webpages\WEBPAGES_RAW"

    #open file
    f = open(r"C:\Users\farme\Downloads\webpages\WEBPAGES_RAW\bookkeeping.json")

    #load json
    data = json.load(f)

    #create soup object
    soup = BeautifulSoup()

    #loop through and open files
    for entry in data:
        location = entry.split("/")
        url = data[entry]

        #create path for file
        path = parentDirectory + "\\" + location[0] + "\\" + location[1]

        # data_dict = cbor.load(open(path, "rb"))

        # content = data_dict[b'raw_content'][b'value'] if b'raw_content' in data_dict and b'value' in data_dict[b'raw_content'] else ""

        # print(content)

        with open(path, "r") as website:
            print(website)
        

        

createIndex()