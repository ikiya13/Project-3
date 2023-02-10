import json
from bs4 import BeautifulSoup
import cbor

check = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
wordList = []

def createIndex():
    # variables
    allTokens = []
    parentDirectory = "/Users/aniketpratap/Desktop/WEBPAGES_RAW"

    # open file
    f = open("/Users/aniketpratap/Desktop/WEBPAGES_RAW/bookkeeping.json")

    # load json
    data = json.load(f)

    # create soup object
    soup = BeautifulSoup()

    # loop through and open files
    for entry in data:
        location = entry.split("/")
        url = data[entry]

        # create path for file
        path = parentDirectory + "/" + location[0] + "/" + location[1]

        # data_dict = cbor.load(open(path, "rb"))

        # content = data_dict[b'raw_content'][b'value'] if b'raw_content' in data_dict and b'value' in data_dict[b'raw_content'] else ""

        # print(content)

        with open(path, 'r') as website:

            word = ''
            for content in website:
                soup = BeautifulSoup(content, 'html.parser')
                based = soup.find_all('div')
                for i in soup:
                    for c in i.text:
                        if c not in check and len(word) > 1:
                            # wordList.append(word)
                            print(word, path)
                            word = ''
                        if c in check:
                            word += c.lower()
                        if c == "":
                            # wordFreq[word] = 1 + wordFreq.get(word, 0)
                            # wordCount += 1
                            break


createIndex()
