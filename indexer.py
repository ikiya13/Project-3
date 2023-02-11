import json
from re import X
from bs4 import BeautifulSoup
import os

#parentDirectory = sys.argv[1]
check = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
wordList = []
index1 = dict()

def createIndex(commandlineArgument):
    # variables
    allTokens = []
    parentDirectory = commandlineArgument

    # open file
    f = open(os.path.join(commandlineArgument, "bookkeeping.json"))

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
        print(path)



        # index1 STRUCTURE
        # { token : { url : { "frequency": int, "bold": int, "header": int, "title": int}}}

        with open(path, 'r') as website:

            word = ''
            for content in website:
                webWords = []
                soup = BeautifulSoup(content, 'html.parser')
                based = soup.find_all('div')
                # if bold chunk

                # if header chunk h1 h2 h3

                # if title chunk

                # if div chunk
                for i in soup:
                    for c in i.text:
                        if c not in check and len(word) > 1:
                            webWords.append(word)
                            wordList.append(word)
                            if word in index1:
                                if url in index1[word]:
                                    index1[word][url]["frequency"] += 1
                                else:
                                    index1[word][url] = {"frequency":1, "bold": 0, "header": 0, "title": 0}
                            else:
                                index1[word] = {url:{"frequency":1, "bold": 0, "header": 0}, "title": 0}


                            word = ''
                        if c in check:
                            word += c.lower()
                        if c == "":
                            # wordFreq[word] = 1 + wordFreq.get(word, 0)
                            # wordCount += 1
                            break

            # by this point index1 is completed
            # iterate through index1 to create inverted_index with tdif




# python3 main.py /Users/akbenothman/Desktop/WEBPAGES_RAW