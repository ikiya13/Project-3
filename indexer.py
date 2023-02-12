import json
from re import X
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, RegexpTokenizer
import os

blackList = {'xhtml1', 'img', 'alt', 'xhtml', 'width', 'src', 'png', 'doctype', 'html', 'public', 'tr', 'dtd', 'w3c'}

g = RegexpTokenizer(r'[A-Z]*[a-z]*[1-9]*\w+')

def tokenizeWebpage(path):
    # variables
    wordList = []

    #open webpage
    page = open(path, "r", encoding="utf8")

    #tokenize page
    for content in page:
        soup = BeautifulSoup(content, 'html.parser')
        # if bold chunk

        # if header chunk h1 h2 h3

        # if title chunk

        # if div chunk
        for i in soup.strings:
            words = g.tokenize(i)

            for word in words:
                wordList.append(word.lower())

    return wordList



def createIndex(commandlineArgument):
    # index1 STRUCTURE
    # { token : { url : { "frequency": int, "bold": int, "header": int, "title": int}}}
    index1 = dict()

    parentDirectory = commandlineArgument

    # open file and load json
    f = open(os.path.join(commandlineArgument, "bookkeeping.json"))
    jsonData = json.load(f)

    for entry in jsonData:
        location = entry.split("/")
        url = jsonData[entry]
        print(location)

        # create path for file and open file
        path = os.path.join(parentDirectory, location[0], location[1])
        
        websiteTokens = tokenizeWebpage(path)

        for word in websiteTokens:
            if word in index1:
                if url in index1[word]:
                    index1[word][url]["frequency"] += 1
                else:
                    index1[word][url] = {"frequency":1, "bold": 0, "header": 0, "title": 0}
            else:
                # index1[word] = {url:{"frequency":1, "bold": 0, "header": 0}, "title": 0}
                index1[word] = {url:{"frequency":1, "bold": 0, "header": 0, "title": 0}}
        

    #return complete index
    return index1
                    
    # by this point index1 is completed
    # iterate through index1 to create inverted_index with tdif




# python3 main.py /Users/akbenothman/Desktop/WEBPAGES_RAW