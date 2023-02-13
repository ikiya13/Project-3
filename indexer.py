import os
import sys
import json
import logging
import nltk
from bs4 import BeautifulSoup

# Download the NLTK Punkt tokenizer and the WordNet lemmatizer
nltk.download("punkt")
nltk.download("wordnet")

#create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

def createIndex():
    # Dictionary to store the index
    index = {}

    # The location of the corpus
    corpus_dir = sys.argv[1]

    # Open JSON file and loop over entries
    jsonData = json.load(open(os.path.join(corpus_dir, "bookkeeping.json")))    

    for filename in jsonData:
        # Get full filepath
        location = filename.split("/")
        url = jsonData[filename]
        file_path = os.path.join(corpus_dir, location[0], location[1])
        logging.info("Processing file: %s", location)

        # Open file
        with open(file_path, "r", encoding = "utf-8") as f:
            contents = f.read()

            # parse the HTML contents of the file
            soup = BeautifulSoup(contents, "lxml")

            # Get the text
            text = soup.get_text()

            # Tokenize the text
            tokens = nltk.tokenize.word_tokenize(text.lower())

            # Lemmatize the tokens
            lemmatized_tokens = [nltk.stem.WordNetLemmatizer().lemmatize(token) for token in tokens]

            # Add the tokens to the index
            for token in lemmatized_tokens:
                # If the token already exists
                if token in index:
                    # If the URL is already included
                    if url in index[token]:
                        index[token][url]["frequency"] += 1
                    # If we are at a new URL
                    else:
                        index[token][url] = {"frequency":1, "bold": 0, "header": 0, "title": 0}
                # New token encountered
                else:
                    index[token] = {url:{"frequency":1, "bold": 0, "header": 0, "title": 0}}

    # The index has now been constructed
    return index