import os
import sys
import json
import nltk
from bs4 import BeautifulSoup

# Download the NLTK Punkt tokenizer and the WordNet lemmatizer
nltk.download("punkt")
nltk.download("wordnet")

def createIndex():
    # Dictionary to store the index
    index = {}

    # The directory containing the offline corpus of webpages
    corpus_dir = sys.argv[1]

    # Iterate over all the files in the corpus directory
    jsonData = json.load(open(os.path.join(corpus_dir, "bookkeeping.json")))    

    for filename in jsonData:
        location = filename.split("/")
        url = jsonData[filename]
        print(location)
        file_path = os.path.join(corpus_dir, location[0], location[1])
        with open(file_path, "r", encoding = "utf-8") as f:
            contents = f.read()

            # Use BeautifulSoup to parse the HTML contents of the file
            soup = BeautifulSoup(contents, "html.parser")

            # Get the text from the HTML contents
            text = soup.get_text()

            # Tokenize the text using the NLTK tokenizer
            tokens = nltk.tokenize.word_tokenize(text.lower())

            # Lemmatize the tokens using the NLTK WordNetLemmatizer
            lemmatized_tokens = [nltk.stem.WordNetLemmatizer().lemmatize(token) for token in tokens]

            # Add the tokens to the index
            for token in lemmatized_tokens:
                if token in index:
                    if url in index[token]:
                        index[token][url]["frequency"] += 1
                    else:
                        index[token][url] = {"frequency":1, "bold": 0, "header": 0, "title": 0}
                else:
                    index[token] = {url:{"frequency":1, "bold": 0, "header": 0, "title": 0}}

    # The index has now been constructed
    return index