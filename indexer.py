import math
import os
import sys
import json
import logging
import nltk
from bs4 import BeautifulSoup
import pickle

# Download the NLTK Punkt tokenizer and the WordNet lemmatizer
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

#create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

def createIndex():
    # Dictionary to store the index
    # index1 STRUCTURE
    # { token : { url : { "frequency": int, "bold": int, "header": int, "title": int}}}
    index = {}

    # Regex expression
    regexWord = nltk.tokenize.RegexpTokenizer(r"\w+")

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
            # tokens = nltk.tokenize.word_tokenize(text.lower())
            tokens = regexWord.tokenize(text.lower())

            # Lemmatize the tokens
            lemmatized_tokens = []
            for token in tokens:
                if token not in nltk.corpus.stopwords.words("english"):
                    lemmatized_tokens.append(nltk.stem.WordNetLemmatizer().lemmatize(token, get_wordnet_pos(token)))

            # Add the tokens to the index
            for token in lemmatized_tokens:
                # If the token already exists
                if token in index:
                    # If the URL is already included
                    if url in index[token]:
                        index[token][url]["frequency"] += 1
                        index[token][url]["tf"] += (1 / len(lemmatized_tokens))
                    # If we are at a new URL
                    else:
                        index[token][url] = {"frequency": 1, "tf": (1 / len(lemmatized_tokens)), "bold": 0, "header": 0, "title": 0}
                # New token encountered
                else:
                    index[token] = {url:{"frequency": 1, "tf": (1 / len(lemmatized_tokens)), "bold": 0, "header": 0, "title": 0}}

    # The index has now been constructed, add TF-IDF
    index = addTFIDF(index, len(jsonData))

    #return
    return index


# Function to get the POS of a word for lemmatization
# Code taken from https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}
    
    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)



def addTFIDF(index, corpusLen):
    # Calculate TF-IDF
    for token in index:
        # { token : { url : { "frequency": int, "bold": int, "header": int, "title": int}}}
        for url in index[token]:
            # Using the definition provided in class, we are to use the raw frequency of the word and not the relative frequency stored in "tf"
            tf = index[token][url]["frequency"]
            idf = math.log(corpusLen / len(index[token]))
            tfidf = (1 + math.log(tf)) * idf

            #add TF-IDF
            index[token][url]["tf-idf"] = tfidf
    
    #TF-IDF has been added
    return index


def searchIndex(tokens, index):

    if len(tokens) == 2:
        # 2 words
        # n grams
        # cosine similarity method
        token1 = tokens[0]
        token2 = tokens[1]
        postings1 = index[token1]
        postings2 = index[token2]
        pass
    else:
        # 1 keyword search for M1
        token = tokens[0]
        postings = index[token]

        # version 1: return top 20 postings ranked by tfidf
        ranked_tfidf = sorted(postings.items(), key = lambda x: x[1]["tf-idf"], reverse = True)
        if len(postings) > 20:
            ranked_tfidf = ranked_tfidf[:20]
        for tup in ranked_tfidf:
            print(tup[0])
        


# python main.py /Users/akbenothman/Desktop/WEBPAGES_RAW