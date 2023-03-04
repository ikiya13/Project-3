import itertools
import math
import os
import sys
import json
import numpy as np
from numpy.linalg import norm
import logging
import nltk
from bs4 import BeautifulSoup
from nltk import WordPunctTokenizer

# Download the NLTK Punkt tokenizer and the WordNet lemmatizer
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

wordpunct_tokenize = WordPunctTokenizer().tokenize

KSCORES = 20

tagWeights = {
    "title": 10,
    "h1": 8,
    "h2": 7,
    "h3": 6,
    "h4": 5,
    "h5": 4,
    "h6": 3,
    "strong": 2,
    "em": 2,
    "b": 2,
    "i": 1.5,
    "u": 1.5,
    "a": 0.5,
}


def createIndex():
    # Dictionary to store the index
    # { token : { url : { "frequency": int, "tfidf": float, "weight": float}}}
    global index
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
        with open(file_path, "r", encoding="utf-8") as f:
            contents = f.read()

            # parse the HTML contents of the file
            soup = BeautifulSoup(contents, "lxml")

            # Get the text
            text = soup.get_text()

            # Tokenize the text
            tokens = wordpunct_tokenize(text.lower())

            # Lemmatize the tokens
            lemmatized_tokens = []
            for token in tokens:
                if token not in nltk.corpus.stopwords.words("english") and token.isalpha():
                    lemmatized_tokens.append(nltk.stem.WordNetLemmatizer().lemmatize(token, get_wordnet_pos(token)))

            # Add the tokens to the index
            for token in lemmatized_tokens:
                # If the token already exists
                if token in index:
                    # If the URL is already included
                    if url in index[token]:
                        index[token][url]["frequency"] += 1
                    # If we are at a new URL
                    else:
                        index[token][url] = {"frequency": 1, "weight": 1}
                # New token encountered
                else:
                    index[token] = {url: {"frequency": 1, "weight": 1}}
            # add weights
            calcWeights(soup, url)

    # The index has now been constructed, add TF-IDF
    index = addTFIDF(index, len(jsonData))

    # return
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


# Function to add weights to words based on what tag they appear in
def calcWeights(soup, url):
    # Get all the tags on the page and iterate through them
    for htmlTag in soup.find_all():
        # Get text from given tag
        text = htmlTag.get_text()

        # Tokenize the text
        tokens = wordpunct_tokenize(text.lower())

        for token in tokens:
            # Lemmatize
            tokenLemma = nltk.stem.WordNetLemmatizer().lemmatize(token, get_wordnet_pos(token))

            # Add weight
            if tokenLemma in index:
                if url in index[tokenLemma]:
                    index[tokenLemma][url]["weight"] += tagWeights.get(htmlTag.name, 0)


# Function to add TF-IDF values to each word document pair
def addTFIDF(index, corpusLen):
    # Calculate TF-IDF
    for token in index:
        # { token : { url : { "frequency": int, "bold": int, "header": int, "title": int}}}
        for url in index[token]:
            # Using the definition provided in class, we are to use the raw frequency of the word and not the relative frequency stored in "tf"
            tf = index[token][url]["frequency"]
            idf = math.log(corpusLen / len(index[token]))
            tfidf = (1 + math.log(tf)) * idf

            # add TF-IDF
            index[token][url]["tf-idf"] = tfidf

    # TF-IDF has been added
    return index


# Query the index
def searchIndex(tokens, index):

    if len(tokens) == 2:
        # 2 words
        # n grams
        # cosine similarity method
        token1 = tokens[0]
        token2 = tokens[1]
        # testing tokens
        try:
            postings1 = index[token1]
        except (KeyError):
            try:
                postings2 = index[token2]
                return searchIndex(token2, index)
            except (KeyError):
                print(f"There are no results for this search")
                return
        try:
            postings2 = index[token2]
        except (KeyError):
            return searchIndex(token1, index)

        # making new dict with combined tfidfs
        both_tokens = dict()

        for url in postings1:
            if url in postings2:
                both_tokens[url] = postings1[url]["tf-idf"] + postings2[url]["tf-idf"]
            else:
                both_tokens[url] = postings1[url]["tf-idf"]

        for url in postings2:
            if url not in both_tokens:
                both_tokens[url] = postings2[url]["tf-idf"]

        # search
        ranked_tfidf = sorted(both_tokens.items(), key=lambda x: x[1], reverse=True)
        print(f"Number of results: {len(both_tokens)}")
        if len(both_tokens) > 20:
            ranked_tfidf = ranked_tfidf[:20]
        print(f"Top {len(ranked_tfidf)} results: ")
        for tup in ranked_tfidf:
            print(tup[0])

    else:
        # 1 keyword search for M1
        token = tokens[0]
        try:
            postings = index[token]
        except (KeyError):
            print(f"There are no results for this search")
            return

        # version 1: return top 20 postings ranked by tfidf
        ranked_tfidf = sorted(postings.items(), key=lambda x: x[1]["tf-idf"], reverse=True)
        print(f"Number of results: {len(postings)}")
        if len(postings) > 20:
            ranked_tfidf = ranked_tfidf[:20]
        print(f"Top {len(ranked_tfidf)} results: ")
        for tup in ranked_tfidf:
            print(tup[0])


def search(index, query):
    #clean query
    # Lemmatize the tokens
    queryLemmas = []
    for token in query:
        if token not in nltk.corpus.stopwords.words("english") and token.isalpha():
            queryLemmas.append(nltk.stem.WordNetLemmatizer().lemmatize(token, get_wordnet_pos(token)))
    
    # Get the set of URLs for each token
    postings = [set(index[token]) for token in queryLemmas]

    # Find the intersection of all URL sets
    intersection = set.intersection(*postings)

    #calculate TFIDFs of query
    queryTFIDF = []

    for token in queryLemmas:
        #calculate tfidf for each token in the query
        tf = 1 + math.log(queryLemmas.count(token))
        idf = math.log(len(json.load(open("bookkeeping.json"))) / len(index[token]))
        queryTFIDF.append(tf * idf)
    
    #retrieve TFIDFs for each url in intersection
    documentsTFIDF = {}
    for url in intersection:
        documentsTFIDF[url] = []

        for token in queryLemmas:
            documentsTFIDF[url].append(index[token][url]["tf-idf"])    

    #calculate the cosine similarity scores between the query tfidf and the document tfidf
    cosineScores = {}
    for url in intersection:
        #calculate cosine
        # print("Dot product of " + str(queryTFIDF) + " and " + str(documentsTFIDF[url]) + " is: \t" + str(np.dot(queryTFIDF, documentsTFIDF[url])))
        # print("Norms of " + str(queryTFIDF) + " and " + str(documentsTFIDF[url]) + " are: \t" + str(norm(queryTFIDF)) + " and " + str(norm(documentsTFIDF[url])))
        cosineScores[url] = np.dot(queryTFIDF, documentsTFIDF[url]) / (norm(queryTFIDF) * norm(documentsTFIDF[url]))

        #add in the weights from the HTML tags
        for token in queryLemmas:
            cosineScores[url] += index[token][url]["weight"]
    
    #sort cosines
    cosineScores = dict(sorted(cosineScores.items(), key=lambda item: item[1], reverse = True))

    #get top k
    cosineScores = dict(itertools.islice(cosineScores.items(), KSCORES))
    
    return cosineScores



# python main.py /Users/akbenothman/Desktop/WEBPAGES_RAW
