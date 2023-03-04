from flask import Flask, render_template, request
import pickle
import logging
from pathlib import Path
from urllib.parse import urljoin
from indexer import createIndex, search

app = Flask(__name__)

# create logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# call indexer passing command line argument
path = Path("indexFile.pickle")
if not path.is_file():
    logging.info("Creating index.")
    index = createIndex()
    # save to file
    logging.info("Saving to file.")
    pickle.dump(index, open("indexFile.pickle", "wb"))
else:
    # read from file
    logging.info("Reading from file.")
    index = pickle.load(open("indexFile.pickle", "rb"))

# Analytics
doc_ID = set()
for token, layer1 in index.items():
    for url in layer1:
        doc_ID.add(url)

logging.info(f"Number of unique document id's in index: {len(doc_ID)}")
logging.info(f"Number of unique words in index: {len(index)}")

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def search_route():
    if request.method == 'POST':
        query = request.form['query'].lower().split()
        results = search(index, query)
        return render_template('results.html', results=results, urljoin=urljoin)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
