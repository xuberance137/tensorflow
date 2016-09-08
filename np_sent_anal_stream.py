#!/Users/gopal/projects/learning/tensorflow/venv/bin/python

import os
import pandas as pd
import pyprind
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier

DEBUG_PRINT = False

def text_preprocessor(text):
	text = re.sub('<[^>]*>', '', text) #remove HTML markup
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text) #identify emoticons
	text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '') #convert to lower case and append emoticons to end
	return text

def tokenizer(text):
	text = re.sub('<[^>]*>', '', text) #remove HTML markup
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text) #identify emoticons
	text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '') #convert to lower case and append emoticons to end
	stop = stopwords.words('english')
	tokenized = [w for w in text.split() if w not in stop]
	return tokenized

#generator for returning one docuemnt at a time
def stream_docs(path):
	with open(path, 'r') as csv:
		next(csv) #skip header
		for line in csv:
			text, label = line[:-3], int(line[-2])
			yield text, label

#takes a document stream and returns a particular number of documents 
def get_minibatch(doc_stream, size):
	docs, y = [], []
	try:
		for _ in range(size):
			text, label = next(doc_stream)
			docs.append(text)
			y.append(label)
	except StopIteration:
		return None, None

	return docs, y



if __name__ == '__main__':

	pd.set_option('display.max_columns', None)
	pd.set_option('display.max_rows', None)

	# https://sites.google.com/site/murmurhash/
	# replacing TFIDF vectorizer (requires entire all trainign feature vectors in memory) with Murmur hash vectorizer
	vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=tokenizer)
	clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
	doc_stream = stream_docs(path='./data/aclImdb/movie_data.csv')
	NUM_ITER = 45

	pbar = pyprind.ProgBar(NUM_ITER)
	classes = np.array([0, 1])

	for _ in range(NUM_ITER):
		X_train, y_train = get_minibatch(doc_stream, size=1000) #getting 1000 docs at a time as a minibatch
		if not X_train:
			break
		X_train = vect.transform(X_train)
		clf.partial_fit(X_train, y_train, classes=classes)
		pbar.update()

	X_test, y_test = get_minibatch(doc_stream, size=5000)
	X_test = vect.transform(X_test)

	# Model performance on out of sample data
	print 'Accuracy : ', clf.score(X_test, y_test)

	#updating the model with the new test data
	clf = clf.partial_fit(X_test, y_test)














