#!/Users/gopal/projects/learning/tensorflow/venv/bin/python

import os
import pandas as pd
import pyprind
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

DEBUG_PRINT = False

def text_preprocessor(text):
	text = re.sub('<[^>]*>', '', text) #remove HTML markup
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text) #identify emoticons
	text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '') #convert to lower case and append emoticons to end
	return text

def tokenizer(text):
	return text.split()

def tokenizer_porter(text):
	porter = PorterStemmer()
	words = tokenizer(text)
	return [porter.stem(word) for word in words]

def remove_stopwords(text):
	stop = stopwords.words('english')
	return [w for w in tokenizer_porter(text) if w not in stop]

# pbar = pyprind.ProgBar(50000)
# labels = {'pos':1, 'neg':0}
# df = pd.DataFrame()

# for s in {'test', 'train'}:
# 	for l in ('pos', 'neg'):
# 		path = './data/aclImdb/%s/%s' %(s,l)
# 		for file in os.listdir(path):
# 			with open(os.path.join(path, file), 'r') as infile:
# 				txt = infile.read()
# 				df = df.append([[txt, labels[l]]], ignore_index=True)
# 				pbar.update()

# df.columns = ['review', 'setiment']

# np.random.seed(0)

# df = df.reindex(np.random.permutation(df.index))
# df.to_csv('./data/aclImdb/movie_data.csv', index=False)

if __name__ == '__main__':

	pd.set_option('display.max_columns', None)
	pd.set_option('display.max_rows', None)

	df = pd.read_csv('./data/aclImdb/movie_data.csv')

	if DEBUG_PRINT:

		df['review'] = df['review'].apply(text_preprocessor) #apply preprocesor to each review in the collection

		docs = df['review'][:2]
		#count = CountVectorizer()
		count = CountVectorizer(ngram_range=(2,2)) # for bigrams
		bag = count.fit_transform(docs)

		tfidf = TfidfTransformer()
		np.set_printoptions(precision=5)
		tfidf_val = tfidf.fit_transform(bag)

		print docs[0]
		print count.vocabulary_
		print bag.toarray()
		print
		print tfidf_val.toarray()
		print
		print text_preprocessor("</a> This is :) is :( a test :-) !")
		print
		print tokenizer_porter(docs[0])
		print
		print remove_stopwords(docs[0])




