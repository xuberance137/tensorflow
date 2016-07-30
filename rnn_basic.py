import nltk
import csv
import itertools
import numpy as np

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Reading data and appending with sentence Start/Stop tokens
print "Reading CSV file..."
with open('data/reddit-comments-2015-08.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print "Number of parsed sentences : ", len(sentences)
    
# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count word frequency
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Number of unique word tokens : ", len(word_freq)

vocab = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)

word_to_index = dict([(word, index) for index, word in enumerate(index_to_word)])

print "Vocabulary Size : ", vocabulary_size
print "Most Frequent word : ", vocab[0][0], vocab[0][1]
print "Least Frequent word : ", vocab[-1][0], vocab[-1][1]

# Replace words not in dictionary wtih Unknown Token
for index, sent in enumerate(tokenized_sentences):
	tokenized_sentences[index] = [w if w in word_to_index else unknown_token for w in sent]

print "Example Sentence : ", sentences[0]
print "Example pre-processed Sentence : ", tokenized_sentences[0]
print "Example codified Sentence : ", np.asarray([word_to_index[w] for w in tokenized_sentences[0]])

# Creating training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
# for k, v in word_to_index.iteritems():
# 	print k, v
print "Training input :", X_train[10]
print "Training output :", y_train[10]


