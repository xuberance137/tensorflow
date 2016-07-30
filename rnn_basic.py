import nltk
import csv
import itertools
import numpy as np

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

class RNNNumpy:

	def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
		#assign instance variables
		self.word_dim = word_dim
		self.hidden_dim = hidden_dim
		self.bptt_truncate = bptt_truncate
		#assign random values to network parameters
		#best initialization based on choice of activation function, in this case tanh(x)
		self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),(hidden_dim, word_dim))
		self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim),(hidden_dim, hidden_dim))
		self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim),(word_dim, hidden_dim))

    #Compute softmax values for each sets of scores in x
	def softmax(self, x):
		sf = np.exp(x)
		sf = sf/np.sum(sf, axis=0)
		return sf

	#return calculated outputs and hidden states
	#each o[t] is a vector of probabilities representing words in our vocabulary
	def forward_propagation(self, x):
		#total number of time steps
		T = len(x)
		#save all hidden states in s as these are needed later
		#additional element for initial hidden which is set to zero
		s = np.zeros((T+1, self.hidden_dim))
		#history set for first input sample
		s[-1] = np.zeros(self.hidden_dim)
		#saving output for each step
		o = np.zeros((T, self.word_dim))

		for t in np.arange(T):
			#indexing U by x[t] which is the same as multiplying U with x (binary 1 of K vector)
			s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
			prod = (self.V.dot(s[t])).flatten()
			o[t] = self.softmax(self.V.dot(s[t]))
		return [o, s]

	def predict(self, x):
		# forward propagation and return index of highest score
		o, s = self.forward_propagation(x)
		return np.argmax(o, axis = 1)

	#computing cross entropy
	def calculate_total_loss(self, x, y):
		L = 0
		#for each sentence
		for i in np.arange(len(y)):
			# dimensions of o is [num of words in sentence][vocabulary_size]
			o, s = self.forward_propagation(x[i])
			#only care about predictions of the "correct" words
			#we are indexing based on correct outputs y[i][words] to get probabilities for those values
			correct_word_predictions = o[np.arange(len(y[i])), y[i]]
			# add to the loss based on how off we are using the cross entropy metric
			L += -1 * np.sum(np.log(correct_word_predictions))
		return L

	def calculate_loss(self, x, y):
		# normalize loss by number of training samples
		N = np.sum(len(y_i) for y_i in y)
		return self.calculate_total_loss(x,y)/N

	def bptt(self, x, y):
		T = len(y)
		# Perform forward propagation
		o, s = self.forward_propagation(x)
		# We accumulate the gradients in these variables
		dLdU = np.zeros(self.U.shape)
		dLdV = np.zeros(self.V.shape)
		dLdW = np.zeros(self.W.shape)
		delta_o = o
		delta_o[np.arange(len(y)), y] -= 1.
		# For each output backwards...
		for t in np.arange(T)[::-1]:
		    dLdV += np.outer(delta_o[t], s[t].T)
		    # Initial delta calculation
		    delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
		    # Backpropagation through time (for at most self.bptt_truncate steps)
		    for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
		        # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
		        dLdW += np.outer(delta_t, s[bptt_step-1])              
		        dLdU[:,x[bptt_step]] += delta_t
		        # Update delta for next step
		        delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
		return [dLdU, dLdV, dLdW]


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
rev_word_lookup = dict([(index, word) for index, word in enumerate(index_to_word)])

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

np.random.seed(10)
model = RNNNumpy(vocabulary_size)
o, s = model.forward_propagation(X_train[10])
p = model.predict(X_train[10])
print len(X_train[10])
print "Model Prediction : ", p 
print "Model Prediction : ", [rev_word_lookup[index] for index in p]
# Probablity of prediction is 1/C. random loss = -(1/N)*N*log(1/C) = log(C)
print "Expected Loss for Random Predictions : ", np.log(vocabulary_size)
print "Actual Loss from RNN model : ", model.calculate_loss(X_train[:100], y_train[:100])




