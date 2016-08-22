import nltk
import os
import sys
import re
import pprint
import nltk_util_functions as util_functions

#maximum number of characters in essay string
MAXIMUM_ESSAY_LENGTH = 2000
#maximum number of synonyms accessed
MAXIMUM_SYNS = 5

base_path = os.path.dirname(__file__)
sys.path.append(base_path)

def no_unicode(object, context, maxlevels, level):
    """ change unicode u'foo' to string 'foo' when pretty printing"""
    if pprint._type(object) is unicode:
        object = str(object)
    return pprint._safe_repr(object, context, maxlevels, level)

wordlist = ['broadcast', 'word']
synms = []
#wordlist.append('test')
#wordlist.append('word')
for word in wordlist:
    
    synonyms = []
    regex = r"_"
    pat = re.compile(regex)
    
    synset = nltk.wordnet.wordnet.synsets(word)

    for ss in synset:
        for swords in ss.lemma_names():
            synonyms.append(pat.sub(" ", swords.lower()))
    
    seen = set()
    seen_add = seen.add
    synms = [x for x in synonyms if x not in seen and not seen_add(x)]    
    
    pp = pprint.PrettyPrinter(indent = 4)
    pp.format = no_unicode
    #pp.pprint(synms)
            
text = "A natural language parser is a prograadam that works out the grammatical structure of sentences, for instance, which groups of words go together as phrases and which words are the subject or object of a verb. Probabilistic parsers use knowledge of language gained from hand-parsed sentences to try to produce the most likely analysis of new sentences. These statistical parsers still make some mistakes, but commonly work rather well. Their development was one of the biggest breakthroughs in natural language processing in the 1990s."
#from nltk.tokenize import sent_tokenize
sent_tokenize_list = nltk.tokenize.sent_tokenize(text)
word_tokenize_list = nltk.tokenize.word_tokenize(text)
pos_list = nltk.pos_tag(word_tokenize_list)
# print len(sent_tokenize_list)
# print sent_tokenize_list
# print len(word_tokenize_list)
# print word_tokenize_list
# print pos_list
#
# for item in pos_list:
#     print item, nltk.help.upenn_tagset(item[1])

# print word_tokenize_list
# porter_stemmer = nltk.stem.porter.PorterStemmer()
# wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
# for item in word_tokenize_list:
#     print item, porter_stemmer.stem(item), wordnet_lemmatizer.lemmatize(item)


# Clean text by removing non digit/work/punctuation characters
try:
    text = str(text.encode('ascii', 'ignore'))
except:
    text = (text.decode('utf-8', 'replace')).encode('ascii', 'ignore')

print text

# makes lower case and removes punctuations and digits
text = util_functions.sub_chars(text).lower()

print
print text

if(len(text) > MAXIMUM_ESSAY_LENGTH):
    text = text[0:MAXIMUM_ESSAY_LENGTH]
# Spell correct text using aspell
cleaned_text, spell_errors, markup_text = util_functions.spell_correct(text)

print
print "clean : ", cleaned_text

# Tokenize text
text_tokens = nltk.word_tokenize(cleaned_text)
print
print "Tokenized text : ", text_tokens

# Part of speech tag text
pos = nltk.pos_tag(nltk.word_tokenize(cleaned_text))
print
print "POS : ", pos

"""
Lemmatisation is closely related to stemming. The difference is that a stemmer operates 
on a single word without knowledge of the context, and therefore cannot discriminate between 
words which have different meanings depending on part of speech. However, stemmers are typically 
easier to implement and run faster, and the reduced accuracy may not matter for some applications.
"""


# Stem spell corrected text
porter = nltk.PorterStemmer()
por_toks = [porter.stem(w) for w in text_tokens]

print "stems : ", por_toks

print "synonyms : "
print
for word in text_tokens:
    synonyms = util_functions.get_wordnet_syns(word, MAXIMUM_SYNS)
    print word, synonyms



