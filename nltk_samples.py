import nltk
import os
import sys
import re
import pprint

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
            
text = "A natural language parser is a program that works out the grammatical structure of sentences, for instance, which groups of words go together as phrases and which words are the subject or object of a verb. Probabilistic parsers use knowledge of language gained from hand-parsed sentences to try to produce the most likely analysis of new sentences. These statistical parsers still make some mistakes, but commonly work rather well. Their development was one of the biggest breakthroughs in natural language processing in the 1990s."
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

print word_tokenize_list
porter_stemmer = nltk.stem.porter.PorterStemmer()
wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
for item in word_tokenize_list:
    print item, porter_stemmer.stem(item), wordnet_lemmatizer.lemmatize(item)