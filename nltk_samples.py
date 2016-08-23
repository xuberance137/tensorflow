import nltk
import os
import sys
import re
import pprint
import nltk_util_functions as util_functions
from tqdm import tqdm
import random

#verbosity
PRINT_TEXT_PIPE_STATUS = False
#maximum number of characters in text string
MAXIMUM_ESSAY_LENGTH = 2000
#maximum number of synonyms accessed
MAXIMUM_SYNS = 10
MINIMUM_SYN_FILTER = 7
NUM_STRINGS_GEN = 50
NUM_SYN_SELECT = 2
BREAKOUT = True

base_path = os.path.dirname(__file__)
sys.path.append(base_path)
            
def text_pipeline(text):

    # intialize stemmer and lemmatizer
    porter_stemmer = nltk.stem.porter.PorterStemmer()
    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
    # Clean text by removing non digit/work/punctuation characters
    try:
        text = str(text.encode('ascii', 'ignore'))
    except:
        text = (text.decode('utf-8', 'replace')).encode('ascii', 'ignore')
    # makes lower case and removes punctuations and digits
    text = util_functions.sub_chars(text).lower()
    # limit length
    if(len(text) > MAXIMUM_ESSAY_LENGTH):
        text = text[0:MAXIMUM_ESSAY_LENGTH]
    # Spell correct text using aspell
    cleaned_text, spell_errors, markup_text = util_functions.spell_correct(text)
    # Tokenize text
    text_tokens = nltk.word_tokenize(cleaned_text)
    # Part of speech tag text
    pos = nltk.pos_tag(nltk.word_tokenize(cleaned_text))
    # Stem spell corrected text
    porter = nltk.PorterStemmer()
    por_toks = [porter.stem(w) for w in text_tokens]  
    synonyms_list = []  
    for word in text_tokens:
        synonyms = util_functions.get_wordnet_syns(word, MAXIMUM_SYNS)
        synonyms_list.append(synonyms)

    """
    Lemmatisation is closely related to stemming. The difference is that a stemmer operates 
    on a single word without knowledge of the context, and therefore cannot discriminate between 
    words which have different meanings depending on part of speech. However, stemmers are typically 
    easier to implement and run faster, and the reduced accuracy may not matter for some applications.
    """
    if PRINT_TEXT_PIPE_STATUS:
        print
        print text
        print
        print "clean : ", cleaned_text
        print
        print "Tokenized text : ", text_tokens
        print
        print "POS : ", pos
        print
        print "stems : ", por_toks
        print
        print "synonyms : "
        print
        for word in text_tokens:
            synonyms = util_functions.get_wordnet_syns(word, MAXIMUM_SYNS)
            print word, synonyms
        print
        print "Comparing Stemmer and Lemmatizers : "
        print
        for word in text_tokens:
            print word, porter_stemmer.stem(word), wordnet_lemmatizer.lemmatize(word)
        print
        print len(text_tokens)
        print len(synonyms_list)
        print 
        print synonyms_list

    return cleaned_text, text_tokens, synonyms_list


def generative_pipeline(cleaned_text, synonyms_list, generator_count):

    generated_text = []
    text_tokens = nltk.word_tokenize(cleaned_text)

    print len(text_tokens)
    print len(cleaned_text.split(" "))

    for index in range(len(text_tokens)):
        print index
        if len(synonyms_list[index]) > MINIMUM_SYN_FILTER:
            print text_tokens[index], synonyms_list[index]
            list_of_syn_words = random.sample(synonyms_list[index], NUM_SYN_SELECT)
            print list_of_syn_words
            for item in list_of_syn_words:
                gen_text_tokens = text_tokens
                gen_text_tokens[index] = item
                generated_text_sample = " ".join(gen_text_tokens)
                generated_text.append(generated_text_sample) 

    return generated_text
    

### MAIN FUNCTION ###
if __name__ == '__main__':

    text_string = "A natural language parser is a prograadam that works out the grammatical structure of sentences, for instance, which groups of words go together as phrases and which words are the subject or object of a verb. Probabilistic parsers use knowledge of language gained from hand-parsed sentences to try to produce the most likely analysis of new sentences. These statistical parsers still make some mistakes, but commonly work rather well. Their development was one of the biggest breakthroughs in natural language processing in the 1990s."

    cleaned_text, text_tokens, synonyms_list = text_pipeline(text_string)

    new_text = generative_pipeline(cleaned_text, synonyms_list, NUM_STRINGS_GEN)

    for text in new_text:
        print
        print text

    print 
    print len(new_text)







