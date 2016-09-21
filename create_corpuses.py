#! /usr/bin/python3
#   
#  Create corpus to create gensim topic  models:
#  1. lemmatized
#  
#  v1.0:
#       Based on the following Gensim tutorial:
#       http://radimrehurek.com/topic_modeling_tutorial/2%20-%20Topic%20Modeling.html     
#
# TODO:     Code to provide corpus cleaned via collocation and NER  techniques needs to be
#           merged in later version due to time constraints 

import logging, os ,sys, re, itertools
import nltk
import gensim
from gensim.parsing.preprocessing import STOPWORDS
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.DEBUG# ipython sometimes messes up the logging setup; restore

from nltk.stem import WordNetLemmatizer
wl =WordNetLemmatizer()


################### PURPOSE: #####################################################
################### Yield each article from a directory of .txt files, as a `(title, [tokens])` 2-tuple.
"""
@param STORY_CONTENT_DIR: directory of text files, or a text file, or a string to clean
NOTE: Need a directory of .txt files; no nested directories
"""
def iter_db_articles(STORY_CONTENT_DIR,  style, care_dupl=False):
    dup_E = get_dup(STORY_CONTENT_DIR,care_dupl)  
    if os.path.isdir(STORY_CONTENT_DIR):
        for art_name in os.listdir(STORY_CONTENT_DIR):
            filename = os.path.join(STORY_CONTENT_DIR, art_name)
            if not  filename.split('.')[-1] == "txt":
                print(filename) 
                logging.warning("This above file in STORY_CONTENT_DIR does not have .txt extension") 
            if filename not in dup_E:
	            try:
	                with open(filename, 'r') as input:    
	                    art_text = input.read()
	                    tokens = tokenize(art_text, style)
	                    yield art_name, tokens
	            except Exception as e: 
	                print(e)
	                logging.warning("ITER_DB_ARTILCES: something went wrong opening .txt files in this directory. Do not have nested directories.")
	                pass

################### PURPOSE #####################################################
################### Tokenize text before cleaning
"""
(note) FROM GENSIM DOC, gensim.utils.tokenize:
    "Iteratively yield tokens as unicode strings, removing accent marks 
    and optionally lowercasing the unidoce string by assigning
    True to one of the parameters, lowercase, to_lower, or lower."
"""
def tokenize(text,style):
    if style == "lemmatized":
        _ = list(gensim.utils.tokenize(text, lower=True))
        return  [ wl.lemmatize(word) for word in _  if word not in STOPWORDS and len(word) >2]



################### PURPOSE #####################################################
################### Does directory of text  files contain the same story multiple times?
"""
!!This function is not scalable; if the db cannot be held in memory, leave  class 
  flag, care_dupl to its default value of False
"""
def get_dup(STORY_CONTENT_DIR, care_dupl=False):
        _dup_E = check_duplicate_content(STORY_CONTENT_DIR) if care_dupl and os.path.isdir(STORY_CONTENT_DIR) else []
        if care_dupl: 
             logging.info("ITER_DB_ARTILCES():duplicate files below not added twice to content_dump of script:  ")
             for i in _dup_E:
                 print ("******DUPLICATE: " +i.split('/')[-1])
        return _dup_E


def check_duplicate_content(STORY_CONTENT_DIR): 
    article_storage = {}
    dup_E =[]
    for filename in os.listdir(STORY_CONTENT_DIR):
        filename = STORY_CONTENT_DIR+filename if STORY_CONTENT_DIR[-1] == '/' \
                                              else os.path.join(STORY_CONTENT_DIR, filename)
        try:
            with open(filename, 'r') as input:    
                art_text = input.read()
                art_size = os.path.getsize(filename) 
                if art_size in article_storage:
                   logging.info("CHECK_DUPLICATE_CONTENT(): possible dup_E")
                   dup_E  += [filename for i in range(len(article_storage[art_size]))   if article_storage[art_size][i] == [art_text] ] 
                   if not dup_E:
                       article_storage[art_size].append([art_text])
                else:
                    article_storage[art_size] = [[art_text]]
        except:
            logging.warning("Difficulty reading in article: ,", filename)
            pass
    return dup_E


################### PURPOSE #####################################################
################### Class for corpus object that will be serialized for LDA model
class BOWCorpus(object):
    def __init__(self, dump_file, dictionary, style="lemmatized", care_dupl=False, clip_docs=None):
        self.dump_file = dump_file
        self.dictionary = dictionary
        self.care_dupl= care_dupl
        self.style= style
        self.clip_docs = clip_docs
        print("style is", self.style)
    
    def __iter__(self):
        self.titles = []
        for title, tokens in itertools.islice(iter_db_articles(self.dump_file, self.style, self.care_dupl), self.clip_docs):
            self.titles.append(title)
            yield self.dictionary.doc2bow(tokens)
    
    def __len__(self):
        return self.clip_docs




if __name__ == '__main__':

    STORY_CONTENT_DIR= sys.argv[1]
    CORPUS_NAME = sys.argv[2]
    CORPUS_LOCATION = "./CORPUSES"
    _CARE_DUPL = False
    try: 
        if (sys.argv[3] =="--nodup"): _CARE_DUPL = True
    except: pass

################## STREAM CORPUS 
    doc_stream_lemmatized =  (tokens for _, tokens in iter_db_articles(STORY_CONTENT_DIR, "lemmatized"))

################## CREATE GENSIM DICTIONARY 
    id2word_lemmatized = gensim.corpora.Dictionary(doc_stream_lemmatized)

################## SAVE GENSIM DICTIONARY 
    id2word_lemmatized.save(os.path.join(CORPUS_LOCATION ,CORPUS_NAME) + "_lemmatized.dict")

################## CREATE BAG OF WORD VECTORS 
    bow_corpus_lemmatized = BOWCorpus(STORY_CONTENT_DIR, id2word_lemmatized, style = "lemmatized", care_dupl=_CARE_DUPL)

################## SAVE BAG OF WORD VECTORS 
    gensim.corpora.MmCorpus.serialize(os.path.join(CORPUS_LOCATION, CORPUS_NAME ) + '_lemmatized.mm', bow_corpus_lemmatized)

    #print(bow_corpus_lemmatized.titles)
