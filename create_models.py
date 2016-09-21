#! /usr/bin/python3
#      
#   Create models from corpuses in ./CORPUSES  
#   Get cosine similarity between two articles modeled as topics    
#  
#   v1.0   
#   

import logging, os ,sys, re, itertools, pandas
import nltk
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import TextCorpus, MmCorpus, Dictionary
import numpy as np
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.DEBUG# ipython sometimes messes up the logging setup; restore

from nltk.stem import WordNetLemmatizer
from create_corpuses import tokenize


#################### COMPARE TWO ARTICLES BY TOPICS
def lda_tfidf_lemmatized_for_unseen(text, privacy_lemmatized_dict, lda_tfidf_lemmatized_model):
    bow_vector = privacy_lemmatized_dict.doc2bow(tokenize(text, "lemmatized"))
    lda_vector = lda_tfidf_lemmatized_model[bow_vector]
    return lda_vector

def topic_relev_score(art1, art2, STORY_CONTENT_DIR, privacy_lemmatized_dict, lda_tfidf_lemmatized_model): 
    """ USE stacked LDA_TFIDF transformation since it returned better results in  trial"""
    with open(os.path.join(STORY_CONTENT_DIR,art1), 'r') as input: g1 = input.read();
    with open(os.path.join(STORY_CONTENT_DIR,art2), 'r') as input: g2 = input.read();
    part1 = [lda_tfidf_lemmatized_for_unseen(g1, privacy_lemmatized_dict, lda_tfidf_lemmatized_model)]
    part2 = [lda_tfidf_lemmatized_for_unseen(g2, privacy_lemmatized_dict, lda_tfidf_lemmatized_model)]
    return (np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip(part1, part2)]))


#################### GET 10 TOP TOPICS IN A CORPUS 
def panda_topics(model_file, corpus_file,no=10):
    import sys,gensim,pandas
    model = gensim.models.LdaModel.load(model_file) # Revive the LDA model 
    corpus = gensim.corpora.MmCorpus(corpus_file)   # Revive a corpus
    
    topics = gensim.matutils.corpus2dense(model[corpus], num_terms=model.num_topics)
    weight = topics.sum(1)
    max_topic = weight.argmax()
    topics = model.show_topic(max_topic, no)
    rev = weight.argsort()[::-1][0:10]
    _pd =[ model.print_topic(i, no) for i in rev]
    df2 = pandas.DataFrame(_pd)
    pandas.set_option('max_colwidth', 400)
    df2.index=[ '1st Most Discussed Topic (MDT)','2nd MDT', '3rd MDT', '4th MDT','5th MDT','6th MDT', '7th MDT', '8th MDT','9th MDT','10th MDT']
    df2.rename(columns={1:'Distribution of Words over a Topic'}, inplace=True)
    return df2

if __name__ == '__main__':
	
	CORPUS_NAME = sys.argv[1]
	CORPUS_DIR = "./CORPUSES"
	MODELS_DIR = "./MODELS"
	try:
	    NO_TOP = sys.argv[2]
	except:
	    logging.info("N0_TOP is now set to 100 be default")
	    NO_TOP = 100
	
	####################  Working with persisted corpus and dictionary
	privacy_lemmatized_corpus = MmCorpus(os.path.join(CORPUS_DIR,CORPUS_NAME)+ "_lemmatized.mm")  # Revive a corpus
	privacy_lemmatized_dict = Dictionary.load (os.path.join(CORPUS_DIR, CORPUS_NAME) +"_lemmatized.dict")  # Load a dictionary
	
	
	#################### CREATE LEMMATIZED MODEL
	""" num_of_topics = 100 was chosen as default value"""
	"""
	FROM GENSIM DOCS:
	    "Transformation can be stacked. For example, here well train a TFIDF model,
	    and then train Latent Semantic Analysis on top of TFIDF... 
	
	    The TFIDF transformation only modifies feature weights of each word. Its
	    input and output dimensionality are identical (=the dictionary size)."
	"""
	# TRAIN TFIDF MODEL
	tfidf_lemmatized_model = gensim.models.TfidfModel(privacy_lemmatized_corpus, id2word=privacy_lemmatized_dict)
	
	# TRAIN LDA[TFIDF] MODEL
	lda_tfidf_lemmatized_model = gensim.models.LdaModel(tfidf_lemmatized_model[privacy_lemmatized_corpus], id2word=privacy_lemmatized_dict, num_topics=NO_TOP, passes=40)
	
	# TRAIN LDA MODEL
	lda_lemmatized_model = gensim.models.LdaModel(privacy_lemmatized_corpus, num_topics=NO_TOP, id2word=privacy_lemmatized_dict, passes=40)
	
	#################### SAVE LEMMATIZED MODEL
	lda_tfidf_lemmatized_model.save(os.path.join(MODELS_DIR, CORPUS_NAME) + "_lemmatized_tfidf_lda.model")
	lda_lemmatized_model.save(os.path.join(MODELS_DIR, CORPUS_NAME) + "_lemmatized_lda.model")  
	
	
        #print(lda_tfidf_lemmatized_model.print_topics(-1))  
