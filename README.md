# TopicModeling_Classifier_Recc

**Follow-up to Section 3 Walkthrough:**

3. As I noted in the .ipynb, the classifier results place a privacy label to an article based on fuzzy lines between aggregation and privacy policy. 

4. The issue of multilabels also needs to be addressed. http://scikit-learn.org/dev/modules/multiclass.html


**Follow-up to Section 1/2 Walkthrough:**

1. SYSTEM: 
Ideally you can view the walkthrough in an ipython notebook. For instance, pandas, an ipython tool is heavily used in the walkthrough

2.  TOP TOPICS FUNCION:
Going forward, you will likely get more interesting words for a topic if you do the following:

* use an alternative cleaning technique to "lemmatized"
* use the  gensim  "filter" option to ignore words that are too infrequent or frequent among the corpus

