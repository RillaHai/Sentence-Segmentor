#!/usr/bin/python
#-*- coding:utf-8 -*-
# Natural Language Toolkit: code_classification_based_segmenter

import nltk
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import codecs

###############################training file of treebank################################

sents = nltk.corpus.treebank_raw.sents()
tokens = []
boundaries = set()
boundaries_token = set()
offset = 0
for sent in sents:
    boundaries_token.add(sent[-1])
    tokens.extend(sent)
    offset += len(sent)
    boundaries.add(offset-1)

################################out-of domain test file##################################

out_domain = nltk.corpus.reuters.sents()
out_tokens = []	
out_boundaries = set()
#print out_domain[:3]
for od in out_domain:
    out_tokens.extend(od)
    offset += len(od)
    out_boundaries.add(offset-1) 
#print out_tokens[:5]	

################################## uyidata ###############################################
ts = []
b = set()
with open ("/home/work/ssCat/testss.en") as testfile:
    offset =0
    for line in testfile:
        line = nltk.word_tokenize(line.decode("utf-8"))
        if len(line) >= 1:
            ts.extend(line)
            offset += len(line)
            b.add(offset-1)	
#################################feature designing########################################

def punct_features(tokens, i):
    return {'prev-word': tokens[i-1],\
            'prev-2word': tokens[i-2],\
            'len-prev-2word': len(tokens[i-2]),\
            'punct': tokens[i],\
            'punct-capitalised':tokens[i][0].isupper(),\
            'prev-1letter-capi':tokens[i-1][0].isupper(),\
            'len-next-word':len(tokens[i+1][0]),\
            'next-word-capitalized': tokens[i+1][0].isupper() ,\
            'prev-2word-capitalized':tokens[i-2].isupper(),\
            'prev-2word-digit':tokens[i-2].isdigit(),\
            'next-word-digit':tokens[i+1].isdigit()}

###########################build featuressets from treebank###############################  

featuresets = [(punct_features(tokens, i), (i in boundaries))\
               for i in range(1, len(tokens)-1) if tokens[i] in boundaries_token]
               
######################build featuresets from out of domain file###########################

featuresets1 = [(punct_features(out_tokens, i), (i in out_boundaries))\
               for i in range(1, len(out_tokens)-1) if out_tokens[i] in boundaries_token]

#######################################  uyifeature  #####################################
featuresets2 = [(punct_features(ts, i), (i in b))\
               for i in range(1, len(ts)-1) if ts[i] in boundaries_token]               
###############################10 folds cross validation##################################  
             
#from sklearn import cross_validation
#a = 0.0                     OneClassSVM
#cv = cross_validation.KFold(len(featuresets), n_folds=10, indices=True, shuffle=False, random_state=None, k=None)
#for traincv, testcv in cv:
    #classifier = classifier = SklearnClassifier(LinearSVC()).train(featuresets[traincv[0]:traincv[len(traincv)-1]])
    #a += nltk.classify.util.accuracy(classifier, featuresets[testcv[0]:testcv[len(testcv)-1]])
#print 'accuracy:',a/10

################################devide training dev test #################################

size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
size_dev = int(len(train_set)*0.2)
train_set,dev_set = train_set[size_dev:], train_set[:size_dev]

#classifier = nltk.NaiveBayesClassifier.train(train_set) #0.95 0.81
#classifier = nltk.Conditio"/home/work/ssCat/try"nalExponentialClassifier.train(train_set) #0.987 0.81
#classifier = nltk.DecisionTreeClassifier.train(train_set) #0.972
#classifier = SklearnClassifier(BernoulliNB()).train(train_set) #0.975
classifier = SklearnClassifier(LinearSVC()).train(train_set) #0.99 #0.81

##################################usage of pipeline#########################################

pipeline = Pipeline([('tfidf', TfidfTransformer()),
                     ('chi2', SelectKBest(chi2, k=1000)),
                     ('nb', SVC())])
                     
####################################     print    ##########################################   
                  
#classifier = SklearnClassifier(pipeline).train(train_set) 
print nltk.classify.accuracy(classifier, dev_set)
print nltk.classify.accuracy(classifier, test_set)
print nltk.classify.accuracy(classifier, featuresets2)
#245640print classifier.show_most_informative_features()     #only naive bayes

################################       dump classifier        ##############################

#with open("en_sentence_classifier.pkl", "w") as clf_outfile:
        #pickle.dump(classifier, clf_outfile)
        
################################test for real raw sentence##################################

#with open("en_sentence_classifier.pkl", "r") as clf_infile:
        #classifier = pickle.load(clf_infile)
        
def segment_sentences(words):
    start = 0
    sents = []
    for i, word in enumerate(words):
        if word in '.?!;' and i != len(words)-1 and classifier.classify(punct_features(words, i)) == True:             #
            sents.append(words[start:i+1])
            start = i+1
    if start < len(words):
        sents.append(words[start:])
    return sents

teststring = "If you have your own collection of text files that you would\
 like to access using the above methods, you can easily load them with\
  the help of NLTK's PlaintextCorpusReader. Check the location of your \
  files on your file system; in the following example, we have taken \
  this to be the directory /usr/share/dict. Whatever the location, \
  set this to be the value of corpus_root."

wl = nltk.word_tokenize(teststring)
ss = segment_sentences(wl)
print len(ss)
