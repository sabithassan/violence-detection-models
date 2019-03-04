import sys
import os
import pandas as pd
import numpy as np
import dill as pickled
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer



def print_usage():
    ''' Usage function. the usage is shown when the file is run without arguments/ wrong number of arguments'''
    
    print (
    '''Usage:
    python3 violence-detection-train.py  <input-file> <n>

    <input-file> is the training file. input-file should contain three tab separated columns, first column can be anything
    (could contain ID for example) second column contains the header "text" in first row, followed by the arabic texts, 
    third column contains the header "level" followed  by level of violence for the corresponding text in second column.

    <n> denotes n for word -n grams. Setting n = 1 will create word unigram, setting n = 2 will create word bigram etc.
    <n> must be greater than 0
        
    example Usage:    
    python3 violence-detection-train.py  input/tweet-train.tsv 2
        ''')


def get_word_n_gram_feats (input, n):
    ''' returns word n-gram features for input'''

    tokenizer = tokenizer=lambda x:x.split(' ')
    vectorizer = TfidfVectorizer(lowercase=False, ngram_range= n, analyzer='word',tokenizer = tokenizer)
    X_t = vectorizer.fit_transform(input)
    return X_t, vectorizer


def train_classifier (classifier, input, labels):
    ''' trains classifier on input and labels'''

    clf = OneVsRestClassifier(classifier, n_jobs = -1)
    clf.fit(input, labels)
    return clf




if __name__ == '__main__':
    ''' First loads the training data, then creates word n-gram features, and then trains
        Multinomial Naive Bayes and Linear SVM classifiers.
    '''

    # checks number of arguments
    if (len (sys.argv) != 3):
        print_usage()
        exit()

    # arguments
    filename = sys.argv[1]
    n_gram = int(sys.argv[2])

    # reads file and separates text and labels
    readfile = pd.read_csv(filename, sep = '\t', index_col=0)    
    train_input = readfile['text'].values
    train_labels = readfile['level'].values

    print ("Number of instances", train_input.shape[0])

    # gets word n-gram features
    n_gram_features, vectorizer = get_word_n_gram_feats(train_input, (0,n_gram))
    
    ## trains classifiers. this can be changed to try different classifiers
    MNB_classifier = train_classifier(MultinomialNB(), n_gram_features, train_labels)
    MNB_models = [MNB_classifier, vectorizer]
    print ("Multinomial Naive Bayes training completed")

    SVM_classifier = train_classifier(LinearSVC(), n_gram_features, train_labels)
    SVM_models = [SVM_classifier, vectorizer]
    print ("Linear SVM training completed")

    
    # creates directory for models
    if not os.path.exists('models'):
        os.makedirs('models')

    # saves trained models
    with open("models/MNB_model_word_"+str(n_gram) +"-gram.pckl", "wb") as f:
        for model in MNB_models:
            pickled.dump(model, f)

    with open("models/SVM_model_word_"+str(n_gram) +"-gram.pckl", "wb") as f:
        for model in SVM_models:
            pickled.dump(model, f)



    

    
