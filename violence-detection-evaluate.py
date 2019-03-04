import sys
import pandas as pd
import numpy as np
import dill as pickled
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def print_usage():
    ''' Usage function. the usage is shown when the file is run without arguments/ wrong number of arguments'''
    
    print (
    '''Usage:
    python3 violence-detection-evaluate.py  <input-file> <model-file>

    <input-file> is the training file. input-file should contain three tab separated columns, first column can be anything
    (could contain ID for example) second column contains the header "text" in first row, followed by the arabic texts, 
    third column contains the header "level" followed  by level of violence for the corresponding text in second column.

    <model-file> is the previously trained model. It should contain trained classifier and vectorizer.
        
    example Usage:    
    python3 violence-detection-evaluate.py  ./input/tweet-test.tsv ./models/MNB_model_word_2-gram.pckl
        ''')


def load_model(model_file):
    ''' loads previously trained model'''
    models = []
    with open(model_file, "rb") as f:
        while True:
            try:
                models.append(pickled.load(f))
            except EOFError:
                break
    return models

def print_evaluation(predictions, encoded_labels):

    '''Prints accuracy, precision, recall, f1 score'''

    accuracy = accuracy_score(predictions, encoded_labels)

    precision_micro = precision_score(predictions, encoded_labels, average='micro')
    precision_macro = precision_score(predictions, encoded_labels, average='macro')

    recall_micro = recall_score(predictions, encoded_labels, average='micro')
    recall_macro = recall_score(predictions, encoded_labels, average='macro')


    f1_micro = f1_score(predictions, encoded_labels, average='micro')
    f1_macro = f1_score(predictions, encoded_labels, average='macro')


    print('Accuracy = %.4f' % accuracy)

    print('Precision (Micro) = %4f' % precision_micro)
    print('Precision (Macro) = %4f' % precision_macro)

    print('Recall-Micro = %4f' % recall_micro)
    print('Recall-Macro = %4f' % recall_macro)

    print('F1_micro = %4f' % f1_micro)
    print('F1_macro = %4f' % f1_macro)


from sklearn.metrics import confusion_matrix

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    ''' pretty print for confusion matrixes
        from https://gist.github.com/zachguo/10296432
    '''
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    print ("    " + empty_cell, end='')
    for label in labels: 
        print ("%{0}s".format(columnwidth) % label, end='')
    print("")
    for i, label1 in enumerate(labels):
        print ("    %{0}s".format(columnwidth) % label1, end='')
        for j in range(len(labels)): 
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print (cell, end='')
        print ()


def map_labels(x):
    ''' maps labels of 0,-1,-2 to No violence, Offensive and Obscene'''

    if (x == 0):
        return "No violence"
    elif (x == -1):
        return "Offensive"
    else:
        return "Obscene"



if __name__ == '__main__':
    ''' First loads the evaluation data and previously trained models. Then uses the trained
        model for evaluation of the evaluation data. 
    '''

    # checks number of arguments
    if (len (sys.argv) != 3):
        print_usage()
        exit()

    # arguments
    file_name = sys.argv[1]
    model_name= sys.argv[2]

    clf, vectorizer = load_model(model_name)


    # reads file and separates text and labels
    readfile = pd.read_csv(file_name, sep = '\t', index_col=0)    
    test_input = readfile['text'].values
    test_labels = readfile['level'].values


    print ("Number of instances", test_input.shape[0])

    # gets word n-gram features
    n_gram_features = vectorizer.transform(test_input)
    predicted_labels = clf.predict(n_gram_features)
    predicted_labels = list (map (map_labels, predicted_labels))
    test_labels = list (map (map_labels, test_labels))



    print_evaluation(predicted_labels, test_labels)

    unique_labels = ["No violence", "Offensive", "Obscene"]
    
    conf_matrix = confusion_matrix(test_labels, predicted_labels, unique_labels)
    print_cm(conf_matrix, unique_labels)








    
