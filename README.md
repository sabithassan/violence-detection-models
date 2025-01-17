# violence-detection-models

This repository contains code and data for training classifiers for detection of violence in Arabic text. Currently the models work only for Arabic texts. The models classify given text to be offensive, obscene or containing no violence.

## Data

Data is obtained from the paper http://www.aclweb.org/anthology/W17-3008. The data has two sources: Tweets and deleted comments from AlJazeera

The data is split in different ways for training, development and testing. The files are in **/input** directory

Files in input directory:


1. **AJ-classification.tsv**: Contains all the deleted comments from AlJazeera dataset and their corresponding labels.
2. **AJ-train.tsv**: contains **19692** instances from AJ-classification.tsv
3. **AJ-dev.tsv**: contains **4000** instances from AJ-classification.tsv from the remaining instances
4. **AJ-test.tsv**: contains **8000** remaining instances from AJ-classification

5. **tweet-classification.tsv**: Contains all the deleted comments from Twitter dataset and their corresponding labels.
6. **tweet-train.tsv**: contains **800** instances from tweet-classification.tsv
7. **tweet-dev.tsv**: contains **100** instances from tweet-classification.tsv from the remaining instances
8. **tweet-test.tsv**: contains **200** remaining instances tweet-classification.tsv

9. **all_data.tsv**: concat of AJ-classification.tsv and tweet-classification.tsv
10. **joint-train.tsv** : contains tweet-train.tsv and some instances of AJ-train.tsv

models can be trained and evaluated on combination of these different datasets

## required libraries

For training and evaluating the classifiers, following libraries are needed:
numpy, pandas, scikit-learn and dill

run **pip install (library)** to install these libraries

## Models

The models for violence detection are located in the directory **/models**
There are four trained models: 
1. Linear SVM with word unigram features
2. Linear SVM with word bigram features
3. Multinomial Bayes with word unigram features
4. Multinomial Bayes with word bigram features
5. Linear SVM with char  3-gram features
6. Linear SVM with char 5-gram features
7. Multinomial Bayes with char 3-gram features
8. Multinomial Bayes with char 5-gram features

To use these models directly, make sure to have python 3.6.4 and scikit-learn 0.19.1 installed. Other versions may cause problems with the pickled file.
Otherwise, please refer to the Training section to train the models locally.

## Training and evaluating models

The code provided trains and evaluates Multinomial Naive Bayes and Linear SVM classifier on word or character n-gram features. n can be specified by user


### Training

To train **word** n-gram based classifiers run:

**python3 violence-detection-word-train.py  (input-file) (n)**

(input-file) is the training file. input-file should contain three tab separated columns, first column can be anything
(could contain ID for example) second column contains the header "text" in first row, followed by the arabic texts, 
third column contains the header "level" followed  by level of violence for the corresponding text in second column.
Levels can be 0,-1,-2 corresponding to no violence, offensive and obscene respectively

(n) denotes n for word -n grams. Setting n = 1 will create word unigram, setting n = 2 will create word bigram etc.
(n) must be greater than 0
    
example Usage:    
**python3 violence-detection-word-train.py  input/tweet-train.tsv 2**

This will create the corresponding SVM and MNB models and corresponding vectorizers for word bigram in the directory **/models**

To train **char** n-gram based classifiers run:

**python3 violence-detection-char-train.py  (input-file) (n)**

Usage is similar to word n-gram


### Evaluating

To evaluate a trained model, run:

**python3 violence-detection-evaluate.py  (input-file) (model-file)**

(input-file) is the testing file. input-file should contain three tab separated columns, first column can be anything
(could contain ID for example) second column contains the header "text" in first row, followed by the arabic texts, 
third column contains the header "level" followed  by level of violence for the corresponding text in second column.
Levels can be 0,-1,-2 corresponding to no violence, offensive and obscene respectively


(model-file) is the previously trained model. It should contain trained classifier and vectorizer.
    
example Usage:    
**python3 violence-detection-evaluate.py  input/tweet-test.tsv models/MNB_model_word_2-gram.pckl**

This will print accuracy, f1 score, precision, recall and confusion matrix.

