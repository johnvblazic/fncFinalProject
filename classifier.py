import spacy
import os
import sys
import re
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds, generate_hold_out_split, get_stances
from utils.score import report_score, LABELS, score_submission,print_confusion_matrix
from feature_generator import generate_features

def writeToFile(text):
	with open("errors.txt","a") as f:
		f.write(text)

if len(sys.argv) != 2:
	sys.exit("Incorrect number of arguments. Correct usage is \'python classifier.py <\"SVC\" OR \"MLP\"> \'")

nlp = spacy.load('en')

#########Generate Training Features Section###########
d = DataSet("data")
train_stances = get_stances(d)
train_Xs, train_ys = generate_features(train_stances,d,"train.lemma",nlp)

del d
del train_stances

X_train = np.vstack(tuple([train_Xs]))
y_train = np.hstack(tuple([train_ys]))

del train_Xs
del train_ys

#########Training Section###########
model = SVC(verbose=True)

if sys.argv[1] == "MLP":
	model = MLPClassifier(verbose=True)

model.fit(X_train,y_train)

predicted = [LABELS[int(a)] for a in model.predict(X_train)]
actual = [LABELS[int(a)] for a in y_train]

temp_score, _ = score_submission(actual, predicted)
max_score, _ = score_submission(actual, actual)

score = temp_score/max_score

print("Score for training was - " + str(score))

#########Generate Test Features Section###########
t = DataSet(split="test")
test_stances = get_stances(t)
test_Xs, test_ys = generate_features(test_stances,t,"test.lemma",nlp)


X_test = np.vstack(tuple([test_Xs]))
y_test = np.hstack(tuple([test_ys]))


#########Prediction Section###########
predicted = [LABELS[int(a)] for a in model.predict(X_test)]
actual = [LABELS[int(a)] for a in y_test]

temp_score, cm = score_submission(actual, predicted)
max_score, _ = score_submission(actual, actual)

score = temp_score/max_score

print_confusion_matrix(cm)

print("Score for testing was - " + str(score))

i = 0
c = 0
while c < 100:
	if predicted[i] != actual[i]:
		print(test_stances[i],"|",predicted[i],"|",actual[i],"|",t.articles[test_stances[i]['Body ID']])
		c += 1

	i += 1
