import spacy
import csv
import os
import sys
import re
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds, generate_hold_out_split, get_stances
from utils.score import report_score, LABELS, score_submission,print_confusion_matrix
from feature_generator import generate_relatedness_features, generate_polarity_features, generate_test_stances

def writeToFile(text):
	with open("errors.txt","a") as f:
		f.write(text)

def write(self,filename):
	field_names = ['Headline', 'Body ID', 'Stance']
	with open('my_output.csv', 'w', encoding='utf8') as f:
		writer = csv.DictWriter(f, fieldnames=field_names)
		writer.writeheader()
		for row in self.stances:
			writer.writerows(row)

nlp = spacy.load('en')

#########Generate Training Features Section###########
d = DataSet("data")
train_stances = get_stances(d)
train_Xs, train_ys = generate_relatedness_features(train_stances,d,"train.lemma",nlp)


X_train = np.vstack(tuple([train_Xs]))
y_train = np.hstack(tuple([train_ys]))

del train_Xs
del train_ys

#########Generate Test Features Section###########
t = DataSet(split="test")
test_stances = get_stances(t)
test_Xs, test_ys = generate_relatedness_features(test_stances,t,"test.lemma",nlp)


X_test = np.vstack(tuple([test_Xs]))
y_test = np.hstack(tuple([test_ys]))


#########Training Section###########
model = MLPClassifier(verbose=True)
model1 = MLPClassifier()


if os.path.isfile("relatednessModel.mdl"):
	model1 = joblib.load("relatednessModel.mdl")
else:
	model1.fit(X_train,y_train)

prediction = model1.predict(X_test)
predicted = [LABELS[int(a)] for a in prediction]
actual = [LABELS[int(a)] for a in y_test]

temp_score, _ = score_submission(actual, predicted)
max_score, _ = score_submission(actual, actual)

score = temp_score/max_score

highestScore = score
bestModel1 = model1
bestPrediction = []

for m in range(1):
	model1 = MLPClassifier()
	print("Testing relatedness classifier number " + str(m))

	model1.fit(X_train,y_train)

	#########Prediction Section###########
	prediction = model1.predict(X_test)
	predicted = [LABELS[int(a)] for a in prediction]
	actual = [LABELS[int(a)] for a in y_test]

	temp_score, cm = score_submission(actual, predicted)
	max_score, _ = score_submission(actual, actual)

	score = temp_score/max_score

	if score > highestScore:
		print("New score - " + str(score) + " is better than old score - " + str(highestScore) + " replacing.")
		highestScore = score
		bestModel1 = model1
		bestPrediction = prediction

if os.path.isfile("relatednessModel.mdl"):
	os.remove("relatednessModel.mdl")
joblib.dump(bestModel1,"relatednessModel.mdl")


print_confusion_matrix(cm)
print("Score for related/unrealted testing was - " + str(highestScore))


orig_prediction = bestPrediction
#########--------------Agree/Disagre/Discuss Section---------------###########
train_Xs, train_ys = generate_polarity_features(train_stances,d,"train.lemma",nlp)

X_train = np.vstack(tuple([train_Xs]))
y_train = np.hstack(tuple([train_ys]))

del train_Xs
del train_ys

if os.path.isfile("polarityModel.mdl"):
	model = joblib.load("polarityModel.mdl")
else:
	model.fit(X_train,y_train)


test_Xs,test_ys = generate_test_stances(test_stances,t,"test.lemma",nlp)
X_test = np.vstack(tuple([test_Xs]))
y_test = np.hstack(tuple([test_ys]))

i = 0
c = 0
for val in prediction:
	if int(val) < 3:
		prediction[i] = model.predict(X_test[i].reshape(1,-1))
		c += 1
	i += 1

prediction = orig_prediction
predicted = [LABELS[int(a)] for a in prediction]
actual = [LABELS[int(a)] for a in y_test]

temp_score, cm = score_submission(actual, predicted)
max_score, _ = score_submission(actual, actual)

score = temp_score/max_score

bestModel = model
highestScore = score
for m in range(1):
	model = MLPClassifier()
	print("Testing polarity classifier number " + str(m))
	model.fit(X_train,y_train)

	prediction = orig_prediction
	i = 0
	c = 0
	for val in prediction:
		if int(val) < 3:
			prediction[i] = model.predict(X_test[i].reshape(1,-1))
			c += 1
		i += 1


	predicted = [LABELS[int(a)] for a in prediction]
	actual = [LABELS[int(a)] for a in y_test]

	temp_score, cm = score_submission(actual, predicted)
	max_score, _ = score_submission(actual, actual)

	score = temp_score/max_score

	if score > highestScore:
		print("New score - " + str(score) + " is better than old score - " + str(highestScore) + " replacing.")
		highestScore = score
		bestModel = model


print_confusion_matrix(cm)
print("Score for full testing was - " + str(highestScore))

if os.path.isfile("polarityModel.mdl"):
	os.remove("polarityModel.mdl")
joblib.dump(bestModel,"polarityModel.mdl")

# SO a better way to do this would be to generate the polarity features for every data point and have them saved.
# At that point, step through the predicted values from the first classifier, repredicting the value at each point
# that isn't an unrelated point, and replacing that value in the predictions. Also need to generate a test set that
# has the full output, which should be easily doable if I'm loading the polarity features for every test point, because
# I'll just be feeding the vector I have for that point into the model to make a prediction, so the stances don't really
# matter.