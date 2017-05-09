import spacy
import numpy as np
from tqdm import tqdm
import os
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from spacy.symbols import VERB
from subject_object_extraction import findSVOs
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds, generate_hold_out_split, get_stances
from utils.score import report_score, LABELS, score_submission

pstem = PorterStemmer()
wlem = WordNetLemmatizer()

def gen_or_load_feats(feat_fn, headlines, bodies, feature_file,nlp):
	if not os.path.isfile(feature_file):
		feats = feat_fn(headlines, bodies,nlp)
		np.save(feature_file, feats)

	return np.load(feature_file)

def isNegatedVerb(token):
	"""
	Taken from Textacy.spacy_utils as the library wouldn't install -- Link:
	https://textacy.readthedocs.io/en/latest/_modules/textacy/spacy_utils.html
	Returns True if verb is negated by one of its (dependency parse) children,
	False otherwise.
	Args:
		token (``spacy.Token``): parent document must have parse information
	Returns:
		bool
	"""
	if token.doc.is_parsed is False:
		raise ValueError('token is not parsed')
	if token.pos == VERB and any(c.dep_ == 'neg' for c in token.children):
		return True

	return False

def removeStopWords(text,nlp):
	newText = ""
	for word in text.split():
		if nlp.vocab[word].is_stop:
			pass
		else:
			newText = " ".join((newText,word))

	return newText

def returnLemma(token):
	return token.lemma_

def returnStem(token):
	return pstem.stem(token.lower_)

def getShortenedWords(text,nlp):
	doc = nlp(text)
	wordList = []
	for i,token in enumerate(doc):
		wordList.append(returnStem(token))

	return wordList

def checkNegation(text,nlp):
	doc = nlp(text)
	for word in doc:
		if word.pos == VERB:
			if isNegatedVerb(word):
				return True

	# general return is False
	return False

def countNegation(text,nlp):
	doc = nlp(text)
	verbs = set()
	for word in doc:
		if word.pos == VERB:
			if isNegatedVerb(word):
				verbs.add(word)

	return len(verbs)

def clean(s):
	# Cleans a string: Lowercasing, trimming, removing non-alphanumeric
	return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

def getSVOs(text,nlp):
	doc = nlp(text)
	#Generates triples of Subject, Verb, Object if they exist.
	svoList = []
	svoList = findSVOs(doc)
	return svoList


def genOrLoadFeats(feat_fn, headlines, bodies, feature_file):
	if not os.path.isfile(feature_file):
		feats = feat_fn(headlines, bodies)
		np.save(feature_file, feats)

	return np.load(feature_file)


def wordOverlapFeatures(headlines,bodies,nlp):
	X = []
	for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
		features = []
		clean_headline = clean(headline)
		clean_body = clean(body)
		clean_headline = getShortenedWords(clean_headline,nlp)
		clean_body = getShortenedWords(clean_body,nlp)
		features.append(len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body))))
		X.append(features)

	return X


def negationFeatures(headlines,bodies,nlp):
	
	X = []
	for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
		features = []
		clean_headline = clean(headline)
		clean_body = clean(body)
		if checkNegation(clean_headline,nlp):
			if checkNegation(clean_body,nlp):
				features.append(1)
				features.append(1)
				features.append(countNegation(clean_headline,nlp))
				features.append(countNegation(clean_body,nlp))
			else:
				features.append(1)
				features.append(0)
				features.append(countNegation(clean_headline,nlp))
				features.append(0)
		else:
			if checkNegation(clean_body,nlp):
				features.append(0)
				features.append(1)
				features.append(0)
				features.append(countNegation(clean_body,nlp))
			else:
				features.append(0)
				features.append(0)
				features.append(0)
				features.append(0)

		X.append(features)

	return X


def refutingFeatures(headlines, bodies,nlp):
	_refuting_words = [
		'fake', 'fakes'
		'fraud', 'frauds', 'defraud'
		'hoax',
		'false',
		'deny', 'denies',
		'refute', 'refutes',
		'not',
		'despite',
		'nope',
		'doubt', 'doubts',
		'bogus',
		'debunk',
		'prank', 'pranks',
		'incorrect',
		'inaccurate',
		'erroneous',
		'lie', 'lies',
		'pranks',
		'error', 'errors',
		'retract', 'retracts'
	]
	X = []
	for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
		clean_headline = clean(headline)
		clean_headline = getShortenedWords(clean_headline,nlp)
		features = [1 if word in clean_headline else 0 for word in _refuting_words]
		X.append(features)

	return X

def getNGrams(text,numGrams,nlp):
	doc = nlp(text)
	if n < 1:
		raise ValueError('n must be greater than or equal to 1')

	ngrams_ = (doc[i: i + numGrams]
			   for i in range(len(doc) - n + 1))
	ngrams_ = (ngram for ngram in ngrams_
			   if not any(w.is_space for w in ngram))
	return ngrams_

def nGramFeatures(headlines,bodies,nlp):
	X = []
	for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
		features = []
		clean_headline = clean(headline)
		clean_body = clean(body)
		trigrams_headline = getNGrams(clean_headline,3,nlp)
		trigrams_body = getSVOs(clean_body,3,nlp)
		features.append(len(set(trigrams_headline)))
		features.append(len(set(trigrams_body)))
		#+1 smoothing on denominator -- headlines/bodies without useable SVOs should be 0, not 1
		features.append(len(set(trigrams_headline).intersection(trigrams_body))/float(len(set(trigrams_headline).union(trigrams_body)) + 1))
		X.append(features)

	return X

def svoFeatures(headlines,bodies,nlp):
	X = []
	for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
		features = []
		clean_headline = clean(headline)
		clean_body = clean(body)
		headline_svo = getSVOs(clean_headline,nlp)
		body_svo = getSVOs(clean_body,nlp)
		features.append(len(set(headline_svo)))
		features.append(len(set(body_svo)))
		#+1 smoothing on denominator -- headlines/bodies without useable SVOs should be 0, not 1
		features.append(len(set(headline_svo).intersection(body_svo))/float(len(set(headline_svo).union(body_svo)) + 1))
		X.append(features)

	return X



def generate_features(stances,dataset,name,nlp):
	h, b, y = [],[],[]
	for stance in stances:
		y.append(LABELS.index(stance['Stance']))
		h.append(stance['Headline'])
		b.append(dataset.articles[stance['Body ID']])

	X_overlap = gen_or_load_feats(wordOverlapFeatures, h, b, "features/overlap."+name+".npy",nlp)
	X_refuting = gen_or_load_feats(refutingFeatures, h, b, "features/refuting."+name+".npy",nlp)
	X_negation = gen_or_load_feats(negationFeatures, h, b, "features/negation."+name+".npy",nlp)
	X_svo = gen_or_load_feats(svoFeatures, h, b, "features/svo."+name+".npy",nlp)
	X_ngram = gen_or_load_feats(nGramFeatures, h, b, "features/ngram."+name+".npy",nlp)

	X = np.c_[X_ngram, X_svo, X_refuting, X_overlap, X_negation]
	return X,y

nlp = spacy.load('en')
sentence = "This is an example sentence full of very large and very important words."
sentence2 = "They are not the best words."
sentence3 = "John won the money."
print(sentence)
print(removeStopWords(sentence, nlp))
print(checkNegation(sentence2,nlp))
print(getSVOs(sentence,nlp))
print(getSVOs(sentence2,nlp))
print(getSVOs(sentence3,nlp))

d = DataSet()


train_stances = get_stances(d)
train_Xs, train_ys = generate_features(train_stances,d,"train",nlp)

del d
del train_stances

X_train = np.vstack(tuple([train_Xs[i] for i in ids]))
y_train = np.hstack(tuple([train_ys[i] for i in ids]))

print(fold_stances[0][0])
print(fold_stances[0][1])
print(hold_out_stances[0])
