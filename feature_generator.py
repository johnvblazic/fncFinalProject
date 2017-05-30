import spacy
import numpy as np
from tqdm import tqdm
import math
import operator
import os
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from spacy.symbols import VERB
from subject_object_extraction import findSVOs
from sklearn import feature_extraction
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


def remove_stopwords(l):
	# Removes stopwords from a list of tokens
	return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


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


def dot_product(v1, v2):
	return sum(map(operator.mul, v1, v2))


def cosineSimilarity(v1, v2):
	prod = dot_product(v1, v2)
	len1 = math.sqrt(dot_product(v1, v1))
	len2 = math.sqrt(dot_product(v2, v2))
	return prod / (len1 * len2)


def returnLemma(token):
	return token.lemma_


def returnStem(token):
	return pstem.stem(token.lower_)


def getShortenedWords(text,nlp):
	doc = nlp(text)
	wordList = []
	for i,token in enumerate(doc):
		wordList.append(returnLemma(token))

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


def getSVOVector(svo,nlp):
	tempS = str(svo[0]) + " " + str(svo[1]) + " " + str(svo[2])
	doc = nlp(tempS)
	npArray = np.zeros(300,)
	for token in doc:
		npArray = npArray + token.vector
	return npArray


def jointCosines(head1,head2,body):
	cos1 = cosineSimilarity(head1,body)
	cos2 = cosineSimilarity(head2,body)
	#50/50 split. shouldn't change overall rankings if head2 is 0s
	jointCos = (0.5*cos1) + (0.5*cos2)


def SVOSets(headline_svo, body_svo, nlp):
	headline_1 = np.zeros(300,)
	headline_2 = np.zeros(300,)
	body_1 = np.zeros(300,)
	body_2 = np.zeros(300,)
	body_3 = np.zeros(300,)
	cos_1 = 0.0
	cos_2 = 0.0
	cos_3 = 0.0
	if len(headline_svo) > 0 & len(body_svo) > 0:
		if len(headline_svo) > 1:
			headline_2 = getSVOVector(headline_svo[1],nlp)
			headline_1 = getSVOVector(headline_svo[0],nlp)
		elif len(headline_svo) > 0:
			headline_1 = getSVOVector(headline_1)

		#generate cosine sim for each body and replace as necessary. shouldn't break if nothing is there.
		for trip in body_svo:
			tripNP = getSVOVector(trip,nlp)
			tripCos  = jointCosines(headline_1,headline_2,tripNP)
			if tripCos > jointCosines(headline_1,headline_2,body_1):
				body_3 = body_2
				cos_3 = jointCosines(headline_1,headline_2,body_3)
				body_2 = body_1
				cos_2= jointCosines(headline_1,headline_2,body_2)
				body_1 = tripNP
				cos_1=tripCos
			elif tripCos > jointCosines(headline_1,headline_2,body_2):
				body_3 = body_2
				cos_3 = jointCosines(headline_1,headline_2,body_3)
				body_2 = tripNP
				cos_2 = tripCos
			elif tripCos > jointCosines(headline_1,headline_2,body_3):
				body_3 = tripNP
				cos_3 = tripCos

	#create and return list for headline and body
	headList = [headline_1,headline_2]
	bodyList = [body_1,body_2,body_3]
	return headList,bodyList,cos_1,cos_2,cos_3


def svoFeatures(headlines,bodies,nlp):
	X = []
	for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
		features = []
		clean_headline = clean(headline)
		clean_body = clean(body)
		headline_svo = getSVOs(clean_headline,nlp)
		body_svo = getSVOs(clean_body,nlp)
		headList,bodyList,cos_1,cos_2,cos_3 = SVOSets(headline_svo,body_svo,nlp)
		for head in headList:
			for val in head:
				features.append(val)

		for body in bodyList:
			for val in body:
				features.append(val)

		#features.append(cos_1)
		#features.append(cos_2)
		#features.append(cos_3)
		X.append(features)

	return X


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


def ngrams(input, n):
	input = input.split(' ')
	output = []
	for i in range(len(input) - n + 1):
		output.append(input[i:i + n])
	return output


def chargrams(input, n):
	output = []
	for i in range(len(input) - n + 1):
		output.append(input[i:i + n])
	return output


def append_chargrams(features, text_headline, text_body, size):
	grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
	grams_hits = 0
	grams_early_hits = 0
	grams_first_hits = 0
	for gram in grams:
		if gram in text_body:
			grams_hits += 1
		if gram in text_body[:255]:
			grams_early_hits += 1
		if gram in text_body[:100]:
			grams_first_hits += 1
	features.append(grams_hits)
	features.append(grams_early_hits)
	features.append(grams_first_hits)
	return features


def append_ngrams(features, text_headline, text_body, size):
	grams = [' '.join(x) for x in ngrams(text_headline, size)]
	grams_hits = 0
	grams_early_hits = 0
	for gram in grams:
		if gram in text_body:
			grams_hits += 1
		if gram in text_body[:255]:
			grams_early_hits += 1
	features.append(grams_hits)
	features.append(grams_early_hits)
	return features


def hand_features(headlines, bodies, nlp):

	def binary_co_occurence(headline, body):
		# Count how many times a token in the title
		# appears in the body text.
		bin_count = 0
		bin_count_early = 0
		for headline_token in clean(headline).split(" "):
			if headline_token in clean(body):
				bin_count += 1
			if headline_token in clean(body)[:255]:
				bin_count_early += 1
		return [bin_count, bin_count_early]

	def binary_co_occurence_stops(headline, body):
		# Count how many times a token in the title
		# appears in the body text. Stopwords in the title
		# are ignored.
		bin_count = 0
		bin_count_early = 0
		for headline_token in remove_stopwords(clean(headline).split(" ")):
			if headline_token in clean(body):
				bin_count += 1
				bin_count_early += 1
		return [bin_count, bin_count_early]

	def count_grams(headline, body):
		# Count how many times an n-gram of the title
		# appears in the entire body, and intro paragraph

		clean_body = clean(body)
		clean_headline = clean(headline)
		features = []
		features = append_chargrams(features, clean_headline, clean_body, 2)
		features = append_chargrams(features, clean_headline, clean_body, 8)
		features = append_chargrams(features, clean_headline, clean_body, 4)
		features = append_chargrams(features, clean_headline, clean_body, 16)
		features = append_ngrams(features, clean_headline, clean_body, 2)
		features = append_ngrams(features, clean_headline, clean_body, 3)
		features = append_ngrams(features, clean_headline, clean_body, 4)
		features = append_ngrams(features, clean_headline, clean_body, 5)
		features = append_ngrams(features, clean_headline, clean_body, 6)
		return features

	X = []
	for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
		X.append(binary_co_occurence(headline, body)
				 + binary_co_occurence_stops(headline, body)
				 + count_grams(headline, body))


	return X


def getDocVec(text,nlp):
	doc = nlp(text)
	features = doc.vector
	return features


def docVecFeatures(headlines,bodies,nlp):
	X = []
	for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
		features = []
		clean_headline = clean(headline)
		clean_body = clean(body)
		for val in getDocVec(clean_body,nlp):
			features.append(val)
		for val in getDocVec(clean_headline,nlp):
			features.append(val)
		X.append(features)

	return X


def generate_polarity_features(stances,dataset,name,nlp):
	h, b, y = [],[],[]
	for stance in stances:
		if LABELS.index(stance['Stance']) < 3:
			y.append(LABELS.index(stance['Stance']))
			h.append(stance['Headline'])
			b.append(dataset.articles[stance['Body ID']])

	X_overlap = gen_or_load_feats(wordOverlapFeatures, h, b, "features/polarity.overlap."+name+".npy",nlp)
	X_refuting = gen_or_load_feats(refutingFeatures, h, b, "features/polarity.refuting."+name+".npy",nlp)
	X_negation = gen_or_load_feats(negationFeatures, h, b, "features/polarity.negation."+name+".npy",nlp)
	X_svo = gen_or_load_feats(svoFeatures, h, b, "features/polarity.svo."+name+".npy",nlp)
	X_doc_vec = gen_or_load_feats(docVecFeatures, h, b, "features/polarity.docVec."+name+".npy",nlp)
	X_hand = gen_or_load_feats(hand_features, h, b, "features/polarity.hand."+name+".npy",nlp)

	#X = np.c_[X_ngram, X_svo, X_refuting, X_overlap, X_negation,X_doc_vec]
	#X = np.c_[X_ngram, X_svo, X_refuting, X_overlap, X_negation]
	#X = np.c_[X_ngram, X_svo, X_overlap, X_negation]
	X = np.c_[X_hand, X_svo, X_doc_vec,X_negation,X_refuting]
	return X,y

def generate_relatedness_features(stances,dataset,name,nlp):
	h, b, y = [],[],[]
	for stance in stances:
		if LABELS.index(stance['Stance']) < 3:
			y.append(2)
		else:
			y.append(3)
		#y.append(LABELS.index(stance['Stance']))
		h.append(stance['Headline'])
		b.append(dataset.articles[stance['Body ID']])

	X_overlap = gen_or_load_feats(wordOverlapFeatures, h, b, "features/overlap."+name+".npy",nlp)
	#X_refuting = gen_or_load_feats(refutingFeatures, h, b, "features/refuting."+name+".npy",nlp)
	#X_negation = gen_or_load_feats(negationFeatures, h, b, "features/negation."+name+".npy",nlp)
	#X_svo = gen_or_load_feats(svoFeatures, h, b, "features/svo."+name+".npy",nlp)
	X_doc_vec = gen_or_load_feats(docVecFeatures, h, b, "features/docVec."+name+".npy",nlp)
	X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy",nlp)

	#X = np.c_[X_ngram, X_svo, X_refuting, X_overlap, X_negation,X_doc_vec]
	#X = np.c_[X_ngram, X_svo, X_refuting, X_overlap, X_negation]
	#X = np.c_[X_ngram, X_svo, X_overlap, X_negation]
	X = np.c_[X_overlap, X_doc_vec, X_hand]
	return X,y


def generate_test_stances(stances,dataset,name,nlp):
	h, b, y = [],[],[]
	for stance in stances:
		y.append(LABELS.index(stance['Stance']))
		h.append(stance['Headline'])
		b.append(dataset.articles[stance['Body ID']])
	return y