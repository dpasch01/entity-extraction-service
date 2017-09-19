#!/usr/bin/python

from flask import Flask, request
app = Flask(__name__)

import sys
import os
import re
import subprocess
import platform
import time
import codecs

from signal import *

BASE_DIR = 'twitter_nlp.jar'

if os.environ.has_key('TWITTER_NLP'):
    BASE_DIR = os.environ['TWITTER_NLP']

sys.path.append('%s/python' % (BASE_DIR))
sys.path.append('%s/python/ner' % (BASE_DIR))
sys.path.append('%s/hbc/python' % (BASE_DIR))

import Features
import twokenize
from LdaFeatures import LdaFeatures
from Dictionaries import Dictionaries
from Vocab import Vocab

sys.path.append('%s/python/cap' % (BASE_DIR))
sys.path.append('%s/python' % (BASE_DIR))
import cap_classifier
import pos_tagger_stdin
import chunk_tagger_stdin
import event_tagger_stdin

def GetNer(ner_model, memory="256m"):
    return subprocess.Popen('java -Xmx%s -cp %s/mallet-2.0.6/lib/mallet-deps.jar:%s/mallet-2.0.6/class cc.mallet.fst.SimpleTaggerStdin --weights sparse --model-file %s/models/ner/%s' % (memory, BASE_DIR, BASE_DIR, BASE_DIR, ner_model),
                           shell=True,
                           close_fds=True,
                           stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE)

posTagger = pos_tagger_stdin.PosTagger()
ner_model = 'ner_nopos_nochunk.model'
ner = GetNer(ner_model, memory="256m")
fe = Features.FeatureExtractor('%s/data/dictionaries' % (BASE_DIR))
capClassifier = cap_classifier.CapClassifier()
vocab = Vocab('%s/hbc/data/vocab' % (BASE_DIR))

dictMap = {}
i = 1
for line in open('%s/hbc/data/dictionaries' % (BASE_DIR)):
    dictionary = line.rstrip('\n')
    dictMap[i] = dictionary
    i += 1

dict2index = {}
for i in dictMap.keys():
    dict2index[dictMap[i]] = i

entityMap = {}
i = 0
dict2label = {}
for line in open('%s/hbc/data/dict-label3' % (BASE_DIR)):
    (dictionary, label) = line.rstrip('\n').split(' ')
    dict2label[dictionary] = label

out_fp = sys.stdout

@app.route("/")
def hello():
	start_time = time.time()
	row = request.args.get('tweet').strip().split("\t")
	tweet = row[0]
	line = tweet.encode('utf-8', "ignore")
	words = twokenize.tokenize(line)
	seq_features = []
	tags = []

	goodCap = capClassifier.Classify(words) > 0.9
	pos = posTagger.TagSentence(words)
	pos = [re.sub(r':[^:]*$', '', p) for p in pos]  
	quotes = Features.GetQuotes(words)
	for i in range(len(words)):
		features = fe.Extract(words, pos, None, i, goodCap) + ['DOMAIN=Twitter']
		if quotes[i]:
		    features.append("QUOTED")
		seq_features.append(" ".join(features))
	ner.stdin.write(("\t".join(seq_features) + "\n").encode('utf8'))
		
	for i in range(len(words)):
		tags.append(ner.stdout.readline().rstrip('\n').strip(' '))

	features = LdaFeatures(words, tags)

	for i in range(len(features.entities)):
		type = None
		wids = [str(vocab.GetID(x.lower())) for x in features.features[i] if vocab.HasWord(x.lower())]
		tags[features.entities[i][0]] = "B-ENTITY"
		for j in range(features.entities[i][0]+1,features.entities[i][1]):
		    tags[j] = "I-ENTITY"

	output = ["%s/%s" % (words[x], tags[x]) for x in range(len(words))]
	output = ["%s/%s" % (output[x], pos[x]) for x in range(len(output))]

	row[0] = (" ".join(output))
	end_time = time.time()
	print >> sys.stderr, "Tweet NER duration = %ss" % (str(end_time-start_time))
	return ("\t".join(row)).encode('utf8')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)


