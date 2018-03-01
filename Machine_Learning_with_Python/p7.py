# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

from unidecode import unidecode

def read_file(filename):
	# Read a file
	in_file = open(filename, "r")
	text = in_file.read()
#unidecode(u"\u5317\u4EB0")

	in_file.close()
	return unidecode(text)
filename = "/root/web/rideshare/python/good/play/offering.txt"
#train_text = state_union.raw("2005-GWBush.txt")
#sample_text = state_union.raw("2006-GWBush.txt")

train_text = read_file(filename); 
filename = "/root/web/rideshare/python/good/play/new_offering.txt"
sample_text = read_file(filename); 
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
	try:
		for i in tokenized:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)
		        namedEnt = nltk.ne_chunk(tagged)
			print(namedEnt)	

	except Exception as e:
		print(str(e))

process_content()
