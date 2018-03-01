# Named Entity Recognition - Natural Language Processing With Python and NLTK p.7
# Named entity recognition is useful to quickly find out what the subjects of discussion are.
# NLTK comes packed full of options for us. We can find just about any named entity, or we can look for specific ones.
#
# NLTK can either recognize a general named entity, or it can even recognize locations, names, monetary amounts, dates, and more. 
# https://youtu.be/LFXsG7fueyk
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
# train_filename = "/root/web/rideshare/python/good/play/offering.txt"
# train_text = read_file(train_filename); 
# sample_filename = "/root/web/rideshare/python/good/play/new_offering.txt"
# sample_text = read_file(sample_filename); 
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

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
