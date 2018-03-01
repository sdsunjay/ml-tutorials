# NLTK Corpora - Natural Language Processing With Python and NLTK p.9
# Remember from the beginning, we talked about this term, "corpora."
#
# Again, corpora is just a body of texts. Generally, corpora are grouped by some sort of defining characteristic.
#
# NLTK is a massive toolkit for you. part of what they give you is a ton of highly valuable corpora to learn with, train against, and some of them are even capable of using in production.
#
# This video is going to be all about accessing your corpora!
# https://youtu.be/TKAXDqoG2dc
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw('bible-kjv.txt')
tok = sent_tokenize(sample)

print(tok[5:15])
