# Natural Language Processing With Python and NLTK p.1 Tokenizing words and Sentences
# Natural Language Processing is the task we give computers to read and understand (process) written text (natural language). By far, the most popular toolkit or API to do natural language processing is the Natural Language Toolkit for the Python programming language. 
#
# The NLTK module comes packed full of everything from trained algorithms to identify parts of speech to unsupervised machine learning algorithms to help you train your own machine to understand a specific bit of text. 
#
# NLTK also comes with a large corpora of data sets containing things like chat logs, movie reviews, journals, and much more!
#
# Bottom line, if you're going to be doing natural language processing, you should definitely look into NLTK!
# https://youtu.be/FLZvOKSCkxY
import nltk, re
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download()
example_text = "Hello Mr. Smith, how are you doing today? The weather is great and Python is awesome. The sky is blue. Do not eat watermelon today!"

print(sent_tokenize(example_text)) 
print(word_tokenize(example_text)) 
