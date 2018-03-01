# Stemming - Natural Language Processing With Python and NLTK p.3
# Another form of data pre-processing with natural language processing is called "stemming." 
#
# This is the process where we remove word affixes from the end of words. 
#
# The reason we would do this is so that we do not need to store the meaning of every single tense of a word. For example:
#
# Reader
# Reading
# Read
#
# Aside from tense, and even one of these is a noun, they all have the same meaning for their "root" stem (read).
#
# This way, we store one single value for the root stem of "read."
# Then, when we wish to learn more, we can look into the affixes that were on the end, like "ing" is an active word, or in the past, then you have reader as someone who reads... then just plain read as either past tense or current. 
# https://youtu.be/yGKTphqxR9Q
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

new_text = "It is very important to be pythonly while you are pythoning with python. All pythoners have pythoned atleast once!."

words = word_tokenize(new_text)

for w in words:
	print(ps.stem(w))
