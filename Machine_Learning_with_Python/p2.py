# Stop Words - Natural Language Processing With Python and NLTK p.2
# One of the largest elements to any data analysis, natural language processing included, is pre-processing. This is the methodology used to "clean up" and prepare your data for analysis. 
#
# One of the first steps to pre-processing is to utilize stop-words. Stop words are words that you want to filter out of any analysis. These are words that carry no meaning, or carry conflicting meanings that you simply do not want to deal with. 
#
# The NLTK module comes with a set of stop words for many language pre-packaged, but you can also easily append more to this list. 
# https://youtu.be/w36-U-ccajM
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = "This is an example sentence showing off stop word filtration."
stop_words = set(stopwords.words("english"))

words = word_tokenize(example_sentence)

filtered_sentence = []

for w in words:
	if w not in stop_words:
		filtered_sentence.append(w)

print(filtered_sentence)
