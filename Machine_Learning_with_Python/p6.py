# Chinking - Natural Language Processing With Python and NLTK p.6
# Chinking is a part of the chunking process with natural language processing with NLTK. A chink is what we wish to remove from the chunk.
# We define a chink in a very similar fashion compared to how we defined the chunk. 
#
# The reason why you may want to use a chink is when your chunker is getting almost everything you want, but is also picking up some things you don't want.
# You could keep adding chunker rules, but it may be far easier to just specify a chink to remove from the chunk.
# https://youtu.be/EymPQgCtcAE
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
	try:
		for i in tokenized:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)
			#chinking - to remove stuff
			
			chunkGram = """Chunk: {<.*>?} """
			chunkParser = nltk.RegexpParser(chunkGram)
			chunked = chunkParser.parse(tagged)
			print(chunked)
			

	except Exception as e:
		print(str(e))

process_content()
