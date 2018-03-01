import nltk, re
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download()
example_text = "Hello Mr. Smith, how are you doing today? The weather is great and Python is awesome. The sky is blue. Do not eat watermelon today!"

print(sent_tokenize(example_text)) 
print(word_tokenize(example_text)) 
