# WordNet - Natural Language Processing With Python and NLTK p.10
# Part of the NLTK Corpora is WordNet.
# I wouldn't totally classify WordNet as a Corpora,
# if anything it is really a giant Lexicon, but, either way,
# it is super useful. With WordNet we can do things like look up words and their meaning according to their parts of speech.
# we can find synonyms, antonyms, and even examples of the word in use. 
# https://youtu.be/T68P5-8tM-Y

from nltk.corpus import wordnet

syns = wordnet.synsets("program")

#synset
print(syns[0].name())

# just the word
print(syns[0].lemmas()[0].name())

# definition
print(syns[0].definition())

# examples
print(syns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
   for l in syn.lemmas():
      synonyms.append(l.name())
      if l.antonyms():
	 antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

# Similarities

syns1 = wordnet.synsets("look")
syns1 = wordnet.synsets("seek")
w1 = wordnet.synset("look.v.01")
w2 = wordnet.synset("seek.v.02")

# wup - Wu-Palmer
# The Wu & Palmer calculates relatedness by considering the depths of the two synsets in the WordNet taxonomies, along with the depth of the LCS (Least Common Subsumer).
# The formula is score = 2 * depth (lcs) / (depth (s1) + depth (s2)).
print('Word 1 : ' + str(w1))
print('Word 2 : ' + str(w2))
# round to nearest 2 decimal places
print(str(round(w1.wup_similarity(w2)*100,2)) + '% Similar')

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")
print('Word 3 : ' + str(w1))
print('Word 4 : ' + str(w2))
print(str(round(w1.wup_similarity(w2)*100,2)) + '% Similar')
