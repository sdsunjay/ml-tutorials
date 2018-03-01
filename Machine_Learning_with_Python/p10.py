from nltk.corpus import wordnet

syns = wordnet.synsets("look")
print(syns)
syns = wordnet.synsets("seek")
print(syns)
w1 = wordnet.synset("look.v.01")
w2 = wordnet.synset("seek.v.02")

print(w1.wup_similarity(w2))
