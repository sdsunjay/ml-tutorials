# What Makes a Good Feature? - Machine Learning Recipes 3

# Good features are informative, independent, and simple. In this episode, we'll introduce these
# concepts by using a histogram to visualize a feature from a toy dataset. Updates: many thanks for
# the supportive feedback! I’d love to release these episodes faster, but I’m writing them as we go.
# That way, I can see what works and (more importantly) where I can improve. 
#
# We've covered a lot of ground already, so next episode I'll review and reinforce concepts,
# introduce clearer syntax, spend more time on testing, and continue building intuition for
# supervised learning. 
#
# I also realize some folks had dependency bugs with Graphviz (my fault!). Moving forward, I won't
# use any libraries not already installed by Anaconda or Tensorflow. 
# https://www.youtube.com/watch?v=N9fDIAflCMY

import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height, lab_height], stacked=True, color=['r','b'])
plt.show()

