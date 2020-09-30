---
layout: post
title: Putting ML models into Production
---

_Converting a Decision Function into Probabilities in Sci-kit Learn_

In production the stakes are high. People are going to be reading the outputs from the model. And the outputs better make sense. 

Recently my team and I created a NLP classifier and put it into production on a large insurance dataset. It uses TfidfVectorizer and LinearSVC to classify free-text. Nothing fancy really. But I quickly realised just that putting something into production is so different to the theory. 

In production, it's really important to get the probability of a model prediction. For example, if your model classifies something with a probability of 50% someone should investigate that prediction. If they find a mistake you've prevented the model from disrupting a pivotal system in the company. ML should add value to the system not detract from it.

In this article I'll walk through a way you can extract the probabilities from an ordinary decision function in scikit-learn. This was critical in putting our model into production. 


# Probability calibration

Not only do we want the predicted class label, we also want the probability of that prediction. Scikit-learn has an [interesting section](https://scikit-learn.org/stable/modules/calibration.html) on this topic in their documentation.

We'll need to create a regressor (or calibrator) that maps the output of the classifier to a value between `0` and `1`. This calibrator will then give us the probability of each prediction. 

Essentially the calibrator will try to predict:

p(yi = 1|fi), where fi is the output of the classifier.

Or more plainly: Given the output of our classifier, what is the probability that we are 100% certain about this output?



# Code

<!-- `predict_proba` is the function that we'll need here. While the function itself seems like an unfinished sentence, it is incredibly useful.  -->

The code for this is really simple. Part of the reason is that the beautiful people at sci-kit learn have hidden all the complex maths behind layers of abstraction. 

All you need to do is this:







