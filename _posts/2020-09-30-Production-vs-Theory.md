---
layout: post
title: Machine Learning in Production
---

*Lessons Learned from Productionizing a ML Pipeline*

In production the stakes are high. People are going to be reading the outputs from the model. And the outputs better make sense. 

Recently my team and I created a NLP classifier and put it into production on a large insurance dataset. It uses TfidfVectorizer and LinearSVC to classify free-text.

But I quickly realised just that putting something into production is so different to the theory.

![examples](https://i.redd.it/5zz3wrn5ypm41.jpg )

In production, I think, it's really important to get the probability of a model prediction. For example, if your model classifies something with a probability of 50% someone should investigate that prediction. If they find a mistake you've prevented the model from disrupting a pivotal system in the company.

But obtaining the probability of a prediction is not always so straight forward.

In this article I'll walk through a way you can extract the probabilities from a SVM classifier in scikit-learn.

This was critical for putting our model into production. 


# Probability calibration

Not only do we want the predicted class label, we also want the probability of that prediction. Scikit-learn has an [interesting section](https://scikit-learn.org/stable/modules/calibration.html) on this topic in their documentation.

We'll need to create a regressor (or calibrator) that maps the output of the classifier to a value between `0` and `1`. This calibrator will then give us the probability of each prediction. 

Essentially the calibrator will try to predict:

![img](/images/production/posterior.png)

where `f` is the output of the classifier.

Or more plainly: Given the output of our classifier, what is the probability that we are 100% certain about this output?


This [paper](https://www.researchgate.net/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods) by [John Platt](https://en.wikipedia.org/wiki/John_Platt_(computer_scientist)) notes that a sigmoid function could be used as a regressor. We obtain the following:


<div style="text-align: center"><img src="/images/production/sigmoid.png" width="300" /></div>

![img](/images/production/sigmoid.png)


To find A and B we can use Maximum Likelihood Estimation.


# Show me the Code

The code for this is really simple. Scikit-learn hides most of the complexity behinds layers of abstraction. 


All you need to do is this:


```python
# from: https://stackoverflow.com/questions/26478000/converting-linearsvcs-decision-function-to-probabilities-scikit-learn-python
svm = LinearSVC()
clf = CalibratedClassifierCV(svm) 
clf.fit(X_train, y_train)
```

CalibratedClassifierCV will fit the training data using a k-fold cross validation approach. The default is 5-fold. See more information [here](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html#sklearn.calibration.CalibratedClassifierCV). 

Then we'll extract the average probability of the predicted class across all the k-folds. `predict_proba` is the function we'll need here. While the function itself seems like an unfinished sentence, it is incredibly useful.

```python
y_proba = clf.predict_proba(X_test)
```

The get the predicted class we can simply use the predict function.

```python
clf.predict(X_test)
``` 

# Evaluating the calibrator

How well does the calibrator fit the data? How can we tell?

To do that we can use the `sklearn.metrics.brier_score_loss`. More information can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss). 

If this score is very high then we cannot look at the probability outputs from the calibrator - they are useless. Instead we'll need to look into better methods of fitting the calibrator. This [article](https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/) has some good approaches to fitting the calibrator. 

# End

That's it really!

I'm sure I'll have more to add as we continue to maintain this model. I wanted to share this little trick that helped us put our model into production effectively. Hopefully someone out there finds it useful.

If I've made a mistake or you're interested in reaching out please feel free to reach to me on [twitter](https://twitter.com/neeliyer11).










