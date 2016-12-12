---
layout: post
title:  "Image Fisher Vectors In Python"
date:   2014-12-05 22:10:33 +0200
permalink: machinelearning/fisher-vectors-python
---

Although the state of the art in image classification (while writing this post) is deep learning,
Bag of words approaches still perform well on many image datasets.

Fisher vectors is the state of the art in that approach, allowing training more discriminative classifiers with a lower vocabulary size.

I wrote a simple Python implementation for calculating fisher vectors and using it to classify image categories:

[https://github.com/jacobgil/pyfishervector](https://github.com/jacobgil/pyfishervector)

You can look here for a derivation: 

[https://hal.inria.fr/hal-00830491/PDF/journal.pdf](https://hal.inria.fr/hal-00830491/PDF/journal.pdf)

[http://www.vlfeat.org/api/fisher-derivation.html](http://www.vlfeat.org/api/fisher-derivation.html)

The main improvement here is extracting a richer feature vector from images compared to bag of words.

In Bag of Words, for each local feature we find the closest word in the vocabulary, and add +1 to the histogram of the vocabulary in the input image.
But we could have sampled more data: 
- How far is each feature from its closest vocabulary word.
- How far is the feature from other vocabulary words.
- The distribution of the vocabulary words themselves. 


## Brief outline of fisher vectors
  

#### Vocabulary learning with GMM:

 - Sample many features from input images.
 - Fit a Gaussian Mixture Model on those features. 
 - The result is a vocabulary of dominant features in the image, and their distributions.

#### Image representation based on the vocabulary:

 -  Measure the expectation of the difference and distance of the image features, from each Gaussian distrubution, using the likelihood a feature belongs to certain gaussian. 
 - Concatenate the resulting vector for each vocabulary word, into one large descriptor vector.

There is also a normalization step that I will skip here but is a part of the implementation, that is important if the features are fed into a classifier like SVM that needs normalized inputs.


This is a generalization of bag of words. If you set the likelihood of a feature to a vocabulary word to be 1 to it's closest word and 0 to the rest, 
and if you redefine the distance to be a constant "1", you get the original bag of words model.



# Trying out the implementation
{% highlight bash %}
python fisher.py <path_to_image_directory> <vocabulary size>
{% endhighlight %}

The image directory should contain two sub folders, one for the images of each class.

It currently just trains a model and then classifies the images.
The input images definitely need to be partitioned into training and validation parts.

One more thing:
Fisher vectors are successfully used in face recognition, check out:
[http://www.robots.ox.ac.uk/~vgg/publications/2013/Simonyan13/extras/simonyan13_ext.pdf](http://www.robots.ox.ac.uk/~vgg/publications/2013/Simonyan13/extras/simonyan13_ext.pdf)

In their paper they extract features densely from a grid,  reduce the dimensionality with PCA, and augment the features with their spacial location.



