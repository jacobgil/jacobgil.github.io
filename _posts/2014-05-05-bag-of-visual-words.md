---
layout: post
title:  "Bag of visual words for image classification"
date:   2014-05-05 22:10:33 +0200
permalink: machinelearning/bag-of-words
---

[Github repository](https://github.com/jacobgil/BagOfVisualWords)

I wanted to play around with Bag Of Words for visual classification, so I coded a Matlab implementation that uses VLFEAT for the features and clustering.
It was tested on classifying Mac/Windows desktop screenshots.


For a small testing data set (about 50 images for each category), the best vocabulary size was about 80.
It scored 97% accuracy on the training set, and 85% accuracy on the cross validation set,
so the over-fitting can be improved a bit more.

Overview:
---------

1. Collect a data set of examples. I used a python script to download images from Google.
2. Partition the data set into a training set, and a cross validation set (80% - 20%).
3. Find key points in each image, using [SIFT](http://en.wikipedia.org/wiki/Scale-invariant_feature_transform).
4. Take a patch around each key point, and calculate it's [Histogram of Oriented Gradients (HoG).](http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) Gather all these features.
5. Build a visual vocabulary by finding representatives of the gathered features (quantization).
This done by [k-means clustering](http://en.wikipedia.org/wiki/K-means_clustering).
6. Find the distribution of the vocabulary in each image in the training set.
This is done by a histogram with a bin for each vocabulary word.
The histogram values can be either hard values, or soft values.
Hard values means that for each descriptor of a key point patch in an image, we add 1 to the bin of the vocabulary word closest to it in absolute square value.
Soft values means that each patch votes to all histogram bins, but give a higher weight to bin representing words that are similar to that patch. [Take a look here.](http://dare.uva.nl/document/126930)
7. Train an SVM on the resulting histograms (each histogram is a feature vector, with a label).
8. Test the classifier on the cross validation set.
9. If results are not satisfactory, repeat 5 for a different vocabulary size and a different SVM parameters.

Visualization of the vocabulary learned by the clustering
---------------------------------------------------------
![enter image description here](http://3.bp.blogspot.com/-HWhxedpytz8/U2e6QIePZzI/AAAAAAAAFhc/6nIFZlYPXT4/s1600/bag.jpg)
