---
layout: post
title:  "Deriving confidence intervals for variants of Recall, Precision and F1"
date:   2023-06-14 22:10:33 +0200
permalink: stats/confidence_intervals
tags: [Statistics, Machine Learning, Metrics]
categories: [Statistics]
excerpt: ""

---
{% include katex.html %}

- [Introduction](#introduction)
- [The computation flow for confidence intervals with the delta method](#the-flow)
- [The confusion matrix multinomial distribution](#the-confusion-matrix-multinomial-distribution)
- [The central limit theorem applied on the confusion matrix](#the-central-limit-theorem-applied-on-the-confusion-matrix)
- [Expressing the metrics in terms of the confusion matrix](#expressing-the-metrics-in-terms-of-the-confusion-matrix)
- [The multi-variate delta method](#the-multi-variate-delta-method)
- [Computing the derivative for binary F1](#computing-the-derivative-for-binary-f1)
- [Computing the derivative for Macro Recall](#computing-the-derivative-for-macro-recall)
- [Computing the derivative for Macro Precision](#computing-the-derivative-for-macro-precision)



# Introduction

In the development of the [python confidence interval library](https://github.com/jacobgil/confidenceinterval), for the analytic confidence intervals of some metrics I've been relying on results from the remarkable [Confidence interval for micro-averaged F1 and macro-averaged F1 scores](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8936911/#S10) paper *by Kanae Takahashi, Kouji Yamamoto, Aya Kuchiba and Tatsuki Koyama.*



In the paper they derive confidence intervals for Micro F1 and Macro F1 (and by extension to Micro Precision/Recall, since they are equal to Micro F1).

However there are a few common variants that the paper didn't address:
- Macro Precision
- Macro Recall
- Binary F1, which is extremely common since [it's the default in the scikit-learn library](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).


The next sections derive the confidence intervals for these missing metrics in the spirit of the paper above, using the delta method.
Some of the sections are a bit more verbose than in the paper that (elegently) combines some steps together, however I found it helpful to break it down a bit more.

These were implemented in the [python confidence interval library](https://github.com/jacobgil/confidenceinterval).

# The computation flow for confidence intervals with the delta method

This is the computation flow we're going to go through:
1. Approximate the (normal) distribution of the confusion matrix probabilities $$p_{ij}$$.
2. Express the metrics as functions of $$p_{ij}$$: $$metric(p_{ij})$$.
3. Use the delta method approximate the (normal) distrubution of $$metric(p_{ij})$$.
4. Plug in our estimate of $$p_{ij}$$ based on the observed data, and get the variance of $$metric(p_{ij})$$ 
5. Once we have the variance, we can get the confidence interval lower and upper bounds using the gaussian distribution.

# The confusion matrix multinomial distribution
$$C_{ij}$$ is the confusion matrix: the number of predictions with the ground truth category i, that were actually predicted as j.
*Note that here wee keep the [scikit-learn notation](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html), intead of the notation in the paper that's transposed.*
We have an actual observed confusion matrix $$\hat{C_{ij}}$$, but we assumed it was sampled from a distribution $$C_{ij}$$.

The core assumption here is that $$C_{ij}$$ has a multinomial distribution with parameters $$p_{ij}$$.

$$E(C_{ij}) = n p_{ij}$$

$$Cov(C_{ij}, C_{ij}) = Var(C_{ij}) = np_{ij}(1-p_{ij})$$

And when ij != kl:


$$Cov(C_{ij}, C_{kl}) = -np_{ij}p_{kl}$$


By combining the two cases above, the covariance matrix of the multinomial distribution can be written as:
$$Cov(C_{ij}, C_{kl}) = n *  [diag(p)_{ij} - (pp^T)_{ij}]$$



We don't know what $$p_{ij}$$ actually is. But our best guess for it, the maximum likelihood estimator, is:

$$\hat{p_{ij}} = \frac{C_{ij}}{n}$$.

n = $$\sum_{ij}C_{ij}$$, is the total number of predictions.

# The central limit theorem applied on the confusion matrix
$$C_{ij}$$ can be seen as the sum of n individual trial binary variables $$X_{ij}$$, where $$X_{ij}=1$$ with probability $$p_{ij}$$.

$$\hat{p_{ij}} = \frac{\sum_{k=1}^{N}{X_{ijk}}}{n} =  \frac{\hat{C_{ij}}}{n}$$

From the central limit theorem, since $$\hat{p_{ij}}$$ is the average of many variables, we know that it has a normal distribution.
We also know it's mean and covariance, since $$\hat{p_{ij}} =  \frac{\hat{C_{ij}}}{n}$$ and we know from above what the distribution of $$C_{ij}$$ is.

$$E[\hat{p}] = \frac{E[C]}{n}, Cov(\hat{p}) = \frac{Cov(C)}{n^2}$$

$$\hat{p_{ij}} \sim Normal(E[p], \frac {diag(p) - (pp^T)]}{n}$$


# Expressing the metrics in terms of the confusion matrix


- Binary F1: $$metric(p) = F1_{binary} = \frac {2p_{11} }{2p_{11} + p_{01} + p_{10}} = \frac {2p_{11} }{d}$$

- Macro Recall: $$metric(p) = R = \frac{1}{r}\sum_{i=1}^{r} \frac{p_{ii}}{\sum_j{p_{ij}}}$$

- Macro Prcecision: $$metric(p) = P = \frac{1}{r}\sum_{i=1}^{r} \frac{p_{ii}}{\sum_j{p_{ji}}}$$

$$r$$ = number  of categories


If we plug in our estimate for p, we get the (point estimation of the) metric. 

# The multi-variate delta method

The various metrics above are functions $$metric(\hat{p})$$. We know from above that $$\hat{p}$$ has approximately a normal distribution.
The multi-variate delta method gives us a recepie to get the distribution of the $$metric(\hat{p})$$:


$$metric(\hat{p}) \sim Normal(metric(p), \frac{\partial metric(p)^T}{\partial p} Cov(p) \frac{\partial metric(p)}{\partial p})$$

We also know from above that $$Cov(p) = \frac {diag(p) - pp^T}{n}$$


Now the only thing missing is to compute those derivatives!


# Computing the derivative for binary F1

$$f1 = \frac {2p_{11} }{2p_{11} + p_{01} + p_{10}} = \frac {2p_{11} }{d}$$

$$\frac{\partial f1}{\partial p_{10}} = \frac{\partial f1}{\partial p_{01}} =  -2\frac {p_{11}} {d^2} = -\frac {f1} {d}$$
$$\frac{\partial f1}{\partial p_{11}} = \frac 2 {d} - \frac {4 p_{11}} {d^2} = \frac {2(1-f1)} {d}$$

The code can be found [here](https://github.com/jacobgil/confidenceinterval/blob/main/confidenceinterval/takahashi_methods.py#L386).

# Computing the derivative for Macro Recall
$$metric(p) = R = \frac{1}{r}\sum_{i=1}^{r} \frac{p_{ii}}{\sum_j{p_{ij}}} =\frac{1}{r}\sum_{i=1}^{r} \frac{p_{ii}}{d_i} = \frac{1}{r}\sum_{i=1}^{r}R_i$$

$$\frac{\partial R}{\partial p_{ii}} =\frac{1}{r} [\frac{1}{di} - \frac{p_{ii}}{di^2}] = \frac{1}{r} \frac{1-R_i}{d_i}$$
$$\frac{\partial R}{\partial p_{ij}} = -\frac{1}{r}  \frac{p_{ii}}{d_i^2} = -\frac{1}{r}  \frac{R_i}{d_i}$$

In terms of computation, the non diagonal row elements will all be the same expression of the $$R_i$$ of that row.

The code can be found [here](https://github.com/jacobgil/confidenceinterval/blob/main/confidenceinterval/takahashi_methods.py#L260)

# Computing the derivative for Macro Precision
$$metric(p) = P = \frac{1}{r}\sum_{i=1}^{r} \frac{p_{ii}}{\sum_j{p_{ji}}} =\frac{1}{r}\sum_{i=1}^{r} \frac{p_{ii}}{d_i} = \frac{1}{r}\sum_{i=1}^{r}P_i$$

$$\frac{\partial P}{\partial p_{ii}} =\frac{1}{r} [\frac{1}{di} - \frac{p_{ii}}{di^2}] = \frac{1}{r} \frac{1-P_i}{d_i}$$

$$\frac{\partial p}{\partial p_{ji}} = -\frac{1}{r}  \frac{p_{ii}}{d_i^2} = -\frac{1}{r}  \frac{P_i}{d_i}$$

*Note* how for precision we derive for $$p_{ji}$$ instead of $$p_{ij}$$.
In terms of computation, the non diagonal columns elements will all be the same expression of the $$P_i$$ of that column.


The code can be found [here](https://github.com/jacobgil/confidenceinterval/blob/main/confidenceinterval/takahashi_methods.py#L78)