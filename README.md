Large Scale Image Classification Based on a Multi-Class Logistic Regression Model
=======================

## Introduction
This is a project, which provides efficient and accurate multi-class classification based on logistic regression for large-scale data. The compputation cost was reduced considerably by random sampling in training examples for each iteration. The approach was tested on a large data pool with 1,000,000 examples, which have 900 features and span a wide variety of 164 classes. It achieved a mean average precision up to 50% with training time limited to only 6 hours on a regular PC.

## Table of Contents
* What it can do
* What it includes
* How it works
* Contributors
* Additional information

## What it can do
Learn the weighting vector of the logistic regression model efficiently from a large-scale training data set, and produce accurate classification results for testing examples.

## What it includes
* main.m: main script
* txt2bin.m: convert data file(s) in text format into binary format
* logistic.m: training data processing with cross-validation (80% for training and 20% for testing)
* logistic_final.m: training data processing with full data set and producing outputs for testing data as well
* Presentation Slides.pdf & Project Report.pdf: supporting documents

## How it works
* Convert data files not in binary format into binary (use txt2bin.m if original files are in text format)
* Determine global settings according to the specific problem and given data set
  ** NumExamples: Number of training examples
  ** NumTestExamples: Number of testing examples
  ** Dimension: Dimension of the training examples
* Run logistic.m to determine appropriate settings for the following which produce best cross validation results
  ** NumSampled: Number of sampled training examples for each iteration
  ** NumIteration: Maximum number of iterations allowed
  ** StepSize: Step size for each iteration
* Clone the obtained settings above to logistic_final.m
* Run logistic_final.m to produce the weighting vector of the logistic regression model and classification results for given testing examples

## Contributors
* Tianlong Song

## Additional information
Please refer to the presentation slides and project report in this repository.
