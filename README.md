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
* main.m--main script
* txt2bin.m--convert data file(s) in text format into binary format
* logistic.m--training data processing with cross-validation
* logistic_final.m--training data processing with full data set and producing outputs for testing data
* Presentation Slides.pdf & Project Report.pdf--Supporting documents

## How it works
* Convert any data files not in binary format into binary

## Contributors
* Tianlong Song

## Additional information
Please refer to the presentation slides and project report in this repository.

