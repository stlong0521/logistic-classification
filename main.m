clc;
clear all;
addpath(genpath(pwd));

% Convert data file(s) in text format into binary format
% txt2bin;

% Training data processing with cross-validation (80% for training and 20% for testing)
[vec_w,accuracy,map] = logistic();

% Training data processing with full data set and producing outputs for testing data as well
% vec_w = logistic_final();