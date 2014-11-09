% Logistic Regression: Training data processing with cross-validation (80% for training and 20% for testing)
function [  vec_w,...                               % Weighting vector
            accuracy,...                            % Pure accuracy
            map...                                  % Mean average precision
         ] = logistic()

% Settings
NumExamples = 1000000;                              % Number of training examples
NumSampled = 250;                                   % Number of sampled training examples
Dimension = 900;                                    % Dimension of the training examples
NumIteration = 100;                                 % Maximum number of iterations allowed
StepSize = 0.005;                                   % Step size for each iteration

% Open the files for data reading
fid_bin_TrainSet = fopen('.\\data\\bin_training.dat','rb');
fid_bin_TrainLabels = fopen('.\\data\\bin_training_label.dat','rb');

% Initialization
num_class = 164;
column = Dimension + 1;
ww = zeros(num_class,column);
cnt = 1;
gradient = ones(num_class,column);
avrg_overall = zeros(1,Dimension);
vari_overall = zeros(1,Dimension);
SampledTrainSet = zeros(NumSampled,Dimension);
SampledTrainLabels = zeros(NumSampled,1);
TestScores = zeros(100,num_class);
TestExamples = zeros(100,num_class);
TestResult = zeros(100,num_class);
Pki = zeros(100,num_class);

tic

% Training
while(norm(gradient)>10)
    gradient = zeros(num_class,column);
    
    % Fetch the sampled training set and labels
    addr = randperm(0.8*NumExamples,NumSampled);
    for i=1:NumSampled
        fseek(fid_bin_TrainSet,(addr(i)-1)*Dimension,'bof');
        x = fread(fid_bin_TrainSet,Dimension,'unsigned char');
        SampledTrainSet(i,:) = x';
        
        fseek(fid_bin_TrainLabels,addr(i)-1,'bof');
        y = fread(fid_bin_TrainLabels,1,'unsigned char');
        SampledTrainLabels(i) = y;
    end
    
    % Normalization
    [row_train,~] = size(SampledTrainSet);
    avrg = mean(SampledTrainSet);
    vari = var(SampledTrainSet,1);
    avrg_overall = ((cnt-1)*avrg_overall + avrg)/cnt;
    vari_overall = ((cnt-1)*vari_overall + vari)/cnt;
    SampledTrainSet = (SampledTrainSet-repmat(avrg_overall,row_train,1))./repmat(sqrt(vari_overall),row_train,1);
    trainSet_train = [ones(row_train,1) SampledTrainSet];
    trainLabels_train = SampledTrainLabels;
    
    % Iteration
    for index=1:row_train 
        xi = trainSet_train(index,:);
        yi = trainLabels_train(index);
        tmp = ww*xi';
        tmp = tmp-max(tmp);
        tmp = exp(tmp);
        prob = tmp/sum(tmp);
        gradient = gradient - repmat(prob,1,column).*repmat(xi,num_class,1);
        gradient(yi,:) = gradient(yi,:) + xi;
    end
    if(cnt<=NumIteration)
        ww = ww + StepSize/sqrt(cnt)*gradient;
    else
        break;
    end
    cnt = cnt + 1;
end

toc
tic

% Get the validation set and labels, to calculate the number of validation errors
fseek(fid_bin_TrainSet,0.8*NumExamples*Dimension,'bof');
fseek(fid_bin_TrainLabels,0.8*NumExamples,'bof');
err_cnt = 0;
for i=1:0.2*NumExamples
    x = fread(fid_bin_TrainSet,Dimension,'unsigned char');
    xi = [1 (x'-avrg_overall)./sqrt(vari_overall)];
    yi = fread(fid_bin_TrainLabels,1,'unsigned char');
    tmp = ww*xi';
    [~,yi_estimated] = max(tmp);
    err_cnt = err_cnt + (yi~=yi_estimated);
    tmp = tmp-max(tmp);
    tmp = exp(tmp);      
    prob = tmp/sum(tmp);
    [C,I] = min(TestScores);
    for j=1:num_class
        if(prob(j)>C(j))
            TestScores(I(j),j) = prob(j);
            TestExamples(I(j),j) = i;
        end
    end
end
[~,IX] = sort(TestScores,'descend');

for i=1:num_class
    TestResult(:,i) = TestExamples(IX(:,i),i);
end

% Calculate the precision of the first k predictions
for i=1:num_class
    err = 0;
    for k=1:100
        ind = TestResult(k,i);
        fseek(fid_bin_TrainLabels,0.8*NumExamples+ind-1,'bof');
        yi = fread(fid_bin_TrainLabels,1,'unsigned char');
        err = err + (yi~=i);
        Pki(k,i) = 1 - err/k;
    end
end 

fclose('all');

toc

% Output
vec_w = ww;
accuracy = 1 - err_cnt/(0.2*NumExamples);
map = sum(sum(Pki)/100)/num_class;