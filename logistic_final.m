% Logistic Regression: Training data processing with full data set and producing outputs for testing data as well
function [  vec_w...                                % Weighting vector
         ] = logistic_final()

% Settings
NumExamples = 1000000;                              % Number of training examples
NumTestExamples = 262102;                           % Number of testing examples
NumSampled = 250;                                   % Number of sampled training examples
Dimension = 900;                                    % Dimension of the training examples
NumIteration = 100;                                 % Maximum number of iterations allowed
StepSize = 0.005;                                   % Step size for each iteration

% Open the files for data reading
fid_bin_TrainSet = fopen('.\\data\\bin_training.dat','rb');
fid_bin_TrainLabels = fopen('.\\data\\bin_training_label.dat','rb');
fid_bin_TestSet = fopen('.\\data\\bin_testing.dat','rb');
fid_TestResult = fopen('.\\data\\testing_label.txt','w');

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

tic

% Training
while(norm(gradient)>10)
    gradient = zeros(num_class,column);
    
    % Fetch the sampled training set and labels
    addr = randperm(NumExamples,NumSampled);
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

% Get the testing set and labels, to report the clissification results
for i=1:NumTestExamples
    x = fread(fid_bin_TestSet,Dimension,'unsigned char');
    xi = [1 (x'-avrg_overall)./sqrt(vari_overall)];
    tmp = ww*xi';
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

for i=1:num_class
    for j=1:100
        fprintf(fid_TestResult,'%d\t%d\r\n',i,TestResult(j,i));
    end
end

fclose('all');

toc

% Output
vec_w = ww;