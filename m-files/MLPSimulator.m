% clc; clear;
load('wdbc_MLP_Learning_Testing_Validation_Data.mat');

% assign data sets
LearningSet = Learning_wdbc_MLP;
TestingSet = Testing_wdbc_MLP;
ValidationTestingSet = Testing;
numOutputNeurons = 1;

% assign defualt variables
maxIter = 5000;
thresh = 0.1;
validThresh = 0.1;

printRate = 1000;
numExperiments = 5;
numNeurons = 8;
EarlyStop = 2500;

fprintf('This MLP simulator employs three different learning techniques\n');
fprintf('on the wdbc_MLP_Learning_Testing_Validation_Data dataset\n');





experiment_prompt = '\nPlease enter how many experiments you would like to perform. \nEnter 0 for the default amount of 5: ';
x = input(experiment_prompt);

neuron_prompt = '\nPlease enter how many hidden neurons you would like, 6-10 are recommended. \nEnter 0 to test with default amount of 8: ';
y = input(neuron_prompt);

iter_prompt = '\nPlease enter the Iteration Limit. \nEnter 0 for default value of 5000:';
iter_input=input(iter_prompt);

early_stop_prompt ='\nPlease enter the Early Stop thresh. \nEnter 0 for default value of 2500 \n or enter the same value you entered for Iteration Limit for no Early Stop:';
early_stop_input=input(early_stop_prompt);

if ~x==0
   numExperiments = x; 
end

if ~y==0
    numNeurons = y;
end

if ~iter_input==0
    maxIter = iter_input;
end

if ~early_stop_input==0 
    EarlyStop = early_stop_input;
elseif early_stop_input >= maxIter
    EarlyStop = maxIter;
end

% make space for storing data after tests
allCR = zeros(numExperiments,1);
allIter = zeros(numExperiments,1);
allLRMSE = zeros(numExperiments,1);
allTRMSE = zeros(numExperiments,1);

allCR_Val = zeros(numExperiments,1);
allIter_Val = zeros(numExperiments,1);
allLRMSE_Val = zeros(numExperiments,1);
allTRMSE_Val = zeros(numExperiments,1);
allLRMSEV_Val = zeros(numExperiments,1);

allCR_IncVal = zeros(numExperiments,1);
allIter_IncVal = zeros(numExperiments,1);
allLRMSE_IncVal = zeros(numExperiments,1);
allTRMSE_IncVal = zeros(numExperiments,1);
allLRMSEV_IncVal = zeros(numExperiments,1);


for i=1:numExperiments
    fprintf('\n------MLPLearning.m------\n');
    fprintf('Experiment: %d\n',i);
    [tempNetW, allIter(i,1), allLRMSE(i,1)] = LearningMLP(LearningSet,numNeurons,numOutputNeurons,thresh,maxIter,printRate);
    [allTRMSE(i,1),allCR(i,1)] = TestingMLP2(TestingSet,numNeurons,tempNetW);
    fprintf('---------------------------\n');
end

valThresh=min(allTRMSE);

for i=1:numExperiments
    fprintf('\n------LearningMLPWithValidation.m------\n');
    fprintf('Experiment: %d\n',i);
    [tempNetW_Val, allIter_Val(i,1),allLRMSE_Val(i,1),allLRMSEV_Val(i,1)] = LearningMLPWithValidation(LearningSet,ValidationSet,numNeurons,numOutputNeurons,valThresh,maxIter,printRate,EarlyStop);
    [allTRMSE_Val(i,1),allCR_Val(i,1)] = TestingMLP2(ValidationTestingSet,numNeurons,tempNetW_Val);
    fprintf('-----------------------------------------\n');
end

for i=1:numExperiments
    fprintf('\n------IncLearningwValidation.m------\n');
    fprintf('Experiment: %d\n',i);
    [tempNetW_IncVal, allIter_IncVal(i,1),allLRMSE_IncVal(i,1),allLRMSEV_IncVal(i,1)] = IncLearningwValidation(LearningSet,numNeurons,numOutputNeurons,valThresh,maxIter,printRate,ValidationSet,EarlyStop);
    [allTRMSE_IncVal(i,1),allCR_IncVal(i,1)] = TestingMLP2(ValidationTestingSet,numNeurons,tempNetW_IncVal);
    fprintf('--------------------------------------\n');
end

meanCR = mean(allCR);
meanCR_Val = mean(allCR_Val);
meanCR_IncVal = mean(allCR_IncVal);

experiments = [1:numExperiments]';

LearningErrorResults = table;
LearningErrorResults.Experiment = experiments;
LearningErrorResults.Iterations = allIter;
LearningErrorResults.LearningRMSE = allLRMSE;
LearningErrorResults.TestingRMSE = allTRMSE;
LearningErrorResults.ClassificationRate = allCR;

ValidationErrorResults = table;
ValidationErrorResults.Experiment = experiments;
ValidationErrorResults.Iterations = allIter_Val;
ValidationErrorResults.LearningRMSE = allLRMSE_Val;
ValidationErrorResults.ValidationRMSE = allLRMSEV_Val;
ValidationErrorResults.TestingRMSE = allTRMSE_Val;
ValidationErrorResults.ClassificationRate = allCR_Val;

IncrementalLearningResults = table;
IncrementalLearningResults.Experiment = experiments;
IncrementalLearningResults.Iterations = allIter_IncVal;
IncrementalLearningResults.LearningRMSE = allLRMSE_IncVal;
IncrementalLearningResults.ValidationRMSE = allLRMSEV_IncVal;
IncrementalLearningResults.TestingRMSE = allTRMSE_IncVal;
IncrementalLearningResults.ClassificationRate = allCR_IncVal;

MeanResults = table;
MeanResults.LearningError = meanCR;
MeanResults.ValidationError = meanCR_Val;
MeanResults.IncrementalLearning = meanCR_IncVal;

fprintf('\nLearning Error Results\n');
disp(LearningErrorResults);
fprintf('\nValidation Error Results\n');
disp(ValidationErrorResults);
fprintf('\nIncremental Learning Results\n');
disp(IncrementalLearningResults);
fprintf('\nMean Results\n');
disp(MeanResults);



        
        
        
        
        
        
        
        
        
        
        
        