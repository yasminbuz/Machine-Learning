close all;
clc;
clear all;

%split dataset into training and test 
MainData1 = readtable('epilepsyfinal.csv');

MainData = table2array(MainData1) %convert table to array

PD = 0.20 ; % 20% holdout for test data 

cv = cvpartition(size(MainData,1),'HoldOut',PD); %partition the data
idx= cv.test %logical function used to split 

%split main data into train and test 
Ptrain = MainData(~idx,:);
Ptest = MainData(idx,:);

Xtraining = Ptrain(:, 2:179); %X feature training
Xtest = Ptest (:, 2:179); %X feature test 

Ytraining = Ptrain(:,180); %Y target feature training
Ytest = Ptest(:,180); %Y target test 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Random Forest Model Training model 

%Training model on training data, using hyperparameter selection from grid
%search 
Mdl = TreeBagger(70,Xtraining,Ytraining,'OOBPrediction','On',... 
    'MinLeafSize',3 ,'OOBPredictorImportance','on','NumPredictorsToSample',178);

%Validation of training model using oobError, for miss-classification
%probability result
BagError = oobError(Mdl); 
Bpredict = oobPredict(Mdl);

    
% Ploting OOB error to evaluate training model performance 
figure
plot(BagError)
xlabel('The Number of Grown Trees')
ylabel('The Out-of-Bag Classification Error')
title ('The OOB error for Training Model')
