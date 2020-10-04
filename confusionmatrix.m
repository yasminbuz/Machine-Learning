close all;
clc;
clear all;

%split dataset into training and test 
MainData1 = readtable('epilepsyfinal.csv');

MainData = table2array(MainData1) %convert table to array

PD = 0.20 ;  % 20% holdout for test data 

cv = cvpartition(size(MainData,1),'HoldOut',PD);%partition the data
idx= cv.test %logical function used to split 

%split main data into train and test 
Ptrain = MainData(~idx,:);
Ptest = MainData(idx,:);

Xtraining = Ptrain(:, 2:179); %X feature training
Xtest = Ptest (:, 2:179);

Ytraining = Ptrain(:,180); %Y target feature 
Ytest = Ptest(:,180);


% Training final model on test data 
MdlTest = TreeBagger(45,Xtest,Ytest,... 
    'MinLeafSize',3 ,'OOBPredictorImportance','on','NumPredictorsToSample',178);


% Prediction results of test model
YPrediction1 = MdlTest.predict(Xtest);
YPrediction = str2num(cell2mat(YPrediction1)); %converting to correct data type

[Yfit,scores] = predict(MdlTest, Xtest);

%generate confusion matrix of test results from function confusionmat
[confusion1] = confusionmat(Ytest,YPrediction , 'Order', [0,1]);
classlbl = ["Seizure" , "Non-Seizure"]
confchart = confusionchart(confusion1, classlbl, "Title" , "Confusion Matrix");

