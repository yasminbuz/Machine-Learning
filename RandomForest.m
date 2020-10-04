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
%Grid search code is found here, it has been commented out as it takes
%many hours to run

%num_estimators = [100, 300, 500, 800, 1200]; 
%num_predictors = [1:178];
%leaf_size = [5, 10, 20, 25]; % maximum leaf size of 25 
%n_Trees = [10, 20, 30, 40, 50,60,70,80,90,100,150];

%for loops to output optimal hyperparameters
%for i = 1:length(n_Trees) 
    %for j = 1:20 (leaf_size)
        %for k = 1:length(num_predictors)
            
        % Training model with best hyperparameters obtained from grid search   
       % Mdl = TreeBagger(i,Xtraining,Ytraining,'OOBPrediction','On',... 
   % i,'OOBPredictorImportance','on',k); 
        %end 
    %end 
%end 


%x = [10,20,30,40,50,60,70,80,90,100] - randomly grid searches columns


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Random Forest Training Model 

%for reproducibility, controls random number generation
rng(1); 

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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training final model on test data 


MdlTest = TreeBagger(45,Xtest,Ytest,... 
    'MinLeafSize',3 ,'OOBPredictorImportance','on','NumPredictorsToSample',178);


% Prediction results of test model
YPrediction1 = MdlTest.predict(Xtest);
YPrediction = str2num(cell2mat(YPrediction1)); %converting to double data type

[Yfit,scores] = predict(MdlTest, Xtest);

%generate confusion matrix of test results from function confusionmat
[confusion1] = confusionmat(Ytest,YPrediction , 'Order', [0,1]);
classlbl = ["Seizure" , "Non-Seizure"] 
confchart = confusionchart(confusion1, classlbl, "Title" , "Confusion Matrix");

%Using perfcurve function to plot ROC-AUC curve
[Xr,Yr, T, AUC] = perfcurve(Ytest,YPrediction,1);

plot(Xr,Yr,'color','b')

set(gca,'XLim',[-0.02,0.2],'YLim',[0.5,1.02]) %manually set x axis 
xlabel('Ratio of False Positives')
ylabel('Ratio of False Negatives')
title('Performance curve for Epilepsy classification')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% To calculate the precision rate, Recall Rate, Accuracy Rate and F1 score
% from the confusion matrix

trpoRF = Yr(2,1);  %X values from confusion matrix                                                         
fapoRF = Xr(2,1);     %Y values from confusion matrix                                                      
trueneg = (1-fapoRF);  %false positives of RF Test                                                       
falseneg = (1-trpoRF);  %false negatives of RF Test                                                       
prec1RF = trpoRF/(trpoRF + fapoRF);   %precision calculations                                       
recallRF = trpoRF/(trpoRF + falseneg);  %recall calculations  

%From the above we calculated the accuracy and F1 score of the test model
accRF = (trpoRF + trueneg)/(trpoRF + trueneg +fapoRF +falseneg);                      
f1_score = 2 * (prec1RF * recallRF)/(prec1RF + recallRF);                 

%displaying above calculations
disp('The Accuracy Rate of Testing dataset is:');
disp(accRF);

disp('The precision of Testing dataset is:');
disp(prec1RF);

disp('the Recal score Testing dataset is:');
disp(recallRF);

disp('The f1-Score of the Testing dataset is:')
disp(f1_score);

disp('AUC value of the Testing dataset is:')
disp(AUC)





