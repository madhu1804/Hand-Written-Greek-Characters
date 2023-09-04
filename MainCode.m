%% Classification of Digit

% Clear data and figures
clc
clear
close all

%% Importing Data

tic
fprintf('Initializing Training Dataset...')
% Add folders
addpath(genpath(pwd))
% Get images from directory
file=dir('train_high_resolution\*.jpg');
% Import label data
Data=importdata('TrainData.xlsx');
% Extract Label
Label=Data.data;
t=toc;
fprintf('Done in %0.3f seconds\n\n',t)
tic

%% Extracting Feature

% Extract Features from Image Dataset
for i=1:length(file)
    % Read Image
    I=imread(file(i).name);
    % Resize Image
    I=imresize(I,[32 32]);
    % Binarize Image
    I=double(imbinarize(I));
    % Binary Vector Feature 
    TrainFeature(i,:)=I(:)';
    % Extract HOG Features
    hogfea = extractHOGFeatures(I,'CellSize',[8 8]);
    % Store Feature
    TrainFeatureHOG(i,:)=hogfea;
    % Display Progress
    fprintf('Extract Feature %d out of %d\n',i,length(file))
end
t=toc;
fprintf('\nExtracting Feature...Done in %0.3f seconds\n\n',t)

% Plot No. of images in each class
figure
histogram(Label)
ylabel('No. of Observations')
xlabel('Class')
title('Number of Images in each Class (Train set)')

%% Training Model

fprintf('Training Model\n')
fprintf('Model    |      Feature      |    Training Time   |   Accuracy\n')
fprintf('---------------------------------------------------------------\n')
% For Binary Vector
tic
[trainedClassifier1, validationAccuracy1] = trainClassifierEnsemble(TrainFeature,Label);
t=toc;
fprintf('Ensemble |   Binary Vector   |     %6.3f sec     |  %0.2f%%\n',t,validationAccuracy1*100)
tic
[trainedClassifier2, validationAccuracy2] = trainClassifierKNN(TrainFeature,Label);
t=toc;
fprintf('KNN      |   Binary Vector   |     %6.3f sec     |  %0.2f%%\n',t,validationAccuracy2*100)
tic
[trainedClassifier3, validationAccuracy3] = trainClassifierSVM(TrainFeature,Label);
t=toc;
fprintf('SVM      |   Binary Vector   |     %6.3f sec     |  %0.2f%%\n',t,validationAccuracy3*100)
tic
[trainedClassifier4, validationAccuracy4] = trainClassifierTree(TrainFeature,Label);
t=toc;
fprintf('Tree     |   Binary Vector   |     %6.3f sec     |  %0.2f%%\n',t,validationAccuracy4*100)
% For PCA
tic
[trainedClassifier5, validationAccuracy5] = trainClassifierEnsemblePCA(TrainFeature,Label);
t=toc;
fprintf('Ensemble |   PCA             |     %6.3f sec     |  %0.2f%%\n',t,validationAccuracy5*100)
tic
[trainedClassifier6, validationAccuracy6] = trainClassifierKNNPCA(TrainFeature,Label);
t=toc;
fprintf('KNN      |   PCA             |     %6.3f sec     |  %0.2f%%\n',t,validationAccuracy6*100)
tic
[trainedClassifier7, validationAccuracy7] = trainClassifierSVMPCA(TrainFeature,Label);
t=toc;
fprintf('SVM      |   PCA             |     %6.3f sec     |  %0.2f%%\n',t,validationAccuracy7*100)

[trainedClassifier8, validationAccuracy8] = trainClassifierTreePCA(TrainFeature,Label);
t=toc;
fprintf('Tree     |   PCA             |     %6.3f sec     |  %0.2f%%\n',t,validationAccuracy8*100)
% For HOG Feature
tic
[trainedClassifier9, validationAccuracy9] = trainClassifierEnsembleHOG(TrainFeatureHOG,Label);
t=toc;
fprintf('Ensemble |   HOG             |     %6.3f sec     |  %0.2f%%\n',t,validationAccuracy9*100)
tic
[trainedClassifier10, validationAccuracy10] = trainClassifierKNNHOG(TrainFeatureHOG,Label);
t=toc;
fprintf('KNN      |   HOG             |     %6.3f sec     |  %0.2f%%\n',t,validationAccuracy10*100)
tic
[trainedClassifier11, validationAccuracy11] = trainClassifierSVMHOG(TrainFeatureHOG,Label);
t=toc;
fprintf('SVM      |   HOG             |     %6.3f sec     |  %0.2f%%\n',t,validationAccuracy11*100)
tic
[trainedClassifier12, validationAccuracy12] = trainClassifierTreeHOG(TrainFeatureHOG,Label);
t=toc;
fprintf('Tree     |   HOG             |     %6.3f sec     |  %0.2f%%\n\n',t,validationAccuracy12*100)

% Plot Accuracy of Models
Accuracy=[validationAccuracy1 validationAccuracy2 validationAccuracy3 validationAccuracy4;...
          validationAccuracy5 validationAccuracy6 validationAccuracy7 validationAccuracy8;...
          validationAccuracy9 validationAccuracy10 validationAccuracy11 validationAccuracy12];
X = categorical({'Ensemble','KNN','SVM','Tree'});
figure
bar(X,Accuracy'*100)
xlabel('Trained Model')
ylabel('Accuracy (%)')
legend('Binary Vector','PCA','HOG')
title('Accuracy of Models (Training)')

%% Testing Model

%% Importing Data

tic
fprintf('\nInitializing Testing Dataset...')
% Add folders
addpath(genpath(pwd))
% Get images from directory
file=dir('test_high_resolution\*.jpg');
% Import label data
Data=importdata('TestData.xlsx');
% Extract Label
Label=Data.data;
t=toc;
fprintf('Done in %0.3f seconds\n\n',t)
tic

%% Extracting Feature

% Extract Features from Image Dataset
for i=1:length(file)
    % Read Image
    I=imread(file(i).name);
    % Resize Image
    I=imresize(I,[32 32]);
    % Binarize Image
    I=double(imbinarize(I));
    % Binary Vector Feature 
    TestFeature(i,:)=I(:)';
    % Extract HOG Features
    hogfea = extractHOGFeatures(I,'CellSize',[8 8]);
    % Store Feature
    TestFeatureHOG(i,:)=hogfea;
    % Display Progress
    fprintf('Extract Feature %d out of %d\n',i,length(file))
end
t=toc;
fprintf('\nExtracting Feature...Done in %0.3f seconds\n\n',t)

% Plot No. of images in each class
figure
histogram(Label)
ylabel('No. of Observations')
xlabel('Class')
title('Number of Images in each Class (Test set)')

%% Recognition of Digit

fprintf('\n\nTesting Model\n')
fprintf('Model    |      Feature      |    Training Time   |   Accuracy\n')
fprintf('---------------------------------------------------------------\n')
% For Binary Features
tic
PredictClass1=trainedClassifier1.predictFcn(TestFeature);
TestingAccuracy1=sum((PredictClass1==Label))/numel(PredictClass1);
t=toc;
fprintf('Ensemble |   Binary Vector   |     %6.3f sec     |  %0.2f%%\n',t,TestingAccuracy1*100)
tic
PredictClass2=trainedClassifier2.predictFcn(TestFeature);
TestingAccuracy2=sum((PredictClass2==Label))/numel(PredictClass2);
t=toc;
fprintf('KNN      |   Binary Vector   |     %6.3f sec     |  %0.2f%%\n',t,TestingAccuracy2*100)
tic
PredictClass3=trainedClassifier3.predictFcn(TestFeature);
TestingAccuracy3=sum((PredictClass3==Label))/numel(PredictClass3);
t=toc;
fprintf('SVM      |   Binary Vector   |     %6.3f sec     |  %0.2f%%\n',t,TestingAccuracy3*100)
tic
PredictClass4=trainedClassifier4.predictFcn(TestFeature);
TestingAccuracy4=sum((PredictClass4==Label))/numel(PredictClass4);
t=toc;
fprintf('Tree     |   Binary Vector   |     %6.3f sec     |  %0.2f%%\n',t,TestingAccuracy4*100)
% Plot Confusion plot for four ML Models
figure
subplot(221)
confusionchart(Label,PredictClass1)
title('Model: Ensemble')
subplot(222)
confusionchart(Label,PredictClass2)
title('Model: KNN')
subplot(223)
confusionchart(Label,PredictClass3)
title('Model: SVM')
subplot(224)
confusionchart(Label,PredictClass4)
title('Model: Tree')
sgtitle('Feature: Binary Vector')

% For PCA Feature
tic
PredictClass5=trainedClassifier5.predictFcn(TestFeature);
TestingAccuracy5=sum((PredictClass5==Label))/numel(PredictClass5);
t=toc;
fprintf('Ensemble |   PCA             |     %6.3f sec     |  %0.2f%%\n',t,TestingAccuracy5*100)
tic
PredictClass6=trainedClassifier6.predictFcn(TestFeature);
TestingAccuracy6=sum((PredictClass6==Label))/numel(PredictClass6);
t=toc;
fprintf('KNN      |   PCA             |     %6.3f sec     |  %0.2f%%\n',t,TestingAccuracy6*100)
tic
PredictClass7=trainedClassifier7.predictFcn(TestFeature);
TestingAccuracy7=sum((PredictClass7==Label))/numel(PredictClass7);
t=toc;
fprintf('SVM      |   PCA             |     %6.3f sec     |  %0.2f%%\n',t,TestingAccuracy7*100)
tic
PredictClass8=trainedClassifier8.predictFcn(TestFeature);
TestingAccuracy8=sum((PredictClass8==Label))/numel(PredictClass8);
fprintf('Tree     |   PCA             |     %6.3f sec     |  %0.2f%%\n',t,TestingAccuracy8*100)
% Plot Confusion plot for four ML Models
figure
subplot(221)
confusionchart(Label,PredictClass5)
title('Model: Ensemble')
subplot(222)
confusionchart(Label,PredictClass6)
title('Model: KNN')
subplot(223)
confusionchart(Label,PredictClass7)
title('Model: SVM')
subplot(224)
confusionchart(Label,PredictClass8)
title('Model: Tree')
sgtitle('Feature: PCA')

% For HOG Feature
tic
PredictClass9=trainedClassifier9.predictFcn(TestFeatureHOG);
TestingAccuracy9=sum((PredictClass9==Label))/numel(PredictClass9);
t=toc;
fprintf('Ensemble |   HOG             |     %6.3f sec     |  %0.2f%%\n',t,TestingAccuracy9*100)
tic
PredictClass10=trainedClassifier10.predictFcn(TestFeatureHOG);
TestingAccuracy10=sum((PredictClass10==Label))/numel(PredictClass10);
t=toc;
fprintf('KNN      |   HOG             |     %6.3f sec     |  %0.2f%%\n',t,TestingAccuracy10*100)
tic
PredictClass11=trainedClassifier11.predictFcn(TestFeatureHOG);
TestingAccuracy11=sum((PredictClass11==Label))/numel(PredictClass11);
t=toc;
fprintf('SVM      |   HOG             |     %6.3f sec     |  %0.2f%%\n',t,TestingAccuracy11*100)
tic
PredictClass12=trainedClassifier12.predictFcn(TestFeatureHOG);
TestingAccuracy12=sum((PredictClass12==Label))/numel(PredictClass12);
t=toc;
fprintf('Tree     |   HOG             |     %6.3f sec     |  %0.2f%%\n\n',t,TestingAccuracy12*100)
% Plot Confusion plot for four ML Models
figure
subplot(221)
confusionchart(Label,PredictClass9)
title('Model: Ensemble')
subplot(222)
confusionchart(Label,PredictClass10)
title('Model: KNN')
subplot(223)
confusionchart(Label,PredictClass11)
title('Model: SVM')
subplot(224)
confusionchart(Label,PredictClass12)
title('Model: Tree')
sgtitle('Feature: HOG')

% Plot Accuracy of Models
Accuracy=[TestingAccuracy1 TestingAccuracy2 TestingAccuracy3 TestingAccuracy4;...
          TestingAccuracy5 TestingAccuracy6 TestingAccuracy7 TestingAccuracy8;...
          TestingAccuracy9 TestingAccuracy10 TestingAccuracy11 TestingAccuracy12];
X = categorical({'Ensemble','KNN','SVM','Tree'});
figure
bar(X,Accuracy'*100)
xlabel('Trained Model')
ylabel('Accuracy (%)')
legend('Binary Vector','PCA','HOG')
title('Accuracy of Models (Testing)')

%% Display Random Class of Digits

Idx=randi([1 length(file)],1,6);
figure
for i=1:numel(Idx)
    % Read Image
    I=imread(file(Idx(i)).name);
    % Display Image
    subplot(2,3,i)
    imshow(I)
    % Resize Image
    I=imresize(I,[32 32]);
    % Binarize Image
    I=double(imbinarize(I));
    % Binary Vector Feature 
    TestFeature(i,:)=I(:)';
    % Extract HOG Features
    hogfea = extractHOGFeatures(I,'CellSize',[8 8]);
    PredictClass=trainedClassifier9.predictFcn(hogfea);
    title(['Class : ',num2str(PredictClass)])
end