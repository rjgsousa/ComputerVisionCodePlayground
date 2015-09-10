
%% data loading function
%% trainSetFeatures - returns the data features
%% trainSetClass    - returns the data true class
function [trainSetFeatures,trainSetClass] = read_data(file,sep)
    
    data = dlmread(file,sep);
    trainSetClass = data(:,end);
    trainSetFeatures = data(:,1:end-1);

    return;