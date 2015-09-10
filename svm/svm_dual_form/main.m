
%% SVM DUAL FORMULATION                                                    
function main()
    load_libraries();
    %% some libraries loading
    
    % close all;    clc;
    [trainSetFeatures, trainSetClass] = read_data('binary_data.txt', ' ');
    
    %% testing data creation
    x1 = -2.5:0.1:2.5;
    x2 = -2.5:0.1:2.5;
    [x11,x22] = meshgrid(x1, x2);
    test_data_X = [x11(:) x22(:)];
    testSetClass = ones(size(test_data_X,1), 1); %libSVM needs but does not really uses 
	
    %% basic/common config
    Cvalue = 1;
    epsilon = 1e-9;
    
    %% ----------------------------------------------------------------------------------
    %% libsvm
    configStr = sprintf('-s 0 -t 1 -d 2 -r 1 -g 1 -c %d -e %d', Cvalue,epsilon);
    net = svmtrain(trainSetClass, trainSetFeatures, configStr)
    [test_data_Y0, accuracy] = svmpredict (testSetClass, test_data_X, net);
    
    %% ----------------------------------------------------------------------------------
    %% specific options for the my_svm_dual_train
    options = struct('C',Cvalue,'epsilon',epsilon,'kernel','polynomial','degree',2);
    tic
        svm_model    = my_svm_dual_train (trainSetFeatures, trainSetClass, options,'intpoint')
        test_data_Y1 = my_svm_dual_test ( svm_model, test_data_X );
    toc
    
    %% libsvm prediction
    plotregion(trainSetClass,trainSetFeatures,test_data_X,test_data_Y0,x1,x2,x11,x22,net.SVs);
    %% my_svm prediction
    plotregion(trainSetClass,trainSetFeatures,test_data_X,test_data_Y1,x1,x2,x11,x22,svm_model.supportVector);