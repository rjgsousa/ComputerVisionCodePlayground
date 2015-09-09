
%% Exercise two                              
%
function ex2()
    warning off all
    % clc;
    ex2pre() % main    
    [D1 D2] = ex2a();
    [parzenDensityMap1,H1] = ex2b(D1);
    fprintf(1,'2b) Smooth estimator: %f\n',H1);
    ex2c(parzenDensityMap1,D2);
    
    [parzenDensityMap2,H2] = ex2d(D1);
    H2
    fprintf(1,'2d) Smooth estimator: %f\n',H2);
    ex2c(parzenDensityMap2,D2);

    % error with training set
    fprintf(1,'------------------------------\n');
    fprintf(1,'Evaluate Error for the training set (map1)\n');
    ex2c(parzenDensityMap1,D1);
    fprintf(1,'Evaluate Error for the training set (map2)\n');
    ex2c(parzenDensityMap2,D1);

    ex2e(D1,D2,H1)
    % close all
    return

%% Exercise two (synthetic dataset generation)
%
function ex2pre()
    sigma = [.3 0; 0 .5];
    D1 = gauss([70 50],[0 0; 1 1],cat(3,sigma,sigma));
    D1.data = D1.data + .25*randn(120,2);

    [D1, D2] = gendat(D1,.4);

    save 'synthetic.mat' D1
    save 'synthetic_test.mat' D2

%% Exercise 2a) dataset loading                  
%
function [D1 D2] = ex2a()
    
    D1 = load('synthetic.mat');
    D1 = D1.D1;

    D2 = load('synthetic_test.mat');
    D2 = D2.D2;
    
    return
    
%% Exercise 2b) Parzen Classifier
%
function [parzenDensityMap,H] = ex2b(D1)
% compute the optimum smooth parameter H
    [parzenDensityMap,H] = parzenc(D1);
    
    figure
    subplot(2,2,1)
    scatterd(D1);
    plotc(parzenDensityMap)
    plotm(parzenDensityMap)
    
    subplot(2,2,2)
    scatterd(D1);
    plotm(parzenDensityMap,3)
    
    return
    
%% Exercise 2c) error evaluation         
%
function ex2c(W,D)
    E = testd(D*W);
    fprintf(1,'Classification Error %f\n',E);
    return
    
function [parzenDensityMap,H] = ex2d(D1)
% density estimator
    [parzenDensityMap,H] = parzendc(D1);
    
    subplot(2,2,3)
    scatterd(D1);
    plotc(parzenDensityMap)
    plotm(parzenDensityMap)

    subplot(2,2,4)
    scatterd(D1);
    plotm(parzenDensityMap,3)
    return
    
function ex2e(D1,D2,H)
    parzenDensityMap = parzenm(D1,H);
    
    fprintf(1,'2e) Smooth estimator: %f\n',H);
    Density = D2*parzenDensityMap

    return
    
    