    
%% Main Function for exercise 3
function knn()
    warning off all

    ex3pre();
    [D1 D2] = ex3a();
    fprintf(1,'Exercise 3.a\n');
    knnMap = ex3a1(D1,D2,1);
    fprintf(1,'Exercise 3.b\n');
    bestK = ex3b(D1);
    fprintf(1,'Exercise 3.c\n');
    bestK = ex3c(D1,5);
    fprintf(1,'Exercise 3.c)iii)\n');
    bestK = ex3c3(D1,5);

    fprintf(1,'Exercise 3.d)\n');
    knnMap = ex3a1(D1,D2,bestK);

    fprintf(1,'Exercise 3.e)\n');
    ex3d(D1);
    
    ex3e(D1);
    
    return
    

%% Exercise (synthetic dataset generation)
%
function ex3pre()
    sigma = [.3 0; 0 .5];
    D1 = gauss([70 50],[0 0; 1 1],cat(3,sigma,sigma));
    D1.data = D1.data + .25*randn(120,2);

    [D1, D2] = gendat(D1,.4);

    save 'synthetic.mat' D1
    save 'synthetic_test.mat' D2
    
    
%% Exercise 3 - dataset loading                  
%
function [D1 D2] = ex3a()
    
    D1 = load('synthetic.mat');
    D1 = D1.D1;

    D2 = load('synthetic_test.mat');
    D2 = D2.D2;
    
    return

%% Exercise 3a - train classifier
function [knnMap] = ex3a1( D1, D2, K )
    
    knnMap = knnc(D1,K);
    testMap(knnMap,D2);
    
    return

%%  Evaluate error                 
function testMap(W,D)
    E = testc( D*W );
    fprintf(1,'Error %f\n',E);
    return
    
%% leave one out                  
function bestK = ex3b(D1)
    minError = inf;

    for K=1:5
        E = testk(D1,K);
        if E < minError
            minError = E;
            bestK    = K;
            fprintf(1,'New best K %d obtained error of %f\n',K,minError);
        end
    end
    
    return
    
%% cross validation                
function bestK = ex3c(D1,nfolds)
    minError = inf;
    bestK  = 0;
    E      = [];
    nelem  = size( D1, 1 );
    fnelem = round(nelem / 5);
    
    [folds, data] = split_data(D1,fnelem);
    for K=1:5
        for n = 1:nfolds
            foldsind = setxor(n,1:nfolds);
            train = [];
            for j = 1:length(foldsind)
                train = [train; data(folds{foldsind(j)},:)];
            end
            val   = data(folds{n},:);
            train = dataset(train(:,1:end-1),train(:,end));
            val   = dataset(val(:,1:end-1),val(:,end));
            
            W = knnc(train,K);
            E = [E testc(val,W)];
        end
        if mean(E) < minError
            minError = mean(E);
            bestK = K;
            fprintf(1,'New best K %d obtained error of %f\n',K,minError);
        end
    end
    
    return

function [folds, data] = split_data(D1,fnelem)
    folds = {};
    
    data = [+D1 D1.nlab];
    idx  = randperm(size(D1,1));
    data = data(idx,:);
    j = 1;
    for i=1:fnelem:size(D1,1)
        idx = min(i:i+fnelem-1,size(D1,1));
        folds{j} = idx;
        j = j + 1;
    end

    return
    
%%                                                             
function bestK = ex3c3(D1,nfolds)
    minError = inf;
    for K = 1:5
        W = knnc([],K);
        E = crossval(D1,W,nfolds);
        if E < minError
            minError = E;
            bestK = K;
            fprintf(1,'New best K %d obtained error of %f\n',K,minError);
        end
        
    end
    return
    
function ex3d(D1)

    j = 1;
    for K = 1:5
        W = knnc(D1,K);
        
        figure
        subplot(2,1,j)
        scatterd(D1)
        plotc(W)
        plotm(W)
        
        W = knnm(D1,K);
        subplot(2,1,j+1);
        plotm(W,3)
    end
    return
    
function ex3e(D1)
    W = knnc(D1,3)
    
    scatterd(D1)
    plotc(W)