
%% primal linear formulation for SVM                
%
function main()
    clc; clear all; warning off all
    
    rand('twister',3120);
    
    path = 'libsvm';
    rmpath(genpath(path))
    addpath(genpath(path))
    
    data = dlmread('data.txt');
   
    idx  = randperm(size(data,1));
    data = data(idx,:);

    d = size(data,2)-1;
    N = size(data,1);
    
    x = data(1:20,:);
    y = x(:,end);
    x(:,end) = [];
    
    xtest = data(21:end,:);
    ytest = data(21:end,end);
    xtest(:,end) = [];

    minc = min(y);
    maxc = max(y);
    y  = 2*(y-minc)/(maxc-minc)-1;

    % LIBSVM
    ypredictlibsvm = libsvm(x,y,xtest,ytest);
    
    x     = [x, ones(size(x,1),1)];
    xtest = [xtest, ones(size(xtest,1),1)];
    
    H  = eye(d+1,d+1);
    y1 = repmat(y,1,size(x,2));
    
    A  = -y1 .* x;
    b  = -ones(length(y),1);
    lb = zeros(length(y),1);
    
    [w,fval,exitcode] = quadprog(H,[],A,b,[],[],lb)
    
    ypredictmysvm  = xtest*w;
    ypredictmysvm  = sum(ypredictmysvm,2);
    ind = ypredictmysvm > 0;
    
    ypredictmysvm(ind)  = 1;
    ypredictmysvm(~ind) = -1;

    [ypredictlibsvm ypredictmysvm];
    
    % ------------------------------------------
    y = remap(y);
    ypredictlibsvm = remap( ypredictlibsvm );
    ypredictmysvm  = remap( ypredictmysvm  );

    x(:,end) = [];
    xtest(:,end) = [];
    % plot_features([x,y]);
    plot_features([xtest,ypredictlibsvm]);
    plot_features([xtest,ypredictmysvm]);

    x = 0:.1:2;
    y = (-w(3) - x.*w(1))./w(2);
    plot(x,y,'r-')
    xlim([0 2]);
    ylim([0 2]);
    
    errorlibsvm = mre(ytest,ypredictlibsvm);
    errormysvm  = mre(ytest,ypredictmysvm);

    fprintf(1,'LIBSVM error: %f\n',errorlibsvm);
    fprintf(1,'MYSVM error: %f\n',errormysvm);
    

%%                                                
function ypredict = libsvm(x,y,xtest,ytest)
   
    configStr = '-s 0 -t 0 -c 0';
    model = svmtrain(y,x,configStr);
    ypredict = svmpredict(ones(size(ytest,1),1),xtest,model,'-b 0');
    
    model
    
    return

%%     
function y= remap(y)
    y = (y+1)/2+1;

function merror = mre(y,ypredict)

    merror = length(find((y - ypredict) ~= 0))/length(y);
    
    return