function model = my_svm_dual_train(features, classes, options, qqprog)
    
    C   = options.C;
    tol = options.epsilon;
    %%  svm options

    %% train data size
    N = size(features,1);
    
    %% when targets have values equal to zero,
    %% the values in the hessian matrix are annulated
    %% so, this is exchanged towards Vapnick notation
    maxC = max(classes);
    minC = min(classes);
    classes = (classes*2-minC-maxC)/(maxC-minC);
    
    classes_matrix = repmat(classes, 1, N);

    %% aplies a kernel (linear, polynomial, and so on)
    K = my_svm_kernelfunction( features, features, options );

    %% Hmatrix calculation
    H=K.*classes_matrix.*classes_matrix';

    %% f is multiplied with (-1) because we are minimizing
    %% whereas dual form maximizes..
    f = -ones(N,1);
    Aeq = classes';
    beq = 0;
    lb = zeros(N,1);
    ub = C.*(ones(N,1));
    
    % Finder of minimums of the problem 
    %% alpha are the lagrange multipliers
    switch qqprog
      case 'quadprog'
        quadprog_options = optimset('MaxIter',5000,'TolFun',1e-6,'TolX', 1e-6);
        [alpha, fval, exitflag, output, lambda] = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],quadprog_options);
      case 'intpoint'
        optimizer = intpoint_pr('sigfig_12_maxiter_1000_margin_0.005_bound_10');
        [alpha, dual, how] = optimize(optimizer, f, H, Aeq, beq, lb, ub);
    end
    
    %% we want only the lagrange multpliers with values above 0
    %% values equal to zero do nothing in the prediction
    supportVectorIdx = (alpha  > tol);
    supportVector = features(supportVectorIdx,:);
    supportVectorAlphaClasses = classes(supportVectorIdx).*alpha(supportVectorIdx);
    
    %% b value assessment
    supportVectorIdxM = find(alpha > tol & alpha < C-tol);
    supportVectorM = features(supportVectorIdxM,:);
    supportVectorClassesM = classes(supportVectorIdxM);
    
    % mathematical computation for the Radial form
    K = my_svm_kernelfunction(supportVector, supportVectorM, options);
    supportVectorAlphaClassesRepetition = repmat(supportVectorAlphaClasses, 1,length(supportVectorM));
    bias = supportVectorClassesM' - sum(supportVectorAlphaClassesRepetition.*K ,1);
    bias = (1/length(bias))*(sum(bias));
    
    model = struct('supportVector', supportVector, 'supportVectorAlphaClasses', supportVectorAlphaClasses, 'bias', bias, 'options',options);
    
    return;