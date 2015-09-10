function K = my_svm_kernelfunction(V1, V2, options)
    kernel = options.kernel;

    switch kernel
      case 'linear'
        % linear transformation
        K = V1*V2';            
      case 'polynomial'
        % polynomial
        K = (V1*V2'+1).^options.degree;
      case 'rbf'
        % radial basis function
        [N1, d]  = size(V1);
        [N2, d2] = size(V2);
        
        dif  = combvec (V1', V2');
        dif1 = (dif(1:d, :)- dif(1+d:end, :));
        S = sum((dif1.^2),1);
        K = exp(-options.gamma.*reshape(S, N1, []));
        
      otherwise
        error('Unknown kernel function');            
    end
    K = double(K);
    
    return;