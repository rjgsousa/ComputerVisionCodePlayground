
% svm prediction function
function prediction = my_svm_dual_test(model,Test_data)
    
    K = my_svm_kernelfunction(Test_data, model.supportVector, model.options);
    P= K.*repmat(model.supportVectorAlphaClasses, 1, size(Test_data, 1))';
    P1 = sum(P,2)+model.bias;
    prediction = (P1 >0);

    return;