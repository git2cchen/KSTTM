% A demo for KSTTM, change 'flag' to use K-STTM-Sum or for K-STTM-Prod 
clear all

% some parameters in VBMF  d{i} = VBMF(Y, cacb, sigma2);
cacb = 100;
sigma2 = 0.1; % should be less than 1 in this case, the larger, the less 
% singualr values. Play with it to get a good performance.

% model parameters 
class_num=2; % class number
weight=1; % weight on the first and second modes of the tensor data,
% which determines the importance of those two modes compared with the
% third tensor mode, no need to change the value if no obvious clue.
flag='p';% 'a' for K-STTM-Sum, 'p' for K-STTM-Prod

% fMRI data preparation
load data_04799.mat
tensor_data=reshape(tensor_data,320,[]);
train_X=tensor_data([1:100,161:260],:);
test_X=tensor_data([101:160,261:end],:);
train_labels=labels([1:100,161:260]);
test_labels=[-ones(60,1);ones(60,1)];

A=tabulate(train_labels);
train_samples{1}=train_X(1:A(1,2),:);
for i=2:class_num
    train_samples{i}=train_X(1+sum(A(1:i-1,2)):sum(A(1:i,2)),:);
end

% image classification
for c1=1:class_num-1
    for c2=c1+1:class_num
      
        % training and valid data and label preparation
        samplenum=70;
        validnum=30;
        traindata=[train_samples{c1}(1:samplenum,:);train_samples{c2}(1:samplenum,:)];
        trainingL=[-ones(size(train_samples{c1}(1:samplenum,:),1),1);ones(size(train_samples{c2}(1:samplenum,:),1),1)];
        validdata=[train_samples{c1}(end-validnum+1:end,:);train_samples{c2}(end-validnum+1:end,:)]; 
        validL=[-ones(size(train_samples{c1}(end-validnum+1:end,:),1),1);ones(size(train_samples{c2}(end-validnum+1:end,:),1),1)];
        X=traindata;
        Y=trainingL;
        N=size(X,1);
        X=reshape(X,[N 64 64 8]);
        X=permute(X,[1 4 2 3]);
        
        d=3;
        X=permute(X,[2 3 4 1]);
        [u,s,v] = VBMF(reshape(X,size(X,1),[]), cacb, sigma2);
        TT{1}=reshape(u,[1 size(u,1) size(u,2)]);
        for t=2:d-1
            [u,s,v] = VBMF(reshape(s*v',size(u,2)*size(X,t),[]), cacb, sigma2);
            TT{t}=reshape(u,[size(u,1)/size(X,t) size(X,t) size(u,2)]);
        end
        TT{d}=reshape(s*v',size(u,2),size(X,3),size(X,4));
        U_common=reshape(TT{1},size(TT{1},1)*size(TT{1},2),size(TT{1},3))*reshape(TT{2},size(TT{2},1),size(TT{2},2)*size(TT{2},3));
        U_common=reshape(U_common,size(TT{1},2)*size(TT{2},2),size(TT{2},3));
        traindata_temp = reshape(X,[8*64,N*64]);
        TT{d}=reshape(pinv(U_common)*traindata_temp,[size(TT{2},3),64,N]);
        X=TT;
        
        validdata=reshape(validdata,[validnum*2 64 64 8]);
        validdata=permute(validdata,[4 2 3 1]);
        validdata_temp = reshape(validdata,[64*8,validnum*2*64]);
        TT_valid{3}=reshape(pinv(U_common)*validdata_temp,[size(TT{2},3),64,validnum*2]);
        TT_valid{1}=TT{1};
        TT_valid{2}=TT{2};
        validdata=TT_valid;
       
        % training and validation start
        sigmarange =[0.1 1 10 100 1000];
        Crange = [1e-3 1e-2 1e-1 1e-0 1e1 1e2 1e3];
        for sigma_i = 1:size(sigmarange,2)
            sigma=sigmarange(sigma_i);
            tic
            [ K] = kernel_mat( X, N,d,sigma,weight,flag);
            fprintf('kernel matrix costs time is   ')
            toc
            for C_i = 1:size(Crange,2)
                sigma=sigmarange(sigma_i);
                C=Crange(C_i);
                [ alpha, b] = svm_solver( K, Y, C, N);
                % for validation
                tic
                Ypred = predict(validdata, alpha, b, X, Y, sigma,d,weight,flag);
                fprintf('validation costs time is:   ')
                toc
                Ypred(Ypred>0)=1;
                Ypred(Ypred<0)=-1;
                truelabel=[-ones(validnum,1);ones(validnum,1)];
                diff=Ypred-truelabel;
                diff(diff~=0)=1;
                valid_error(sigma_i,C_i)=sum(diff)/(validnum*2);
                
                % compute the testing error
                XX=test_X;
                N_test=size(XX,1);
                XX=reshape(XX,[N_test 64 64 8]);
                XX=permute(XX,[4 2 3 1]);
                testdata_temp = reshape(XX,[64*8,N_test*64]);
                TT_test{3}=reshape(pinv(U_common)*testdata_temp,[size(TT{2},3),64,N_test]);
                TT_test{1}=TT{1};
                TT_test{2}=TT{2};
                XX=TT_test;
                scoremat=zeros(N_test,class_num);
                Ypred = predict(XX, alpha, b, X, Y, sigma,d,weight,flag);
                Ypred(Ypred>0)=1;
                Ypred(Ypred<0)=-1;
                diff=Ypred-test_labels;
                diff(diff~=0)=1;
                test_error(sigma_i,C_i)=sum(diff)/N_test
            end
        end
    end
end
