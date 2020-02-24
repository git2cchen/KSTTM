function [K] = kernel_mat(X,N,d,sigma,weight,flag)
% [K] = kernel_mat(X,N,d,sigma,weight,flag)
% -------------
% Kernel matrix construction for K-STTM-Prod and K-STTM-Prod given the trai
% ning TT-format data, number of training smaples, the order of tensor
% data, gaussian kernel parameter sigma, weight on the first and second
% modes of the tensor data, and the flag.

% X         =   the training TT-format data,
%
% N         =   number of training smaples,
%
% d         =   the order of tensor data,
%
% sigma     =   gaussian kernel width parameter sigma,
%
% weight    =   weight on the first and second modes of the tensor data,
% which determince the importance of those two modes compared with the
% third tensor modes,
%
% flag      =   either 'a' or 'p', corresponding to K-STTM-Sum and
% K-STTM-Prod,
%
% K         =   The learned kernel matrix.
%
% Reference
% ---------
%
% Kernelized Support Tensor Train Machines

% 20/02/2020, Cong CHEN

K = zeros(N);

%     scalable and relatively fast way to compute kernel matrix K
if flag=='a'
    
    for s=1:N
        N=size(X{d},3);
        Kfast = zeros(1,N);
        x=X;
        x{d}=X{d}(:,:,s);
        
        if s==1
            i=1;
            Ktemp=zeros(size(x{i},1)*size(x{i},3),N*size(X{i},1)*size(X{i},3));
            % first compute the -2*xi*xj matrix
            Xtemp=reshape(permute(x{i},[1 3 2]),[size(x{i},1)*size(x{i},3),size(x{i},2)]);
            Xtemp1=reshape(permute(X{i},[1 3 2]),[size(X{i},1)*size(X{i},3),size(X{i},2)]);
            temp=Xtemp*Xtemp1';
            Ktemp=Ktemp+repmat((-2)*temp,1,N);
            % secondly compute the xi^2+xj^2 matrix
            temp=sum(Xtemp.^2,2);
            temp1=repmat(temp,1,size(X{i},1)*size(X{i},3));
            temp=sum(Xtemp1.^2,2);
            temp2=repmat(temp,1,size(x{i},1)*size(x{i},3));
            temp3=temp1+temp2';
            Ktemp=Ktemp+repmat(temp3,1,N);
            Ktemp_1=weight*exp(Ktemp./(-2*sigma^2));
            
            % linear kernel, replace line 46-59 with the following codes if a linear kernel is
            % selected to be applied on this mode.  Modify this in predict.m also
			%Ktemp=zeros(size(x{i},1)*size(x{i},3),N*size(X{i},1)*size(X{i},3));
            %Xtemp=reshape(permute(x{i},[1 3 2]),[size(x{i},1)*size(x{i},3),size(x{i},2)]);
            %Xtemp1=reshape(permute(X{i},[1 3 2]),[size(X{i},1)*size(X{i},3),size(X{i},2)]);
            %temp=Xtemp*Xtemp1';
            %Ktemp_1=Ktemp+repmat(temp,1,N);
			
			%polynomial kernel, replace line 46-59 with the following codes if a polynomial kernel is
            % selected to be applied on this mode.  Modify this in predict.m also
            %polyorder=2;
            %b_p=0;
			%Ktemp=zeros(size(x{i},1)*size(x{i},3),N*size(X{i},1)*size(X{i},3));
            %Xtemp=reshape(permute(x{i},[1 3 2]),[size(x{i},1)*size(x{i},3),size(x{i},2)]);
            %Xtemp1=reshape(permute(X{i},[1 3 2]),[size(X{i},1)*size(X{i},3),size(X{i},2)]);
            %temp=Xtemp*Xtemp1';
			%temp=(temp+b_p).^polyorder;
            %Ktemp_1=Ktemp+repmat(temp,1,N);
            
            
            i=2;
            Ktemp=zeros(size(x{i},1)*size(x{i},3),N*size(X{i},1)*size(X{i},3));
            % first compute the -2*xi*xj matrix
            Xtemp=reshape(permute(x{i},[1 3 2]),[size(x{i},1)*size(x{i},3),size(x{i},2)]);
            Xtemp1=reshape(permute(X{i},[1 3 2]),[size(X{i},1)*size(X{i},3),size(X{i},2)]);
            temp=Xtemp*Xtemp1';
            Ktemp=Ktemp+repmat((-2)*temp,1,N);
            
            
            % secondly compute the xi^2+xj^2 matrix
            temp=sum(Xtemp.^2,2);
            temp1=repmat(temp,1,size(X{i},1)*size(X{i},3));
            temp=sum(Xtemp1.^2,2);
            temp2=repmat(temp,1,size(x{i},1)*size(x{i},3));
            temp3=temp1+temp2';
            Ktemp=Ktemp+repmat(temp3,1,N);
            Ktemp_2=weight*exp(Ktemp./(-2*sigma^2));
        end
        
        % i=d
        Xtemp=x{d};
        Xtemp1=reshape(permute(X{d},[1 3 2]),[size(X{d},1)*size(X{d},3),size(X{d},2)]);
        temp=Xtemp*Xtemp1';
        prodtemp=(-2)*temp;
        temp=sum(Xtemp.^2,2);
        temp1=repmat(temp,1,size(X{d},1)*size(X{d},3));
        temp=sum(Xtemp1.^2,2);
        temp2=repmat(temp,1,size(x{d},1));
        squaretemp=temp1+temp2';
        
        Ktemp_d=exp((prodtemp+squaretemp)./(-2*sigma^2));
        
        Ktemp_final=zeros(size(Ktemp_2,1),size(Ktemp_2,2));
        for j=1:N
            Ktemp_final(:,1+size(Ktemp_2,1)*(j-1):size(Ktemp_2,1)*j)=Ktemp_final(:,1+size(Ktemp_2,1)*(j-1):size(Ktemp_2,1)*j)+kron(ones(size(Ktemp_d,1)),Ktemp_1(:,1+size(Ktemp_1,1)*(j-1):size(Ktemp_1,1)*j))+kron(Ktemp_d(:,1+size(Ktemp_d,1)*(j-1):size(Ktemp_d,1)*j),ones(size(Ktemp_1,1)));
        end
        
        Ktemp_final=Ktemp_final+Ktemp_2;
        a=size(X{2},1)*size(X{2},3);
        tranMat=zeros(N,a*N);
        for j=1:N
            tranMat(j,1+(j-1)*a:j*a)=ones(1,a);
        end
        Kfast=Kfast+ones(1,a)*Ktemp_final*tranMat';
        K(s,:)=Kfast;
    end
end

if flag=='p'
    %     scalable and relatively fast way to compute kernel matrix K
    
    for s=1:N
        N=size(X{d},3);
        Kfast = zeros(1,N );
        x=X;
        x{d}=X{d}(:,:,s);
        if s==1
            % i=1
            i=1;
            Ktemp=zeros(size(x{i},1)*size(x{i},3),N*size(X{i},1)*size(X{i},3));
            % first compute the -2*xi*xj matrix
            Xtemp=reshape(permute(x{i},[1 3 2]),[size(x{i},1)*size(x{i},3),size(x{i},2)]);
            Xtemp1=reshape(permute(X{i},[1 3 2]),[size(X{i},1)*size(X{i},3),size(X{i},2)]);
            temp=Xtemp*Xtemp1';
            Ktemp=Ktemp+repmat((-2)*temp,1,N);
            
            % secondly compute the xi^2+xj^2 matrix
            temp=sum(Xtemp.^2,2);
            temp1=repmat(temp,1,size(X{i},1)*size(X{i},3));
            temp=sum(Xtemp1.^2,2);
            temp2=repmat(temp,1,size(x{i},1)*size(x{i},3));
            temp3=temp1+temp2';
            Ktemp=Ktemp+repmat(temp3,1,N);
            Ktemp_1=weight*exp(Ktemp./(-2*sigma^2));
            
            % linear kernel, replace line 140-154 with the following codes if a linear kernel is
            % selected to be applied on this mode. Modify this in predict.m also
			%Ktemp=zeros(size(x{i},1)*size(x{i},3),N*size(X{i},1)*size(X{i},3));
            %Xtemp=reshape(permute(x{i},[1 3 2]),[size(x{i},1)*size(x{i},3),size(x{i},2)]);
            %Xtemp1=reshape(permute(X{i},[1 3 2]),[size(X{i},1)*size(X{i},3),size(X{i},2)]);
            %temp=Xtemp*Xtemp1';
            %Ktemp_1=Ktemp+repmat(temp,1,N);
			
			%polynomial kernel, replace line 140-154 with the following codes if a polynomial kernel is
            % selected to be applied on this mode.  Modify this in predict.m also
            %polyorder=2;
            %b_p=0;
			%Ktemp=zeros(size(x{i},1)*size(x{i},3),N*size(X{i},1)*size(X{i},3));
            %Xtemp=reshape(permute(x{i},[1 3 2]),[size(x{i},1)*size(x{i},3),size(x{i},2)]);
            %Xtemp1=reshape(permute(X{i},[1 3 2]),[size(X{i},1)*size(X{i},3),size(X{i},2)]);
            %temp=Xtemp*Xtemp1';
			%temp=(temp+b_p).^polyorder;
            %Ktemp_1=Ktemp+repmat(temp,1,N);
            
            % i=2
            i=2;
            Ktemp=zeros(size(x{i},1)*size(x{i},3),N*size(X{i},1)*size(X{i},3));
            % first compute the -2*xi*xj matrix
            Xtemp=reshape(permute(x{i},[1 3 2]),[size(x{i},1)*size(x{i},3),size(x{i},2)]);
            Xtemp1=reshape(permute(X{i},[1 3 2]),[size(X{i},1)*size(X{i},3),size(X{i},2)]);
            temp=Xtemp*Xtemp1';
            Ktemp=Ktemp+repmat((-2)*temp,1,N);
            
            % secondly compute the xi^2+xj^2 matrix
            temp=sum(Xtemp.^2,2);
            temp1=repmat(temp,1,size(X{i},1)*size(X{i},3));
            temp=sum(Xtemp1.^2,2);
            temp2=repmat(temp,1,size(x{i},1)*size(x{i},3));
            temp3=temp1+temp2';
            Ktemp=Ktemp+repmat(temp3,1,N);
            Ktemp_2=weight*exp(Ktemp./(-2*sigma^2));
        end
        
        % i=d
        Xtemp=x{d};
        Xtemp1=reshape(permute(X{d},[1 3 2]),[size(X{d},1)*size(X{d},3),size(X{d},2)]);
        temp=Xtemp*Xtemp1';
        prodtemp=(-2)*temp;
        temp=sum(Xtemp.^2,2);
        temp1=repmat(temp,1,size(X{d},1)*size(X{d},3));
        temp=sum(Xtemp1.^2,2);
        temp2=repmat(temp,1,size(x{d},1));
        squaretemp=temp1+temp2';
        Ktemp_d=exp((prodtemp+squaretemp)./(-2*sigma^2));
        
        Ktemp_final=zeros(size(Ktemp_2,1),size(Ktemp_2,2));
        for j=1:N
            Ktemp_final(:,1+size(Ktemp_2,1)*(j-1):size(Ktemp_2,1)*j)=kron(Ktemp_d(:,1+size(Ktemp_d,1)*(j-1):size(Ktemp_d,1)*j),Ktemp_1(:,1+size(Ktemp_1,1)*(j-1):size(Ktemp_1,1)*j));
        end
        Ktemp_final=Ktemp_final.*Ktemp_2;
        a=size(X{2},1)*size(X{2},3);
        tranMat=zeros(N,a*N);
        for j=1:N
            tranMat(j,1+(j-1)*a:j*a)=ones(1,a);
        end
        Kfast=Kfast+ones(1,a)*Ktemp_final*tranMat';
        K(s,:)=Kfast;
        
    end
end