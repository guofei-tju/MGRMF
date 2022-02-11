function [F,weights_1,weights_2,iter_Loss] = grmf_mv(W1,W2,Y,k,Iteration_max,lamda_L1,lamda_L2,lamda_1,lamda_2,p_nearest_neighbor,gamma)
%tju cs, bioinformatics. This program is coded by reference follow:
%ref:
%[1] Zheng X, Ding H, Mamitsuka H, et al. 
%	Collaborative matrix factorization with multiple similarities for predicting drug-target interactions[C]
%	ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2013:1025-1033.
%and
%[2] Shen Z, Zhang Y H, Han K, et al. 
%    miRNA-Disease Association Prediction with Collaborative Matrix Factorization[J]. Complexity, 2017, 2017(9):1-9.
%
%[3] Ezzat A, Zhao P, Wu M, et al. Drug-Target Interaction Prediction with Graph Regularized Matrix Factorization[J]. 
%    IEEE/ACM Transactions on Computational Biology & Bioinformatics, 2016, PP(99):1-1.
%
% Different from the above articles, our program did not use multiple similarities matrix (ref [1]) and WKNKN (ref [2])


%Graph Regularized Matrix Factorization (GRMF)
%This program is used to Collaborative filtering. 
% W1 : the kernel of object 1, (n-by-n-v)
% W2 : the kernel of object 2, (m-by-m-v)
% Y  : binary adjacency matrix, (n-by-m)
% k  : the k is the dimension of the feature spaces
% Iteration_max  : the Iteration_maxis the max numbers of Iteration
%lamda_1 ,%lamda_2 , %lamda_L : these are regularization coefficients of kernel W1, kernel W2, A, B.
%p_nearest_neighbor: the p nearest neighbor samples
fprintf('Graph Regularized Matrix Factorization\n'); 
F=[];
  
[m,n]=size(Y);
L1 = zeros(size(W1));
L2 = zeros(size(W2));
%initial value of A and B
fprintf('initial value of A and B\n'); 
[U1,S_k,V1] = svds(Y,k);
A = U1*(S_k^0.5);  %(m-by-k)
B = V1*(S_k^0.5);  %(n-by-k)

%objective function:
% min    ||Y - AB'||^2 + lamda_l*(||A||^2 + ||B||^2) + lamda_w1*tr(A'*Lw1*A) + lamda_w2*tr(B'*Lw2*B)
% 
% the ||x|| is F norm



%1.Sparsification of the similarity matrices
%1.1
fprintf('Sparsification of the similarity matrices 1\n');
for i=1:size(L1,3)
N_1 = preprocess_PNN(W1(:,:,i),p_nearest_neighbor);
S_1 = N_1.*W1(:,:,i);
% S_1 = W1(:,:,i);
d_1 = sum(S_1);
D_1 = diag(d_1);
L_D_1 = D_1 - S_1;
%Laplacian Regularized
d_tmep_1=pinv(D_1^(1/2));
L_D_11 = d_tmep_1*L_D_1*d_tmep_1;
L1(:,:,i) = L_D_11;
end
%1.2
fprintf('Sparsification of the similarity matrices 2\n');
for i=1:size(L2,3)
N_2 = preprocess_PNN(W2(:,:,i),p_nearest_neighbor);
S_2 = N_2.*W2(:,:,i);
% S_2 = W2(:,:,i);
d_2 = sum(S_2);
D_2 = diag(d_2);
L_D_2 = D_2 - S_2;
%Laplacian Regularized
d_tmep_2= pinv(D_2^(1/2));
L_D_22 = d_tmep_2*L_D_2*d_tmep_2;
L2(:,:,i) = L_D_22;
end
weights_1 = 1/size(L1,3)*ones(size(L1,3),1);
weights_2 = 1/size(L2,3)*ones(size(L2,3),1);
%sloving the problem by alternating least squares
fprintf('Sloving by Alternating least squares\n');

L_D_11_Com = combine_kernels(weights_1.^gamma, L1);
L_D_22_Com = combine_kernels(weights_2.^gamma, L2);
% Y = preprocess_WKNKN(Y,L_D_11_Com,L_D_22_Com,1,0.5);
iter_Loss = [];
for i=1:Iteration_max

    X_A = lamda_1*L_D_11_Com; X_B = B'*B + lamda_L1*eye(k); X_C = Y*B;
    A = sylvester(X_A,X_B,X_C);
    X_A = lamda_2*L_D_22_Com; X_B = A'*A + lamda_L2*eye(k); X_C = Y'*A;
    B = sylvester(X_A,X_B,X_C);
% 	A = (Y*B - lamda_1*L_D_11_Com*A)/(B'*B + lamda_L*eye(k));
% 	B = (Y'*A - lamda_2*L_D_22_Com*B)/(A'*A + lamda_L*eye(k));

	weights_1=computing_weights(A,L1,gamma,1);
	weights_2=computing_weights(B,L2,gamma,1);
    L_D_11_Com = combine_kernels(weights_1.^gamma, L1);
    L_D_22_Com = combine_kernels(weights_2.^gamma, L2);

    Loss = norm(Y-A*B','fro')^2 + lamda_L1* norm(A,'fro')^2 + lamda_L2*norm(B,'fro')^2+ ...
            lamda_1*trace(A'*L_D_11_Com*A)+lamda_2*trace(B'*L_D_22_Com*B);
%     fprintf('the loss in step %d',i);
%     fprintf('the loss is:%f \n',Loss);
    iter_Loss = [iter_Loss;Loss];
end

%reconstruct Y*
fprintf('Reconstruct Y*\n');
F = A*B';

end

function similarities_N = neighborhood_Com(similar_m,kk)

similarities_N=zeros(size(similar_m));

mm = size(similar_m,1);

for ii=1:mm
	
	for jj=ii:mm
		iu = similar_m(ii,:);
		iu_list = sort(iu,'descend');
		iu_nearest_list_end = iu_list(kk);
		
		ju = similar_m(:,jj);
		ju_list = sort(ju,'descend');
		ju_nearest_list_end = ju_list(kk);
		if similar_m(ii,jj)>=iu_nearest_list_end & similar_m(ii,jj)>=ju_nearest_list_end
			similarities_N(ii,jj) = 1;
			similarities_N(jj,ii) = 1;
		elseif similar_m(ii,jj)<iu_nearest_list_end & similar_m(ii,jj)<ju_nearest_list_end
			similarities_N(ii,jj) = 0;
			similarities_N(jj,ii) = 0;
		else
			similarities_N(ii,jj) = 0.5;
			similarities_N(jj,ii) = 0.5;
		end
	
	
	end


end

end


function weights=computing_weights(F,L_l,gamma,dim)

w = zeros(size(L_l,3),1);
weights = w;
e = 1/(gamma - 1);
	for i=1:length(w)
		if dim ==1
			d = F'*L_l(:,:,i)*F;
		else
			d = F*L_l(:,:,i)*F';
		end
		s = (1/trace(d))^e;
		w(i) = s;
	end
	for i=1:length(w)
		weights(i) = w(i)/(sum(w));
	end

end

function L = calDifLaplacian( W, type )
%CALDIFLAPLACIAN 此处显示有关此函数的摘要
% W：输入的邻接/相似矩阵
% type：计算拉普拉斯矩阵的方案or在某些算法中的实现形式
% L返回值：拉普拉斯矩阵

n = length(W);
D = zeros(n);
for i=1:n
    D(i,i)= sum(W(i, :));
end

if strcmp(type,'standard')
    L = D - W;
elseif strcmp(type, 'normalized')
    L = D - W;
    L = pinv(D)*L;     %D的（伪）逆*L——标准化
elseif strcmp(type, 'NJW')
    L = (D^-1/2) * (D - W) * (D^-1/2);
elseif strcmp(type, 'MS')
    L = (D^-1) * W;
end

end