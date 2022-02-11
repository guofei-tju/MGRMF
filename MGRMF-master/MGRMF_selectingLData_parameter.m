
clear
seed = 12345678;
rand('seed', seed);
nfolds = 10; nruns=1;

%% load dataset
dataname = 'LRSSL_dataset'
load(['data/',dataname,'.mat'])

K1 = [];
for i = 1:size(K1_list,3)
    K1(:,:,i) = K1_list(:,:,i);
end

K2 = [];
for j = 1:size(K2_list,3)
    K2(:,:,j) = K2_list(:,:,j);
end
%% load real world dataset
% dataname = 'Coutry2org'
% load(['data/otherdata/',dataname,'.mat'])
% i = 0;j = 0; K1 = []; K2 = [];
% y = net;

%%
globa_true=[];
globa_predict=[];
results = [];
range = [2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5];
p_nearest_neighbors_range = [5,10,15,20,25,30,35,40,45,50];
ks = 300;Iteration_max = 10; gamma_gips = 0.5; lamda_L1s = range;lamda_L2s = range; lamda_1s =range; lamda_2s = range; p_nearest_neighbors = 15; gamma = 5;

% for run=1:nruns
    for k = ks
        for p_nearest_neighbor = p_nearest_neighbors
        for gamma_gip = gamma_gips
            for lamda_L1 = lamda_L1s
            for lamda_L2 = lamda_L2s
                for lamda_1 = lamda_1s
                for lamda_2 = lamda_2s
                    nruns_mean_aupr = [];nruns_mean_auc = [];nruns_mean_running_time = [];
                    for run=1:nruns 
                    %% Kfold cross validation(10-fold)
                        crossval_idx = crossvalind('Kfold',y(:),nfolds);
                        fold_aupr=[];fold_auc=[];fold_loss = [];
                        fold_running_time=[];
                        fold_weights_1={};fold_weights_2 = {};
                        for fold=1:nfolds
                            t1 = clock;
                            train_idx = find(crossval_idx~=fold);
                            test_idx  = find(crossval_idx==fold);

                            y_train = y;
                            y_train(test_idx) = 0;
                            K1(:,:,i+1) = Knormalized(getGipKernel(y_train,gamma_gip));
                            K2(:,:,j+1) = Knormalized(getGipKernel(y_train',gamma_gip));
                            %% For realWorld dataset
    %                         K1(:,:,i+2) = Knormalized(getLinearKernel(y_train));
    %                         K2(:,:,j+2) = Knormalized(getLinearKernel(y_train'));
    %                         K1(:,:,i+3) = Knormalized(getPolyKernel(y_train,2));
    %                         K2(:,:,j+3) = Knormalized(getPolyKernel(y_train',2));
    %                         K1(:,:,i+4) = Knormalized(getPolyKernel(y_train,3));
    %                         K2(:,:,j+4) = Knormalized(getPolyKernel(y_train',3));
                    %             [KD, KL] = GaussianKernel(y_train', 1, 1);
                    %             K1(:,:,4)=kernel_corr(y_train,1,0,1);
                    %             K2(:,:,4)=kernel_corr(y_train,2,0,1);
                            [F,weights_1,weights_2,loss] = grmf_mv(K1(:,:,:),K2(:,:,:),y_train,k,Iteration_max,...
                                                              lamda_L1,lamda_L2,lamda_1,lamda_2,p_nearest_neighbor,...
                                                              gamma);
                            t2=clock;
                            fold_running_time = [fold_running_time;etime(t2,t1)];
                            fold_loss = [fold_loss,loss];
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                          %% 4. evaluate predictions
                            yy=y;
                            test_labels = yy(test_idx);
                            predict_scores = F(test_idx);
                            [X,Y,tpr,aupr_LGC_A_KA] = perfcurve(test_labels,predict_scores,1, 'xCrit', 'reca', 'yCrit', 'prec');

                            [X,Y,THRE,AUC_LGC_KA,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_labels,predict_scores,1);

                            fprintf('---------------\nRUN %d - FOLD %d  \n', run, fold)

                            fprintf('%d - FOLD %d - AUPR: %f - AUC: %f\n', run, fold, aupr_LGC_A_KA,AUC_LGC_KA )

                            fold_weights_1 = [fold_weights_1,{weights_1}];
                            fold_weights_2 = [fold_weights_2,{weights_2}];

                            fold_aupr=[fold_aupr;aupr_LGC_A_KA];
                            fold_auc=[fold_auc;AUC_LGC_KA];

                            globa_true=[globa_true;test_labels];
                            globa_predict=[globa_predict;predict_scores];
                        end
                        all_predict_results = [globa_true,globa_predict];

                        mean_aupr = mean(fold_aupr)
                        mean_auc = mean(fold_auc)
                        mean_running_time = mean(fold_running_time)

                        nruns_mean_aupr = [nruns_mean_aupr;mean_aupr];
                        nruns_mean_auc = [nruns_mean_auc;mean_auc];
                        nruns_mean_running_time = [nruns_mean_running_time;mean_running_time];
                        
                    end
                    mean_nruns_mean_auc = mean(nruns_mean_auc);
                    mean_nruns_mean_aupr = mean(nruns_mean_aupr);
                    mean_nruns_mean_running_time = mean(nruns_mean_running_time);
                   results = cat(1,results,...
                            [run,Iteration_max,k,gamma_gip,lamda_L1,lamda_L2,lamda_1,lamda_2,p_nearest_neighbor,gamma,mean_nruns_mean_aupr,mean_nruns_mean_auc,mean_nruns_mean_running_time]);
                        save_results([dataname,'_results.txt'],results);  
                end
                end
                
            end    
            end
        end
        end
    end
% end 


    
