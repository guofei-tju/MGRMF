

% [X,Y] = meshgrid(beta,gamma);
% matrix_ROC = zeros(length(lamda_1));
% for i = 1:length(beta)
%     for j = 1:length(gamma)
%         matrix_ROC(i,j) = rand();
%     end
% end
% contourf(X,Y,matrix_ROC,10000,'LineStyle','none');
function contour_fig(lamda_1,lamda_2,results)
    [X,Y] = meshgrid(lamda_1,lamda_2);
    matrix_ROC = zeros(length(lamda_1));
    for i = 1:length(lamda_2)
        for j = 1:length(lamda_1)
            matrix_ROC(i,j) = results(find(results(:,3)==X(i,j) & results(:,5)==Y(i,j)),8); 
        end
    end
    contourf(X,Y,matrix_ROC,10000,'LineStyle','none');
end