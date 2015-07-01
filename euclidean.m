%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   EUCLIDEAN + K-NN CLASSIFICATION
% 
%   result: 81%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



cd ~/Desktop/code/sandbox

% load_features;
% load_queries;
% load_images;
load('Dataset.mat');

D = DS.D;
D_labels = DS.DL;
T = DS.T;
T_labels = DS.TL;
% T_labels = T_labels(1:200);

numClasses = param.numClasses;
numTrain = numel(D_labels);
numTest = numel(T_labels);


k = 7;
accuracy = 0;
withdraw = 0;

for n=1:numTest

    dist = sum( (D - repmat(T(:, n), 1, numTrain)).^2, 1 );
    [~, dist_idx] = sort(dist, 'ascend');

    knns = dist_idx(1:k);
    knn_labels = D_labels(knns);
    
    uniq_knn_labels = unique(knn_labels);
    knn_label_counts = histc(knn_labels, uniq_knn_labels);
    
    max_counts = max(knn_label_counts);
    if numel(find(knn_label_counts == max_counts)) > 1
        withdraw = withdraw + 1;
    else
        predicted = uniq_knn_labels(find(knn_label_counts == max_counts));
        if T_labels(n) == predicted
            accuracy = accuracy + 1;
        end
    end

    if mod(n, 100) == 0
        fprintf('.');
    end

end

fprintf('\n');
accuracy = double(accuracy)*100 / (numTest - withdraw);
fprintf('accuracy : %f\n', accuracy);
fprintf('withdraw : %d\n', withdraw);

