
param.algorithm = 'lme_mp';
param.numClasses = 20;
param.lowDim = 100;
param.featureDim = 4096;
param.maxIterW = 10;
param.maxIterU = 10;
param.maxAlter = 5;
param.lr_W = 0.1; % learning rate for W
param.lr_U = 0.001; % learning rate for U
param.c_LM = 0.1; % large margin for classification
param.sp_LM = 0.01; % large margin for structure preserving
param.lambda = 1000; % regularizer coefficient
param.bal_c = 5; % balance term for classification loss
param.bal_sp = 1; % balance term for structure preserving loss
param.softmax_c = 100; % softmax parameter.
param.miniSize = 300; % mini-batch size





cd ~/Desktop/code/sandbox

% load_features;
% load_queries;
% load_images;
load('Dataset.mat');

D = DS.D;
D_labels = DS.DL;


numClasses = param.numClasses;


% Clustering
%% clustering per classes by CRP (but now, K-Means)
load clustered.mat
tic
M = {};
for n=1:numClasses
    labels_per_class = find(D_labels == n);
    D_per_class = D(:, labels_per_class);
    
    [assignments centroids] = kmeans(D_per_class', 10, 'Display', 'iter');
    M{n} = centroids';
end
toc
clear assignments centroids labels_per_class D_per_class n;



% Adjacency matrices of k-NN graphs for each category.
tic
% knn-based graph
A = {};
for n=1:numClasses
    A{n} = knn_graph(M{n}, 2);
    val = is_connected_graph(A{n});
    % fprintf('%d-th graph is connected? %d\n', n, val);
    % imagesc(A{n});
    % pause
end
toc


% epsilon-based nearest neighbors graph.
% A = {};
% for n=1:numClasses
    
%     clusters = M{n};

%     numClusters = size(clusters, 2);
%     A{n} = zeros(numClusters, numClusters);
    
%     for m = 1:numClusters
%         %%%%%%%%%%%%%%%%%%%% NEED !!!  different measure for Caffe features.
%         Diff = clusters - repmat(clusters(:, m), 1, numClusters);
%         Diff = sum(Diff.^2, 1);

%         A{n}(m, find(Diff < param.epsilon(n))) = 1;
%     end
%     A{n} = A{n} - eye(numClusters, numClusters);
%     % imagesc(A{1, 1});
% end
% toc
% clear clusters numClusters Diff n m;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Optimizaiton %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% randomly initialize W and U
if strcmp(param.algorithm, 'ours') == 1 | strcmp(param.algorithm, 'lme_mp') == 1
    lowDim = param.lowDim;
    featureDim = param.featureDim;
    W = randn(lowDim, featureDim);
    U = randn(lowDim, featureDim);
    W = W/norm(W, 'fro');
    U = U/norm(U, 'fro');
    clear lowDim featureDim
elseif strcmp(param.algorithm, 'lme_sp') == 1
    lowDim = param.lowDim;
    featureDim = param.featureDim;
    W = randn(lowDim, featureDim);
    U = randn(lowDim, numClasses);
    clear lowDim featureDim
end

% alternate learning for W and U.
fprintf('\n\n>>> ALGORITHM : %s <<< \n\n', param.algorithm);
n = 0;
while( n < param.maxAlter )
    n = n + 1;
    fprintf('\n============================= Iteration %d =============================\n', n);

    if n <= 5
        param.lr_W = param.lr_W * 0.5;
        param.lr_U = param.lr_U * 0.5;
    end

    if strcmp(param.algorithm, 'ours') == 1
        W = learnW_ours(DS, W, U, M, A, param);
        U = learnU_ours(DS, W, U, M, A, param);
    
    elseif strcmp(param.algorithm, 'lme_mp') == 1
        % W = learnW_lme_mp(DS, W, U, M, param);
        W = learnW_lme_mp_complete(DS, W, U, M, param);
        U = learnU_lme_mp(DS, W, U, M, param);
    
    elseif strcmp(param.algorithm, 'lme_sp') == 1
        W = learnW_lme_sp(DS, W, U, param);
        U = learnU_lme_sp(DS, W, U, param);
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Draw Embeddings %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% As it is
E_whole = [];
E = {};
for alpha=1:numClasses
    E{alpha} = U*M{alpha};
    E_whole = [E_whole E{alpha}];
end
embeddings = E_whole;
clear alpha;


% SVD
embeddings = [];
for alpha=1:numClasses
    UM = U*M{alpha};
    [UU SS VV] = svd(UM);
    
    SS(:, 4:end) = [];
    VV(:, 4:end) = [];

    % SS(4:end, 4:end) = 0;
    UM = UU*SS*VV';
    % keyboard;
    embeddings = [embeddings UM(1:3, :)];
end
clear alpha;

% Draw figure
figure;
hold on;
box on; grid on; axis tight; daspect([1 1 1]);
view(3); camproj perspective
camlight; lighting gouraud; alpha(0.75);
rotate3d on;

for n=1:numClasses
    range = (n-1)*10+1:n*10;
    scatter3(embeddings(1,range),embeddings(2,range), embeddings(3,range), 60, 'filled'); axis equal;
    
    drawnow;
    pause;
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Classification %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TX = DS.T;
T_labels = DS.TL;


hits = {};
misses = {};
accuracies = {};
for alpha=1:numClasses
    hits{alpha} = [];
    misses{alpha} = [];
    accuracies{alpha} = 0;
end


if strcmp(param.algorithm, 'lme_sp') == 1 %% LME SINGLE-PROTOTYPE
    accuracy = 0;

    for n=1:size(TX, 2)
        
        q_i = TX(:, n);
        [~, max_class] = max(q_i'*W'*U);

        if max_class == T_labels(n)
            accuracy = accuracy + 1;
            accuracies{max_class} = accuracies{max_class} + 1;
            hits{max_class} = [hits{max_class} n];
        else
            % misses{max_class} = [misses{max_class} n];
            misses{T_labels(n)} = [misses{T_labels(n)} n];
        %     fprintf('%d th query is categorized as %d (ground truth: %d)\n', n, max_class, T_labels(n));
        end

        if mod(n, 1000) == 0
            fprintf('.');
        end
    end

    accuracy = double(accuracy) * 100 / size(T_labels, 1);
    fprintf('accuracy : %f\n', accuracy);


    category_size = {};
    for alpha=1:numClasses
        category_size{alpha} = numel(find(T_labels == alpha));
        accuracies{alpha} = double(accuracies{alpha}) * 100 / category_size{alpha};
    end
   

else %% OUR MODELS | LME MULTI-PROTOTYPE
    accuracy = 0;

    for n=1:size(TX, 2)
        
        max_sim = -Inf;
        max_class = -1;
        q_i = TX(:, n);

        for alpha=1:numClasses
            UM = U*M{alpha};
            sim = max(q_i'*W'*UM);

            if max_sim < sim
                max_sim = sim;
                max_class = alpha;
            end
        end

        if max_class == T_labels(n)
            accuracy = accuracy + 1;
            accuracies{max_class} = accuracies{max_class} + 1;
            hits{max_class} = [hits{max_class} n];
        else
            % misses{max_class} = [misses{max_class} n];
            misses{T_labels(n)} = [misses{T_labels(n)} n];
        %     fprintf('%d th query is categorized as %d (ground truth: %d)\n', n, max_class, T_labels(n));
        end

        if mod(n, 1000) == 0
            fprintf('.');
        end
    end

    accuracy = double(accuracy) * 100 / size(T_labels, 1);
    fprintf('accuracy : %f\n', accuracy);


    category_size = {};
    for alpha=1:numClasses
        category_size{alpha} = numel(find(T_labels == alpha));
        accuracies{alpha} = double(accuracies{alpha}) * 100 / category_size{alpha};
    end
end








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Show Results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

I = DS.TI;

alpha = 3; % category
target_imgs = misses{alpha};
fig = figure;
hold on;

set(fig, 'Position', [0, 0, 1500, 1200]);

numRows = 8;
numCols = 10;
for row=1:numRows
    for col=1:numCols
        pos_idx = (row-1)*numCols + col;
        img_idx = target_imgs(pos_idx);
        
        subplot(numRows, numCols, pos_idx);
        imagesc(I{1, img_idx});
        axis off;
        axis image;
    end
end
hold off;

clear numRow, numCols;
