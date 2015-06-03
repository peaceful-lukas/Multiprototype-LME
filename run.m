
param.numClasses = 20;
param.lowDim = 100;
param.featureDim = 4096;
param.maxIterW = 1000;
param.maxIterU = 1000;
param.maxAlter = 5;
param.lr_W = 0.01; % learning rate for W
param.lr_U = 0.0001; % learning rate for U
param.c_LM = 100; % large margin for classification
param.sp_LM = 10; % large margin for structure preserving
param.lambda = 1; % regularizer coefficient
param.bal_c = 1; % balance term for classification loss
param.bal_sp = 1; % balance term for structure preserving loss
param.miniSize = 300; % mini-batch size

% param.epsilon = []; % threshold for adjacency matricies in original space.
% param.epsilon(1) = 2500;
% param.epsilon(2) = 2000;
% param.epsilon(3) = 2000;
% param.epsilon(4) = 2000;
% param.epsilon(5) = 1500;
% param.epsilon(6) = 1500;
% param.epsilon(7) = 1500;
% param.epsilon(8) = 1700;
% param.epsilon(9) = 2000;




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
lowDim = param.lowDim;
featureDim = param.featureDim;

W = rand(lowDim, featureDim);
W = normc(W);

U = rand(lowDim, featureDim);
U = normc(U);
clear lowDim featureDim


% alternate learning for W and U.
n = 0;
while( n < param.maxAlter )
    n = n + 1;
    fprintf('\n\n============================= Iteration %d =============================\n', n);

    % param.lr_W = param.lr_W * exp(-n);
    % param.lr_U = param.lr_U * exp(-n);
    W = learnW_new(DS, W, U, M, A, param);
    U = learnU_new(DS, W, U, M, A, param);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Draw Embeddings %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

E_whole = [];
E = {};
for alpha=1:numClasses
    E{alpha} = U*M{alpha};
    E_whole = [E_whole E{alpha}];
end
embeddings = E_whole;
clear alpha;


figure;
hold on;
box on; grid on; axis tight; daspect([1 1 1])
view(3); camproj perspective
camlight; lighting gouraud; alpha(0.75);
rotate3d on;

for n=1:numClasses
    range = (n-1)*numClasses+1:n*numClasses;
    scatter3(embeddings(1,range),embeddings(2,range), embeddings(3,range), 60, 'filled'); axis equal;
    
    drawnow;
    pause;
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Classification %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

E_whole = [];
E = {};
for alpha=1:numClasses
    E{alpha} = U*M{alpha};
    E_whole = [E_whole E{alpha}];
end

TX = DS.T;
T_labels = DS.TL;


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
    else
        fprintf('%d th query is categorized as %d (ground truth: %d)\n', n, max_class, T_labels(n));
    end
end

accuracy = double(accuracy) * 100 / size(T_labels, 1);
fprintf('accuracy : %f\n', accuracy);






