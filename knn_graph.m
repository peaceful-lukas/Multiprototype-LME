function A = knn_graph(X, k)

    numInstances = size(X, 2);
    A = zeros(numInstances, numInstances);

    % X = X - repmat(mean(X, 2), 1, numInstances);

    for n=1:numInstances
        x = X(:, n);
        
        % sim = x' * X;
        % [~, sim_idx] = sort(sim, 'descend');
        % sim_idx(find(sim_idx == n)) = []; % itself.
        % A(n, sim_idx(1:k)) = 1;

        dist = sum((X - repmat(x, 1, numInstances)).^2, 1);
        [~, dist_idx] = sort(dist, 'ascend');
        dist_idx(find(dist_idx == n)) = []; %itself.
        A(n, dist_idx(1:k)) = 1;
        A(dist_idx(1:k), n) = 1;
    end

    A;
end