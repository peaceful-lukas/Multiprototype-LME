function spTriplets = generateStructurePreservingTriplets(DS, W, U, M, A, param)
% spTriplets = generateStructurePreservingTripliets(DS, W, U, M, param)
%-   generate triplets for structure preserving (second term in the objective function.)

    numClasses = param.numClasses;
    miniSize = param.miniSize;
    
    spTriplets = {};
    for alpha = 1:numClasses
        spTriplets{alpha} = [];
        numClusters = size(M{alpha}, 2);

        for vertex = 1:numClusters
            adjacency = A{alpha}(vertex, :);
            k_prime = find(adjacency == 1);
            l = find(adjacency == 0);
            l(find(l == vertex)) = [];

            for n = 1:numel(k_prime)
                repVertex = repmat(vertex, numel(l), 1);
                repKprime = repmat(k_prime(n), numel(l), 1);
                spTriplets{alpha} = [spTriplets{alpha}; repVertex, repKprime, l'];
            end
        end
    end

    spTriplets = getValidStructurePreservingTriplets(DS, W, U, M, spTriplets, param);
    
    % % minibatch sampling
    % miniSizePerClass = ceil(miniSize/numClasses);
    % for alpha = 1:numClasses
    %     spViolCount = size(spTriplets{alpha}, 1);
    %     sample_idx = ceil(spViolCount*rand(miniSizePerClass, 1));
    %     spTriplets{alpha} = spTriplets{alpha}(sample_idx, :);
    % end
end

function spTriplets = getValidStructurePreservingTriplets(DS, W, U, M, spTriplets, param)

    numClasses = param.numClasses;
    sp_LM = param.sp_LM; % large margin for structure preserving

    valids = {};
    for alpha=1:numClasses
        valids{alpha} = [];

        for n=1:size(spTriplets{alpha}, 1)
            k = spTriplets{alpha}(n, 1);
            k_prime = spTriplets{alpha}(n, 2);
            l = spTriplets{alpha}(n, 3);

            val = sp_LM + M{alpha}(:, k)'*U'*U*M{alpha}(:, l) - M{alpha}(:, k)'*U'*U*M{alpha}(:, k_prime);

            if(val > 0)
                valids{alpha} = [valids{alpha} n];
            end
        end

        spTriplets{alpha} = spTriplets{alpha}(valids{alpha}, :);
    end

end