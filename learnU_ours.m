function U = learnU_ours(DS, W, U, M, A, param)

    n = 0;
    
    spTriplets = generateStructurePreservingTriplets(DS, W, U, M, A, param);
    % all_cTriplets = generateAllClassifierTriplets(DS, W, U, M, param);
    all_cTriplets = [];

    tic;
    while(n < param.maxIterU)
        n = n + 1;

        % tic
        cTriplets = generateClassifierTriplets(DS, W, U, M, param);
        % fprintf('generate classifier triplets : %f\n', toc);

        % tic
        dU = getGradient(DS, W, U, M, cTriplets, spTriplets, param);
        % fprintf('get gradients : %f\n', toc);

        % tic
        U = update(U, dU, param.lr_U/(1 + n * param.lr_U));
        % fprintf('update : %f\n', toc);
        % U = update(U, dU, param.lr_U);

        % tic
        spTriplets = generateStructurePreservingTriplets(DS, W, U, M, A, param);        
        % fprintf('generate structure preserving triplets: %f\n', toc);

        if mod(n, 100) == 0
            fprintf('U) iter %d / ', n);
            loss = getSampleLoss(DS, W, U, M, all_cTriplets, spTriplets, param);
            tic;
        end
    end


end



function U = update(U, dU, learning_rate)
    U = U - learning_rate * dU;
end


function dU = getGradient(DS, W, U, M, cTriplets, spTriplets, param)

    X = DS.D;
    numClasses = param.numClasses;
    bal_c = param.bal_c;
    bal_sp = param.bal_sp;
    lambda = param.lambda;
    lowDim = param.lowDim;
    featureDim = param.featureDim;

    
    % gradient for the first term in the objective differential.
    dU_first = zeros(lowDim, featureDim);
    for n=1:size(cTriplets, 1)
        i = cTriplets(n, 1);
        y_i = cTriplets(n, 2);
        alpha = cTriplets(n, 3);
        x_i = X(:, i);

        % k = floor(size(M{alpha}, 2)*rand) + 1;
        % k_prime = floor(size(M{y_i}, 2)*rand) + 1;
        % k = 1;
        % k_prime = 1;
        k = floor(size(M{alpha}, 2)/2);
        k_prime = floor(size(M{y_i}, 2)/2);

        dU_first = dU_first + W*x_i*(M{alpha}(:, k) - M{y_i}(:, k_prime))';
    end
    

    % % gradient for the second term in the objective differential.
    dU_second = zeros(lowDim, featureDim);
    spViolCount = 0;
    for alpha=1:numClasses
        for n=1:size(spTriplets{alpha}, 1)
            k = spTriplets{alpha}(n, 1);
            k_prime = spTriplets{alpha}(n, 2);
            l = spTriplets{alpha}(n, 3);

            dU_second = dU_second ...
                        + U*M{alpha}(:, k)*(M{alpha}(:, l) - M{alpha}(:, k_prime))' ...
                        + U*(M{alpha}(:, l) - M{alpha}(:, k_prime))*M{alpha}(:, k)';
        end
        spViolCount = spViolCount + size(spTriplets{alpha}, 1);
    end

    dU = zeros(lowDim, featureDim);
    if (size(cTriplets, 1) > 0)
        dU = dU + bal_c*dU_first/size(cTriplets, 1);
    end
    if (spViolCount > 0)
        dU = dU + bal_sp*dU_second/spViolCount;
    end
    dU = dU + lambda*U/size(U, 2);

end



function loss = getSampleLoss(DS, W, U, M, cTriplets, spTriplets, param)

    loss = 0;
    
    X = DS.D;
    c_LM = param.c_LM; % large margin for classification
    sp_LM = param.sp_LM; % large margin for structure preserving
    numClasses = param.numClasses;
    bal_c = param.bal_c;
    bal_sp = param.bal_sp;
    lambda = param.lambda; % regularizers' coefficient.
    cTriplets = generateClassifierTriplets(DS, W, U, M, param);
    % cTriplets = getValidClassifierTriplets(DS, W, U, M, cTriplets, param);

    % sum of the classification errors
    cErr = 0;
    for n=1:size(cTriplets, 1)
        i = cTriplets(n, 1);
        y_i = cTriplets(n, 2);
        alpha = cTriplets(n, 3);
        x_i = X(:, i);
        
        % k = floor(size(M{alpha}, 2)*rand) + 1;
        % k_prime = floor(size(M{y_i}, 2)*rand) + 1;
        % k = 1;
        % k_prime = 1;
        k = floor(size(M{alpha}, 2)/2);
        k_prime = floor(size(M{y_i}, 2)/2);
        
        val = c_LM + x_i'*W'*U*( M{alpha}(:, k) - M{y_i}(:, k_prime) );
        cErr = cErr + val;
    end
    if size(cTriplets, 1) > 0
        cErr = cErr / size(cTriplets, 1);
    end

    % sum of the structure preserving errors
    spErr = 0;
    spViolCount = 0;
    for alpha=1:numClasses
        for n=1:size(spTriplets{alpha}, 1)
            k = spTriplets{alpha}(n, 1);
            k_prime = spTriplets{alpha}(n, 2);
            l = spTriplets{alpha}(n, 3);

            val = sp_LM + M{alpha}(:, k)'*U'*U*M{alpha}(:, l) - M{alpha}(:, k)'*U'*U*M{alpha}(:, k_prime);

            spErr = spErr + val;
        end
        spViolCount = spViolCount + size(spTriplets{alpha}, 1);
    end
    if spViolCount > 0
        spErr = spErr / spViolCount;
    end

    loss = bal_c*cErr + bal_sp*spErr + lambda*0.5*(norm(W, 'fro')^2/size(W, 2) + norm(U, 'fro')^2)/size(U, 2);
    fprintf('cViol: %d / spViol: %d / loss: %f / cErr: %f / spErr: %f / nomrW: %f / normU: %f / elapsed time: %f\n', size(cTriplets, 1), spViolCount, loss, cErr, spErr, norm(W, 'fro'), norm(U, 'fro'), toc);
end


function cTriplets = generateAllClassifierTriplets(DS, W, U, M, param)
    D_labels = DS.DL;
    numInstances = numel(D_labels);
    numClasses = param.numClasses;

    cTriplets = [];
    for i=1:numInstances
        cTriplets = [cTriplets; repmat(i, numClasses, 1) repmat(D_labels(i), numClasses, 1) (1:numClasses)'];
    end

    cTriplets(find(cTriplets(:, 2) == cTriplets(:, 3)), :) = [];
    % cTriplets = getValidClassifierTriplets(DS, W, U, M, cTriplets, param);
end


function cTriplets = generateClassifierTriplets(DS, W, U, M, param)

% cTriplets = generateClassifierTriplets(DS, W, U, M, param)
%-   generate triplets for classification (first term in the objective function.)

    D_labels = DS.DL;
    numInstances = numel(D_labels);
    numClasses = param.numClasses;
    miniSize = param.miniSize;

    batchSel = floor(numInstances * rand(miniSize, 1)) + 1;
    corrs = D_labels(batchSel);
    incorrs = floor(numClasses*rand(miniSize, 1)) + 1;
    collapsed = find(incorrs == corrs);
    incorrs(collapsed) = mod(incorrs(collapsed) + 1, numClasses+1);
    incorrs(find(incorrs == 0)) = 1;
    
    cTriplets = [batchSel corrs incorrs];
    cTriplets = getValidClassifierTriplets(DS, W, U, M, cTriplets, param);
end


function cTriplets = getValidClassifierTriplets(DS, W, U, M, cTriplets, param)

    c_LM = param.c_LM; % large margin
    X = DS.D;

    valids = [];

    % choose valid triplets (which violate large margin conditions)
    for n=1:size(cTriplets, 1)
        i = cTriplets(n, 1);
        y_i = cTriplets(n, 2);
        alpha = cTriplets(n, 3);
        x_i = X(:, i);
        

        %% THIS k, k_prime values affect both GET_GRADIENT and GET_SAMPLE_LOSS functions.
        % k = floor(size(M{alpha}, 2)*rand) + 1;
        % k_prime = floor(size(M{y_i}, 2)*rand) + 1;
        % k = 1;
        % k_prime = 1;
        k = floor(size(M{alpha}, 2)/2);
        k_prime = floor(size(M{y_i}, 2)/2);
        
        val = c_LM + x_i'*W'*U*( M{alpha}(:, k) - M{y_i}(:, k_prime) );
        if val > 0
            valids = [valids n];
        end
    end
    
    cTriplets = cTriplets(valids, :);
end
