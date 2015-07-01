function W = learnW_ours(DS, W, U, M, A, param)

    n = 0;

    spTriplets = generateStructurePreservingTriplets(DS, W, U, M, A, param);    
    % all_cTriplets = generateAllClassifierTriplets(DS, W, U, M, param);
    all_cTriplets = [];

    tic;
    while(n < param.maxIterW)
        n = n + 1;
        
        cTriplets = generateClassifierTriplets(DS, W, U, M, param);

        dW = getGradient(DS, W, U, M, cTriplets, param);
        W = update(W, dW, param.lr_W/(1 + n * param.lr_W));
        % W = update(W, dW, param.lr_W);
        
        if mod(n, 100) == 0
            fprintf('W) iter %d / ', n);
            loss = getSampleLoss(DS, W, U, M, all_cTriplets, spTriplets, param);
            tic;
        end
    end


end



function W = update(W, dW, learning_rate)
    W = W - learning_rate * dW;
end



function dW = getGradient(DS, W, U, M, cTriplets, param)
    % getGradient() assumes that all the triplets are valid. 

    X = DS.D;
    bal_c = param.bal_c;
    lambda = param.lambda;
    lowDim = param.lowDim;
    featureDim = param.featureDim;

    dW = zeros(lowDim, featureDim);
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

        dW = dW + U*(M{alpha}(:, k) - M{y_i}(:, k_prime))*x_i';
    end

    if( size(cTriplets, 1) > 0 )
        dW = bal_c*dW/size(cTriplets, 1) + lambda*W/size(W, 2);
    else
        dW = lambda*W/size(W, 2);
    end
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


