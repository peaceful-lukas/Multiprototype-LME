function W = learnW_lme_mp_complete(DS, W, U, M, param)

    n = 0;

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
            loss = getSampleLoss(DS, W, U, M, all_cTriplets, param);
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
    softmax_c = param.softmax_c;
    bal_c = param.bal_c;
    lambda = param.lambda;
    lowDim = param.lowDim;
    featureDim = param.featureDim;

    tripletsIdx = 1:size(cTriplets, 1);
    

    % A = arrayfun(@(n) sum(exp(softmax_c*X(:, cTriplets(n, 1))'*W'*U*M{cTriplets(n, 3)})), tripletsIdx);
    
    % a = arrayfun(@(n) exp(softmax_c*X(:, cTriplets(n, 1))'*W'*U*M{cTriplets(n, 3)}), tripletsIdx, 'UniformOutput', false);
    % b = arrayfun(@(n) (softmax_c*X(:, cTriplets(n, 1))'*W'*U*M{cTriplets(n, 3)}), tripletsIdx, 'UniformOutput', false);
    % c = cellfun(@times, b, a, 'UniformOutput', false);
    % B_k = cellfun(@plus, a, c, 'UniformOutput', false); % (1 x 293)[1x10]    
    % B_grad_k = arrayfun(@(n) arrayfun(@(k) U*M{cTriplets(n, 3)}(:, k)*X(:, cTriplets(n, 1))', 1:size(M{cTriplets(n, 3)}, 2), 'UniformOutput', false), tripletsIdx, 'UniformOutput', false);


    %%%%%%%%%%%%%% NOT WORKING
    % n = 1
    % B = arrayfun(@(n) cellfun(@times, B_k(n), B_grad_k{n}, 'UniformOutput', false), tripletsIdx);
    %%%%%%%%%%%%%%%%%%%%%%%%%%

    dW = zeros(lowDim, featureDim);
    for n=1:size(cTriplets, 1)
        i = cTriplets(n, 1);
        y_i = cTriplets(n, 2);
        alpha = cTriplets(n, 3);
        x_i = X(:, i);

        dW = dW + softmaxGradient(W, U, x_i, M{alpha}, param) - softmaxGradient(W, U, x_i, M{y_i}, param);    
    end
    
    if (size(cTriplets, 1) > 0)
        dW = bal_c*dW/size(cTriplets, 1) + lambda*W/size(W, 2);
    else
        dW = lambda*W/size(W, 2);
    end
end

function sgW = softmaxGradient(W, U, x_i, M_alpha, param)

    c = param.softmax_c;
    d = size(x_i, 1);

    lowDim = param.lowDim;
    featureDim = param.featureDim;


    f = sum(exp(c*x_i'*W'*U*M_alpha));
    g = sum((x_i'*W'*U*M_alpha).*(exp(c*x_i'*W'*U*M_alpha)));
    % f_grad = zeros(lowDim, featureDim);
    % g_grad = zeros(lowDim, featureDim);

    X = c*x_i'*W'*U*M_alpha;
    f_temp = c*exp(X);
    f_grad_k = arrayfun(@(k)(f_temp(k)*U*M_alpha(:, k)*x_i'), 1:size(M_alpha, 2), 'UniformOutput', false);
    f_grad_k_cat = cat(3, f_grad_k{:});
    f_grad = sum(f_grad_k_cat, 3);

    g_temp = exp(X) + (X.*exp(X));
    g_grad_k = arrayfun(@(k)(g_temp(k)*U*M_alpha(:, k)*x_i'), 1:size(M_alpha, 2), 'UniformOutput', false);
    g_grad_k_cat = cat(3, g_grad_k{:});
    g_grad = sum(g_grad_k_cat, 3);

    sgW = (f*g_grad - f_grad*g)/f^2;
end


function loss = getSampleLoss(DS, W, U, M, cTriplets, param)

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
        
        val = c_LM + max(x_i'*W'*U*M{alpha}) - max(x_i'*W'*U*M{y_i});
        cErr = cErr + val;
    end
    if size(cTriplets, 1) > 0
        cErr = cErr / size(cTriplets, 1);
    end


    loss = bal_c*cErr + lambda*0.5*(norm(W, 'fro')^2/size(W, 2) + norm(U, 'fro')^2)/size(U, 2);
    fprintf('cViol: %d / loss: %f / cErr: %f / nomrW: %f / normU: %f / elapsed time: %f\n', size(cTriplets, 1), loss, cErr, norm(W, 'fro'), norm(U, 'fro'), toc);
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
        % keyboard;
        val = c_LM + max(x_i'*W'*U*M{alpha}) - max(x_i'*W'*U*M{y_i});
        if val > 0
            valids = [valids n];
        end
    end
    cTriplets = cTriplets(valids, :);
    
    fprintf('num valids : %d\n', numel(valids));
end




