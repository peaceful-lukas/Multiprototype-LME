function U = learnU_lme_sp(DS, W, U, param)

    n = 0;

    tic;
    while(n < param.maxIterU)
        n = n + 1;

        cTriplets = generateClassifierTriplets(DS, W, U, param);

        dU = getGradient(DS, W, U, cTriplets, param);
        U = update(U, dU, param.lr_U/(1 + n * param.lr_U));
        % U = update(U, dU, param.lr_U);

        if mod(n, 100) == 0
            fprintf('U) iter %d / ', n);
            loss = getSampleLoss(DS, W, U, cTriplets, param);
            tic;
        end
    end


end



function U = update(U, dU, learning_rate)
    U = U - learning_rate * dU;
end



function dU = getGradient(DS, W, U, cTriplets, param)

    X = DS.D;
    bal_c = param.bal_c;
    lambda = param.lambda;
    lowDim = param.lowDim;
    numClasses = param.numClasses;

    
    % gradient for the first term in the objective differential.
    dU = zeros(lowDim, numClasses);
    for n=1:size(cTriplets, 1)
        i = cTriplets(n, 1);
        y_i = cTriplets(n, 2);
        alpha = cTriplets(n, 3);
        x_i = X(:, i);

        bin_y_i = zeros(numClasses, 1);
        bin_alpha = zeros(numClasses, 1);
        bin_y_i(y_i) = 1;
        bin_alpha(alpha) = 1;

        dU = dU + W*x_i*(bin_alpha - bin_y_i)';
    end

    if (size(cTriplets, 1) > 0)
        dU = bal_c*dU/size(cTriplets, 1) + lambda*U/size(U, 2);
    else
        dU = lambda*U/size(U, 2);
    end
end



function loss = getSampleLoss(DS, W, U, cTriplets, param)

    loss = 0;
    
    X = DS.D;
    c_LM = param.c_LM; % large margin for classification
    sp_LM = param.sp_LM; % large margin for structure preserving
    numClasses = param.numClasses;
    bal_c = param.bal_c;
    bal_sp = param.bal_sp;
    lambda = param.lambda; % regularizers' coefficient.
    cTriplets = generateClassifierTriplets(DS, W, U, param);

    % sum of the classification errors
    cErr = 0;
    for n=1:size(cTriplets, 1)
        i = cTriplets(n, 1);
        y_i = cTriplets(n, 2);
        alpha = cTriplets(n, 3);
        x_i = X(:, i);
        
        val = c_LM + x_i'*W'*( U(:, alpha) - U(:, y_i) );
        cErr = cErr + val;
    end
    if size(cTriplets, 1) > 0
        cErr = cErr / size(cTriplets, 1);
    end


    loss = bal_c*cErr + lambda*0.5*(norm(W, 'fro')^2/size(W, 2) + norm(U, 'fro')^2)/size(U, 2);
    fprintf('cViol: %d / loss: %f / cErr: %f / nomrW: %f / normU: %f / elapsed time: %f\n', size(cTriplets, 1), loss, cErr, norm(W, 'fro'), norm(U, 'fro'), toc);
end


% function cTriplets = generateAllClassifierTriplets(DS, W, U, M, param)
%     D_labels = DS.DL;
%     numInstances = numel(D_labels);
%     numClasses = param.numClasses;

%     cTriplets = [];
%     for i=1:numInstances
%         cTriplets = [cTriplets; repmat(i, numClasses, 1) repmat(D_labels(i), numClasses, 1) (1:numClasses)'];
%     end

%     cTriplets(find(cTriplets(:, 2) == cTriplets(:, 3)), :) = [];
%     % cTriplets = getValidClassifierTriplets(DS, W, U, M, cTriplets, param);
% end


function cTriplets = generateClassifierTriplets(DS, W, U, param)

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
    cTriplets = getValidClassifierTriplets(DS, W, U, cTriplets, param);
end


function cTriplets = getValidClassifierTriplets(DS, W, U, cTriplets, param)

    c_LM = param.c_LM; % large margin
    X = DS.D;

    valids = [];

    % choose valid triplets (which violate large margin conditions)
    for n=1:size(cTriplets, 1)
        i = cTriplets(n, 1);
        y_i = cTriplets(n, 2);
        alpha = cTriplets(n, 3);
        x_i = X(:, i);
        
        val = c_LM + x_i'*W'*( U(:, alpha) - U(:, y_i) );
        if val > 0
            valids = [valids n];
        end
    end
    
    cTriplets = cTriplets(valids, :);
end


