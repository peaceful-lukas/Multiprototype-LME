% D_labels = DS.DL;
% [~, sorted_labels_idx] = sort(D_labels, 'ascend');
% Data = DS.D';
% Data = Data(sorted_labels_idx, :);
% [coeff, score, latent] = pca(Data);

% low_Data = score(:, 1:3);
% scatter3(low_Data(:, 1), low_Data(:, 2), low_Data(:, 3), 60, 'filled')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       K-MEANS Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


M_whole = [];
for n=1:numClasses
    M_whole = [M_whole; M{n}'];
end

[coeff, score, latent] = pca(M_whole);
low_Data = score(:, 1:3);


figure;
hold on;
box on; grid on; axis tight; daspect([1 1 1])
view(3); camproj perspective
camlight; lighting gouraud; alpha(0.75);
rotate3d on

rangeStartIdx = 1;
rangeEndIdx = 20;
for n=1:numClasses
    range = rangeStartIdx:rangeEndIdx;
    scatter3(low_Data(range, 1), low_Data(range, 2), low_Data(range, 3), 60, 'filled')
    
    drawnow;
    pause;

    rangeStartIdx = rangeStartIdx + 20;
    rangeEndIdx = rangeStartIdx + 20;
    % fprintf('n: %d / range start idx : %d\n', n, rangeStartIdx);
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       All Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




D_labels = DS.DL;
[~, sorted_labels_idx] = sort(D_labels, 'ascend');
Data = DS.D';
Data = Data(sorted_labels_idx, :);
[coeff, score, latent] = pca(Data);

low_Data = score(:, 1:3);


class_count = [];
for n=1:numClasses
    class_count(n) = size(find(D_labels == n), 1);
end






figure;
hold on;
box on; grid on; axis tight; daspect([1 1 1])
view(3); camproj perspective
camlight; lighting gouraud; alpha(0.75);
rotate3d on

rangeStartIdx = 1;
rangeEndIdx = class_count(1);
for n=1:numClasses
    range = rangeStartIdx:rangeEndIdx;
    scatter3(low_Data(range, 1), low_Data(range, 2), low_Data(range, 3), 60, 'filled')
    
    drawnow;
    pause;

    rangeStartIdx = rangeStartIdx + class_count(n);
    rangeEndIdx = rangeStartIdx + class_count(n+1);
    % fprintf('n: %d / range start idx : %d\n', n, rangeStartIdx);
end
