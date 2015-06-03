cd ~/Desktop/code/caffe_features/train

D = [];
data = dir('*.mat');

% load labels.
load(strcat('~/Desktop/code/caffe_features/train/',data(15663).name));
D_labels = labels;
data(15663) = [];

% load features.
tic
for n=1:numel(data)
    load(data(n).name);
    D = [D feat'];
end
toc


cd ../../sandbox