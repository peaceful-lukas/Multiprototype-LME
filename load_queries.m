cd ~/Desktop/code/caffe_features/test

T = [];
data = dir('*.mat');

% load labels.
load(strcat('~/Desktop/code/caffe_features/test/',data(14977).name));
data(14977) = [];
T_labels = labels;

% % load features.
tic
for n=1:numel(data)
    load(data(n).name);
    T = [T feat'];
end
toc

clear data n labels feat;


cd ../../sandbox