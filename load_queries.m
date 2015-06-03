cd ~/Desktop/code/caffe_features/test

T = [];
data = dir('*.mat');

% load labels.
load(strcat('~/Desktop/code/caffe_features/test/',data(14977).name));
data(14977) = [];
T_labels = labels(1:1000);

% % load features.
tic
for n=1:1000
    load(data(n).name);
    T = [T feat'];
end
toc

clear data n labels feat;


cd ../../sandbox