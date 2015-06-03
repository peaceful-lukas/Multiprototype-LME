cd ~/Desktop/code/caffe_features/train

DI = {};
data = dir('*.jpg');

tic
for n=1:numel(data)
    img = imread(data(n).name);
    DI{n} = img;
end



cd ~/Desktop/code/caffe_features/test

TI = {};
data = dir('*.jpg');

for n=1:numel(data)
    img = imread(data(n).name);
    TI{n} = img;
end

toc

clear data ans labels feat img n;


cd ../../sandbox