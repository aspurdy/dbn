load mnist_uint8;

train_x = gpuArray(double(train_x) / 255);
test_x = gpuArray(double(test_x)  / 255);
train_y = gpuArray(double(train_y));
test_y = gpuArray(double(test_y));

rand('state',0)
%train dbn
sizes = [100 100 500];
opts.numepochs =   10;
opts.batchsize =   100;
opts.momentum  =   0;
opts.alpha     =   1;
opts.decay     =   0.00001;
dbn = DBN(train_x, train_y, sizes, opts);
train(dbn, train_x, train_y);

%% compute most probable class label given test data
error = dbn.predict(test_x, test_y);

fprintf('Classification error is %3.2f%%\n',error*100)

figure('Color','black');
gibbSteps = [1, 10, 100, 1000, 10000];
for i = 1:10
    for j = 1:length(gibbSteps)
        subplot(length(gibbSteps),10,(j-1)*10+i), imshow(dbn.generate(i, 10, gibbSteps(j)));
    end
end

figure('Color','black');
for i = 1:10
    for j = 1:10
        subplot(10,10,(j-1)*10+i), imshow(reshape(-dbn.rbm(1).W((j-1)*10+i,:), 28, 28));
    end
end

