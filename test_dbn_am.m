load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

rand('state',0)
%train dbn
dbn.sizes = [100 100 500];
opts.numepochs =   10;
opts.batchsize = 100;
opts.momentum  =   0.1;
opts.alpha     =   0.1;
dbn = dbnsetup(dbn, train_x, train_y, opts);
dbn = dbntrain(dbn, train_x, train_y, opts);

%% compute most probable class label given test data
error = dbnpred(dbn, test_x, test_y);

fprintf('Classification error is %3.2f%%\n',error*100)

figure('Color','black');
gibbSteps = [0, 10, 100, 500, 1000, 5000];
for i = 1:10
    for j = 1:length(gibbSteps)
        subplot(length(gibbSteps),10,(j-1)*10+i), imshow(dbnsample(dbn, i, 10, gibbSteps(j)));
    end
end