classdef SoftmaxRBM < BernoulliRBM & handle
    %RBM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        U
        vU
        
        d
        vd        
    end
    
    methods
        function rbm = SoftmaxRBM(nVis, nHidden, opts, nClasses)
            rbm@BernoulliRBM(nVis, nHidden, opts);

            % bias for visible units representing class labels
            rbm.d = zeros(nClasses, 1, 'gpuArray');
            rbm.vd = zeros(nClasses, 1, 'gpuArray');
            % weights between target labels and hidden units
            rbm.U = zeros(nHidden, nClasses, 'gpuArray');
            rbm.vU = zeros(nHidden, nClasses, 'gpuArray');
        end
        
        function rbm = train(rbm, x, y)
            assert(isfloat(x), 'x must be a float');
            assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
            m = size(x, 1);
                      
            batchsize = rbm.opts.batchsize;
            numepochs = rbm.opts.numepochs;
            alpha = rbm.opts.alpha;
            momentum = rbm.opts.momentum;
          
            numbatches = m / batchsize;
            assert(rem(numbatches, 1) == 0, 'numbatches not integer');
            err = zeros(numepochs, 1, 'gpuArray');
            for i = 1 : numepochs
                kk = randperm(m);
                for l = 1 : numbatches
                    batch =  x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
                    labels = y(kk((l - 1) * batchsize + 1 : l * batchsize), :);

                    % positive phase
                    x0 = batch;
                    y0 = labels;

                    h0_hat = logistic(repmat(rbm.c', batchsize, 1) + x0 * rbm.W' + y0 * rbm.U');

                    % negative phase
                    h0 = RBM.sample(repmat(rbm.c', batchsize, 1) + x0 * rbm.W' + y0 * rbm.U');
                    y1 = softmax(repmat(rbm.d', batchsize, 1) + h0 * rbm.U);

                    x1 = RBM.sample(repmat(rbm.b', batchsize, 1) + h0 * rbm.W);
                    h1_hat = logistic(repmat(rbm.c', batchsize, 1) + x1 * rbm.W' + y1 * rbm.U');

                    c1 = h0_hat' * x0;
                    c2 = h1_hat' * x1;

                    d1 = h0_hat' * y0;
                    d2 = h1_hat' * y1;

                    % compute weight updates
                    rbm.vW = momentum * rbm.vW + alpha * (c1 - c2) / batchsize;
                    rbm.vU = momentum * rbm.vU + alpha * (d1 - d2) / batchsize;
                    rbm.vb = momentum * rbm.vb + alpha * sum(x0 - x1)' / batchsize;
                    rbm.vc = momentum * rbm.vc + alpha * sum(h0 - h1_hat)' / batchsize;
                    rbm.vd = momentum * rbm.vd + alpha * sum(y0 - y1)' / batchsize;

                    % update weights
                    rbm.W = rbm.W + rbm.vW;
                    rbm.U = rbm.U + rbm.vU;
                    rbm.b = rbm.b + rbm.vb;
                    rbm.c = rbm.c + rbm.vc;
                    rbm.d = rbm.d + rbm.vd;

                    % reconstruction error
                    err(i) = err(i) + sum(sum((x0 - x1) .^ 2)) / batchsize;
                end

                disp(['epoch ' num2str(i) '/' num2str(numepochs)  '. Average reconstruction error is: ' num2str(err(i) / numbatches)]);

            end
%             plot(err)
        end


    end
    
end

