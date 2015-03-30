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

            % bias for softmax units
            rbm.d = zeros(nClasses, 1, 'gpuArray');
            rbm.vd = zeros(nClasses, 1, 'gpuArray');
            % weights for softmax-to-hidden connections
            rbm.U = zeros(nHidden, nClasses, 'gpuArray');
            rbm.vU = zeros(nHidden, nClasses);
        end
       
        
        function rbm = train(rbm, x, y)
            assert(isfloat(x), 'x must be a float');
            assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
            m = size(x, 1);
                      
            batchsize = rbm.opts.batchsize;
            numepochs = rbm.opts.numepochs;
            alpha = rbm.opts.alpha;
            momentum = rbm.opts.momentum;
            decay = rbm.opts.decay;
            k = rbm.opts.k;
          
            numbatches = m / batchsize;
            assert(rem(numbatches, 1) == 0, 'numbatches not integer');
            x = gpuArray(x);
            for i = 1 : numepochs
                kk = randperm(m);
                for l = 1 : numbatches
                    batch =  x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
                    labels = y(kk((l - 1) * batchsize + 1 : l * batchsize), :);

                    % positive phase
                    v1 = gpuArray(batch);
                    y1 = gpuArray(labels);

                    h1 = RBM.sample(repmat(rbm.c', batchsize, 1) + v1 * rbm.W' + y1 * rbm.U');

                    if i == 1 && l == 1
                        h2 = h1;
                    end
                    
                    % negative phase
                    for j = 1:k
                        y2 = softmax(repmat(rbm.d', batchsize, 1) + h2 * rbm.U);
                        v2 = RBM.sample(repmat(rbm.b', batchsize, 1) + h2 * rbm.W);
                        h2 = RBM.sample(repmat(rbm.c', batchsize, 1) + v2 * rbm.W' + y2 * rbm.U');
                    end
                    
                    c1 = h1' * v1;
                    c2 = h2' * v2;

                    d1 = h1' * y1;
                    d2 = h2' * y2;

                    % compute weight updates
                    rbm.vW = momentum * rbm.vW + alpha * (c1 - c2 - decay * rbm.W) / batchsize;
                    rbm.vU = momentum * rbm.vU + alpha * (d1 - d2 - decay * rbm.U) / batchsize;
                    rbm.vb = momentum * rbm.vb + alpha * (sum(v1 - v2)' - decay * rbm.b)     / batchsize;
                    rbm.vc = momentum * rbm.vc + alpha * (sum(h1 - h2)' - decay * rbm.c) / batchsize;
                    rbm.vd = momentum * rbm.vd + alpha * (sum(y1 - y2)' - decay * rbm.d)     / batchsize;

                    % update weights
                    rbm.W = rbm.W + rbm.vW;
                    rbm.U = rbm.U + rbm.vU;
                    rbm.b = rbm.b + rbm.vb;
                    rbm.c = rbm.c + rbm.vc;
                    rbm.d = rbm.d + rbm.vd;

                end
                fprintf('epoch %d / %d\n', i, numepochs);
            end
        end
    end
end

