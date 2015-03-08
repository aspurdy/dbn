classdef BernoulliRBM < RBM & handle
    %RBM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        W
        vW

        b
        vb

        c
        vc
      
        opts
        
    end
    
    methods

        function rbm = BernoulliRBM(nVis, nHidden, opts)
            if  nargin > 0
                rbm.W  = zeros(nHidden, nVis, 'gpuArray');
                rbm.vW = zeros(nHidden, nVis, 'gpuArray');

                rbm.b  = zeros(nVis, 1, 'gpuArray');
                rbm.vb = zeros(nVis, 1, 'gpuArray');

                rbm.c  = zeros(nHidden, 1, 'gpuArray');
                rbm.vc = zeros(nHidden, 1, 'gpuArray');

                rbm.opts = opts;
            end
        end
        
        function x = rbmdown(rbm, x)
            x = logistic(repmat(rbm.b', size(x, 1), 1) + x * rbm.W);
        end
        function x = rbmup(rbm, x)
            x = logistic(repmat(rbm.c', size(x, 1), 1) + x * rbm.W');
        end
        
        function rbm = train(rbm, x)
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
                    batch = x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
                    v1 = batch;
                    h1 = RBM.sample(repmat(rbm.c', batchsize, 1) + v1 * rbm.W');
                    v2 = RBM.sample(repmat(rbm.b', batchsize, 1) + h1 * rbm.W);
                    h2 = logistic(repmat(rbm.c', batchsize, 1) + v2 * rbm.W');

                    c1 = h1' * v1;
                    c2 = h2' * v2;

                    rbm.vW = momentum * rbm.vW + alpha * (c1 - c2)     / batchsize;
                    rbm.vb = momentum * rbm.vb + alpha * sum(v1 - v2)' / batchsize;
                    rbm.vc = momentum * rbm.vc + alpha * sum(h1 - h2)' / batchsize;

                    rbm.W = rbm.W + rbm.vW;
                    rbm.b = rbm.b + rbm.vb;
                    rbm.c = rbm.c + rbm.vc;

                    err(i) = err(i) + sum(sum((v1 - v2) .^ 2)) / batchsize;
                end

                disp(['epoch ' num2str(i) '/' num2str(numepochs)  '. Average reconstruction error is: ' num2str(err(i) / numbatches)]);

            end
%             plot(err);
        end

    end
end

