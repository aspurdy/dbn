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
            x = logsig(repmat(rbm.b', size(x, 1), 1) + x * rbm.W);
        end
        function x = rbmup(rbm, x)
            x = logsig(repmat(rbm.c', size(x, 1), 1) + x * rbm.W');
        end
        
        function rbm = train(rbm, x)            
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
                    % positive phase
                    batch =  x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
                    v1 = batch;
                    h1 = RBM.sample(repmat(rbm.c', batchsize, 1) + v1 * rbm.W');
                    
                    % initialze the persitant chain
                    if i == 1 && l == 1
                       h2 = h1; 
                    end
                    
                    % do k steps of gibbs sampling for negative phase
                    for j = 1:k
                        v2 = RBM.sample(repmat(rbm.b', batchsize, 1) + h2 * rbm.W);
                        h2 = RBM.sample(repmat(rbm.c', batchsize, 1) + v2 * rbm.W');
                    end

                    c1 = h1' * v1;
                    c2 = h2' * v2;
                    
                    rbm.vW = momentum * rbm.vW + alpha * (c1 - c2 - decay * rbm.W) / batchsize;
                    rbm.vb = momentum * rbm.vb + alpha * (sum(v1 - v2)' - decay * rbm.b) / batchsize;
                    rbm.vc = momentum * rbm.vc + alpha * (sum(h1 - h2)' - decay * rbm.c) / batchsize;

                    rbm.W = rbm.W + rbm.vW;
                    rbm.b = rbm.b + rbm.vb;
                    rbm.c = rbm.c + rbm.vc;

                end
                fprintf('epoch %d / %d\n', i, numepochs);
            end
        end
    end
end

