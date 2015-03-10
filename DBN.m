classdef DBN < handle
    %DBN Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        sizes
        rbm = BernoulliRBM;
    end
    
    methods
        function dbn = DBN(x, y, sizes, opts)
            n = size(x, 2);
            m = size(y, 2);
            dbn.sizes = [n, sizes];
            nLayers = numel(dbn.sizes) - 1;
            for i = 1 : nLayers - 1
                dbn.rbm(i) = BernoulliRBM(dbn.sizes(i), dbn.sizes(i + 1), opts);
            end
            
            % for the final layer of the dbn we add some additional
            % parameters for modelling the joint distribution of the inputs and target classes
            dbn.rbm(nLayers) = SoftmaxRBM(dbn.sizes(nLayers), dbn.sizes(nLayers + 1), opts, m);

        end

        function train(dbn, x, y)
            n = numel(dbn.rbm);
            for i = 1 : n-1
                train(dbn.rbm(i), x);
                x = rbmup(dbn.rbm(i), x);
            end
            train(dbn.rbm(n), x, y);
        end
        
        function err = predict(dbn, x, y)
            n = numel(dbn.rbm);
            m = size(y, 1);
            c = size(y, 2);
            % do an upward pass through the network for the test examples
            % to compute the feature activations in the penultimate layer
            for i = 2 : n
                x = rbmup(dbn.rbm(i - 1), x);
            end
            
            % precompute for efficiency
            precom = repmat(dbn.rbm(n).c', m, 1) + x * dbn.rbm(n).W';
            py = zeros(m,c, 'gpuArray');
            % probablities aren't normalized
            for i = 1:c
                py(:, i) = exp(dbn.rbm(n).d(i)) * prod(1 + exp(precom + repmat(dbn.rbm(n).U(:, i)', m, 1)), 2);
            end
            [~, I] = max(py, [], 2);
            prediction = bsxfun(@eq, I, 1:10);
            err = sum(sum(abs(y - prediction)))/2;
            err = err / m;
        end

        function x = generate(dbn, class, c, nGibbSteps)
            % randomly initialize a single input
            x = rand(1, dbn.sizes(1));
            n = numel(dbn.rbm);

            % clamp softmax to this label
            y = zeros(1, c);
            y(class) = 1;

            % do an upward pass through the network for the test examples
            % to compute the feature activations in the penultimate layer
            for i = 1 : n - 1
                x = rbmup(dbn.rbm(i), x);
            end

            % do nGibbSteps iterations of gibbs sampling at the top layer
            for i = 1:nGibbSteps - 1
                h = RBM.sample(dbn.rbm(n).c' + x * dbn.rbm(n).W' + y * dbn.rbm(n).U');
                x = RBM.sample(dbn.rbm(n).b' + h * dbn.rbm(n).W);
            end
            h = RBM.sample(dbn.rbm(n).c' + x * dbn.rbm(n).W' + y * dbn.rbm(n).U');
            x = logistic(dbn.rbm(n).b' + h * dbn.rbm(n).W);

            % do a downward pass to generate sample
            for i = n-1:-1:1
                x = rbmdown(dbn.rbm(i), x);
            end
            x = reshape(x, 28, 28)';
        end
    end
    
end

