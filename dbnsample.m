function x = dbnsample(dbn, class, c, gibbSteps)

    % randomly initialize a single input
    x = rand(1, dbn.sizes(1));
    n = numel(dbn.rbm);
    
    % clamp softmax to this label
    y = zeros(1, c);
    y(class) = 1;

    % do an upward pass through the network for the test examples
    % to compute the feature activations in the penultimate layer
    for i = 1 : n - 1
        x = rbmup(dbn.rbm{i}, x);
    end
    rbm = dbn.rbm{n};
    
    k = 200;
    % do k iterations of gibbs sampling at the top layer
    for i = 1:gibbSteps
        h = sigm(rbm.c' + x * rbm.W' + y * rbm.U');
        x = sigm(rbm.b' + h * rbm.W);
    end
    h = sigmrnd(rbm.c' + x * rbm.W' + y * rbm.U');
    x = sigmrnd(rbm.b' + h * rbm.W);

    % do a downward pass to generate sample
    for i = n-1:-1:1
        x = rbmdown(dbn.rbm{i}, x);
    end
    x = reshape(x, 28, 28)';
end