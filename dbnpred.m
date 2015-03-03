function err = dbnpred(dbn, x, y)
    n = numel(dbn.rbm);
    m = size(y, 1);
    c = size(y, 2);
    % do an upward pass through the network for the test examples
    % to compute the feature activations in the penultimate layer
    for i = 2 : n
        x = rbmup(dbn.rbm{i - 1}, x);
    end
    rbm = dbn.rbm{n};
    % precompute for efficiency
    precom = repmat(rbm.c', m, 1) + x * rbm.W';
    py = zeros(m,c);
    % probablities aren't normalized
    for i = 1:c
        py(:, i) = exp(rbm.d(i)) * prod(1 + exp(precom + repmat(rbm.U(:, i)', m, 1)), 2);
    end
    [~, I] = max(py, [], 2);
    prediction = bsxfun(@eq, I, 1:10);
    err = sum(sum(abs(y - prediction)))/2;
    err = err / m;
end