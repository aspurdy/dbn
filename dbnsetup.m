function dbn = dbnsetup(dbn, x, y, opts)
    n = size(x, 2);
    m = size(y, 2);
    dbn.sizes = [n, dbn.sizes];

    for u = 1 : numel(dbn.sizes) - 1
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;

        dbn.rbm{u}.W  = zeros(dbn.sizes(u + 1), dbn.sizes(u));
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));

        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1);
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end
    
    % for the final layer of the dbn we add some additional
    % parameters for modelling the joint the joint 
    % distribution of the inputs and target classes
    
    % bias for visible units representing class labels
    dbn.rbm{u}.d = zeros(m, 1);
    dbn.rbm{u}.vd = zeros(m, 1);
    % weights between target labels and hidden units
    dbn.rbm{u}.U = zeros(dbn.sizes(u + 1), m);
    dbn.rbm{u}.vU = zeros(dbn.sizes(u + 1), m);
end
