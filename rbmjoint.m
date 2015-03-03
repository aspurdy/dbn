function rbm = rbmjoint(rbm, x, y, opts)
    size(rbm.U)
    assert(isfloat(x), 'x must be a float');
    assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
    m = size(x, 1);
    numbatches = m / opts.batchsize;
    
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');
    err = zeros(opts.numepochs, 1);
    for i = 1 : opts.numepochs
        kk = randperm(m);
        for l = 1 : numbatches
            batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
            labels = y(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
            
            % positive phase
            x0 = batch;
            y0 = labels;
            h0_hat = sigm(repmat(rbm.c', opts.batchsize, 1) + x0 * rbm.W' + y0 * rbm.U');
            
            % negative phase
            h0 = sigmrnd(repmat(rbm.c', opts.batchsize, 1) + x0 * rbm.W' + y0 * rbm.U');
            y1 = softm(repmat(rbm.d', opts.batchsize, 1) + h0 * rbm.U);

            x1 = sigmrnd(repmat(rbm.b', opts.batchsize, 1) + h0 * rbm.W);
            h1_hat = sigm(repmat(rbm.c', opts.batchsize, 1) + x1 * rbm.W' + y1 * rbm.U');
            
            c1 = h0_hat' * x0;
            c2 = h1_hat' * x1;
            
            d1 = h0_hat' * y0;
            d2 = h1_hat' * y1;
            
            % compute weight updates
            rbm.vW = rbm.momentum * rbm.vW + rbm.alpha * (c1 - c2)     / opts.batchsize;
            rbm.vU = rbm.momentum * rbm.vU + rbm.alpha * (d1 - d2)     / opts.batchsize;
            rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * sum(x0 - x1)' / opts.batchsize;
            rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * sum(h0 - h1_hat)' / opts.batchsize;
            rbm.vd = rbm.momentum * rbm.vd + rbm.alpha * sum(y0 - y1)' / opts.batchsize;

            % update weights
            rbm.W = rbm.W + rbm.vW;
            rbm.U = rbm.U + rbm.vU;
            rbm.b = rbm.b + rbm.vb;
            rbm.c = rbm.c + rbm.vc;
            rbm.d = rbm.d + rbm.vd;
            
            % reconstruction error
            err(i) = err(i) + sum(sum((x0 - x1) .^ 2)) / opts.batchsize;
        end
        
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err(i) / numbatches)]);
        
    end
    plot(err)
end
