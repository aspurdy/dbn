function dbn = dbntrain(dbn, x, y, opts)
    n = numel(dbn.rbm);
    if n == 1
        dbn.rbm{1} = rbmjoint(dbn.rbm{1}, x, y, opts);
    else
        dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x, opts);
        for i = 2 : n-1
            x = rbmup(dbn.rbm{i - 1}, x);
            dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x, opts);
        end
        % train final layer jointly with labels
        x = rbmup(dbn.rbm{n - 1}, x);
        dbn.rbm{n} = rbmjoint(dbn.rbm{n}, x, y, opts);
    end
end
