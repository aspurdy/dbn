classdef RBM < matlab.mixin.Heterogeneous & handle
    %RBM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods(Abstract)
        rbmdown(x)
        rbmup(x)
        train(x, opts)
    end
    
    methods(Static)
        function x = sample(P)
            x = double(logsig(P) > rand(size(P)));
        end
    end

end

