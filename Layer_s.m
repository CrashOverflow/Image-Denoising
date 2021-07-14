classdef Layer_s < Layer
        % Layer with sigmoid activation function.
    methods
        
        % Cctivation function.
        function g = actfun(~, x)
            
            g=1.0 ./ (1.0 + exp(-x));
        end
        
        % Derivative.
        function g_i = actfun_der(~, x)
            
            g_i=(1.0 ./ (1.0+exp(-x))).*(1-(1.0 ./ (1.0+exp(-x))));
        end
    end
end
