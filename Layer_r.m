classdef Layer_r < Layer
        % Layer with RELU activation function.
    methods
        
        % Activation function.
        function g = actfun(~, x)
            
                if x < 0
                    g=0;
                else
                    g=x;
                end
        end
        
        % Derivative.
        function g_i = actfun_der(~, x)
            
                if x < 0
                    g_i=0;
                else
                    g_i=1;
                end
        end
    end
end