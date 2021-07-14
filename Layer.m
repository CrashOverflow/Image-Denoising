classdef (Abstract) Layer    
    properties
        b % bias
        W % weights
        nrNeurons % number of neurons for layer
        a % input of layer
        z % output of layer
        delta % delta of layer
    end
    
    methods
      function layer = Layer(nrNeurons)
          % Layer constructor:
          % Build object of type 'Layer'
          %
          % INPUT PARAMS:
          % - nrNeurons: number of neurons of layer
          %
          % OUTPUT PARAMS:
          % - layer: object of type 'Layer'
          
          % Set number of neurons.
          layer.nrNeurons = nrNeurons;
      end
      
      function layer = initLayer(layer_in, nrBackNeurons, maxW, minW)
          % initLayer
          % Function to inilialize weights and bias of layer
          %
          % INPUT PARAMS:
          % - layer_in: input Layer to update
          % - nrBackNeurons: number of neurons of previous layer
          % - maxW: max weight (upper bound)
          % - minW: min weight (lower bound)
          %
          % OUTPUT PARAMS:
          % - layer: object of type 'Layer' with updated attributes.
          
          % Set layer to update.
          layer = layer_in;
          
          % Set bias of layer (randomize).
          layer.b = (maxW - minW) .* rand(1, layer.nrNeurons) + minW;
          
          % Set weights of layer (randomize).
          layer.W = (maxW - minW) .* rand(layer.nrNeurons, nrBackNeurons) + minW;
      end
    end
    
    methods (Abstract)
        % Set activation function.
        actfun(self, x)
        
        % Set derivative of activation function.
        actfun_der(self, x)
    end
end
