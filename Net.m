classdef Net
   properties
      nrInputNeurons % number of input neurons
      nrOutputNeurons % number of output neurons
      
      nrHiddenLevels % number of hidden layers
      nrLevels % number of total layers
      layer   % network layer arrays
      X % input
      
      alpha % alpha used in the GL function (stop criteria)
      
      % Error function and its derivative (this class has been provided of
      % sumOfSquare error function pointers from the main because we're
      % solving regression problems.
      errorFunction
      errorFunction_der
      
   end
   methods
      function net = Net(nrInputNeurons, layerParams, minWeight, maxWeight, errfun, errfun_der, alpha)
        % Net constructor
        % Create an object of type 'net' (feed-forward neural network)
        %
        % INPUT PARAMS:
        % - nrInputNeurons: number of input neurons.
        % - layerParams: defining autoencoder's layers details.
        % - minWeight: lower bound for the weights of network connections
        % - maxWeight: uppder bound for the weights of network connections
        % - errFun: error function
        % - errFun_der: derivative of error function
        % - alpha: alpha used in Generalization Loss stop criteria.
        %
        % OUTPUT PARAMS:
        % Neural network object.

        % Settings net attributes
        net.nrInputNeurons = nrInputNeurons;
        net.nrOutputNeurons = layerParams(size(layerParams)).n_nodes;
        net.nrHiddenLevels = size(layerParams,2)-1;
        net.nrLevels = size(layerParams,2)+1;
        net.alpha = alpha;

        % Setting net error functions
        net.errorFunction = errfun;
        net.errorFunction_der = errfun_der;
        
        % Create layer for each level according to the type of activation
        % function chosen (sigmoid or relu)
        for i = 1 : net.nrLevels-1
            if strcmp(layerParams(i).act_function, "sigmoid")
                
                net.layer{i} = Layer_s(layerParams(i).n_nodes);
                
            elseif strcmp(layerParams(i).act_function, "relu")
                
                net.layer{i} = Layer_r(layerParams(i).n_nodes);
                
            else
                %net.layer{i}=Layer_s(layerParams(i).n_nodes);
                fprintf("Chosen activation function hasn't got any Layer object!")
            end
        end
        
        % Initialize each layer
        % Input layer
        net.layer{1} = net.layer{1}.initLayer(net.nrInputNeurons, maxWeight, minWeight);

        % Hidden layers
        if net.nrHiddenLevels >= 2
            for i = 2 : net.nrHiddenLevels
                net.layer{i} = net.layer{i}.initLayer(net.layer{i-1}.nrNeurons, maxWeight, minWeight);
            end
        end
        
        % Output layer
        net.layer{net.nrLevels-1} = net.layer{net.nrLevels-1}.initLayer(net.layer{net.nrHiddenLevels}.nrNeurons, maxWeight, minWeight);
      
      end 
      
      function net = forwardPropagation(net_in, X)
        % forwardPropagation
        % forward propagation function for train a net
        % INPUT PARAMS:
        % - net: neural network of type 'Net'
        % - X: input matrix (MNIST images)
        % 
        % OUTPUT PARAMS:
        % - net: trained neural network
        
       
        net = net_in; % Set output param with input net
        net.X = X; % Set net input
        z_prev = X; % Previous output array

        % For each level calculate output as required by the feed forward 
        for i = 1 : net.nrHiddenLevels+1
            net.layer{i}.a = (z_prev*net.layer{i}.W');
            net.layer{i}.a = net.layer{i}.a+net.layer{i}.b;
            net.layer{i}.z = net.layer{i}.actfun(net.layer{i}.a);
            z_prev = net.layer{i}.z;
        end
      end
      
      function [bestnet, errorTS_arr, errorVS_arr, totEpochs] = training(net, nEpochs, trainingSetImg, validationSetImg, trainingSetLabel, validationSetLabel, eta)
        % training
        % Train neural network (object of type 'Net') using feed forward
        % back propagation as train function and gradient descent 
        % for update weights
        %
        % INPUT PARAMS:
        % net: neural network (object of type 'Net')
        % nEpochs: number of epochs to perform
        % trainingSetImg: training set (matrix of MNIST images)
        % validationSetImg: validation set (matrix of MNIST images)
        % trainingSetLabel: training set labels (matrix of MNIST label)
        % validationSetLabel: validation set labels (matrix of MNIST label)
        % eta: learning rate
        %
        % OUTPUT PARAMS:
        % net: trained neural net (object of type 'Net')
        % errorTS_arr: array errors gotten on training set for each epoch
        % errorVS_arr: array errors gotten on validation set for each epoch
        % totEpochs: epochs performed before stopped by stop criteria (GL)
        
        % array of errors on training set
        errorTS_arr = zeros(1, nEpochs); 
        
        % array of errors on validation set
        errorVS_arr = zeros(1, nEpochs);

        % Minimum number of epochs to perform
        minEpochsRequired = floor(nEpochs/3); 
        
        bestnet = net; % save current net as best.
        optErr = realmax;
        
        % Training for each epoch
        for e = 1 : nEpochs           
            % Forward propagation on training set 
            forwardPropTS = net.forwardPropagation(trainingSetImg);
            
            % Forward propagation on validation set
            forwardPropVS = net.forwardPropagation(validationSetImg);  
            
            % Calculate error on training set and validation set 
            errorTS = sum(net.errorFunction(forwardPropTS.layer{forwardPropTS.nrHiddenLevels+1}.z, trainingSetLabel)) / size(trainingSetImg, 1);
            errorVS = sum(net.errorFunction(forwardPropVS.layer{forwardPropVS.nrHiddenLevels+1}.z, validationSetLabel)) / size(validationSetImg, 1);
            
            errorTS_arr(e) = errorTS; % set current epoch training error
            errorVS_arr(e) = errorVS; % set current epoch validaiton error
            
            % Back propagation on training set
            forwardPropTS = forwardPropTS.backPropagation(trainingSetLabel);
            
            % Calculare partial derivative for bias and weights
            [bias_der, weights_der] = derWeights(forwardPropTS);
            
            % Update weights
            forwardPropTS = updateWeights(forwardPropTS, bias_der, weights_der, eta);
            
            net = forwardPropTS; % update trained net
            
            if(errorVS < optErr)
                optErr = errorVS;
                bestnet = net; % update best net
            end 
            
            totEpochs = e; % update totEpochs

            % Stopping criteria Generalization Loss 
            fprintf("\nil minimo è %d", optErr);
            if e > minEpochsRequired && net.stopGL(errorVS, optErr) == 1
               break;
            end
            
            fprintf("current epoch: %d\n",e);
        end
      end
      
      function net = backPropagation(net_in, T)
            % backPropagation
            % Calculate deltas from output layer to input layer, as 
            % provided back-propagation function
            % 
            % INPUT PARAMS:
            % - net: neural network (object of type 'Net')
            % - T: target used to compare output values (MNIST labels)
            % 
            % OUTPUT PARAMS:
            % - net: neural network with updated delta and partial
            %        derivatives.
            
            net = net_in; % set input net
            
            % Save output level index (to improve the next steps).
            outputLayer_index = net.nrHiddenLevels+1;
            
            % Calculate deltas for output level.
            net.layer{outputLayer_index}.delta = net.layer{net.nrLevels-1}.actfun_der(net.layer{outputLayer_index}.a) .* net.errorFunction_der(net.layer{outputLayer_index}.z, T);
            
            % Back-propagation (from output layer to input layer calculate
            % deltas starting from last output delta and go on.
            for i = outputLayer_index-1 : -1 : 1
                % Calculate product for each delta (separated from the 
                % next step for readability)
                deltaWeights_product = net.layer{i+1}.delta * net.layer{i+1}.W;
                
                % Calculate delta
                net.layer{i}.delta = net.layer{i}.actfun_der(net.layer{i}.a) .* deltaWeights_product;
            end
      end
        
      
      function [bias_der, weights_der] = derWeights(net_in)
        % derWeights
        % Calculate, for each level, partial derivatives of error function
        % giving weights and bias of level.
        %
        % INPUT PARAMS:
        % - net_in: neural network updated by back-propagation.
        %
        % OUTPUT PARAMS:
        % - bias_der: derivative of bias
        % - weights_der: derivative of weihgts

        % Sum derivative of bias (for update).
        bias_der{1}=sum(net_in.layer{1}.delta, 1);
        
        % Product of matrix of deltas and input.
        weights_der{1}=net_in.layer{1}.delta' * net_in.X;
       
        % Calculate derivative of bias and weights for hidden and output
        % levels, as calculated before.
        for i = 2 : net_in.nrHiddenLevels+1
            % Sum derivative of bias (for update).        
            bias_der{i}=sum(net_in.layer{i}.delta,1);
            
            % Product of matrix of deltas and input.
            weights_der{i}=net_in.layer{i}.delta' * net_in.layer{i-1}.z;
        end
      end

      function net = updateWeights(net, bias_der, weights_der, eta)
        % updateWeights
        % Update net bias and weights using input params.
        % 
        % INPUT PARAMS:
        % - net: neural network (object of type 'Net')
        % - bias_der: derivative of bias
        % - weights_der: derivative of weights
        % - eta: learning rate
        %
        % OUTPUT PARAMS:
        % - net: neural network (object of type 'Net') updated

        % For each layer subtract bias and weights values respectively
        % with bias_der and weights_der, multiplied by eta.
        for i=1:net.nrHiddenLevels+1
                net.layer{i}.b=net.layer{i}.b-(eta*bias_der{i});
                net.layer{i}.W=net.layer{i}.W-(eta*weights_der{i});
        end
      end
      
      function stop_flag = stopGL(net, curr_err, opt_err)
          % stopGL
          % Generalization Loss stop criteria.
          % 
          % INPUT PARAMS:
          % - net: neural network (object of type 'Net')
          % - curr_err: error obtained at current epoch
          % - opt_err: best error obtained so far in the training phase 
          %
          % OUTPUT PARAMS:
          % - stop_flag: 0 - stop criteria not satisfied
          %              1 - stop criteria satisfied
          
          % Set return flag with 0.
          stop_flag = 0;
          
          % Calculate GL value.
          GL_epoch = 100 .* ((curr_err ./ opt_err) - 1);
          
          % Print GL calculated.
          fprintf("\nGL è %d\n", GL_epoch);
          
          % Check if GL calculated exceeds the network alpha (upper limit)
          if GL_epoch > net.alpha
              % If exceeds, set flag with 1
              stop_flag = 1;
          end
      end
   end
end