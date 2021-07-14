classdef Utils
   % In the 'Utils' class there are static methods used in the various 
   % classes, such as activation or error functions, functions for 
   % creating datasets and applying noise, etc.
   
   methods(Static)
        % Sum of square function error used to solve regression problems.
        function z = sumOfSquare(x,y)
           z = 0.5 * sum(((x-y).^2));
        end
        
        % Derivative of sum of square function.
        function z = sumOfSquare_der(x, y)
           z = x - y;
        end
        
        function [image_set,label_set] = createSet(matImages, matLabel, set_dim)
            % Generates a subset of the dataset provided as input to fill 
            % the training, validation and test set.

            % Initializes structures to be output.
            image_set = zeros(set_dim, 784);
            label_set = zeros(set_dim, 10);
            
            % Size of a partition.
            partition = floor(set_dim/10);
           
            % For each image the number of elements inserted in the dataset
            % is counted (not to unbalance it in favor of a class)
            labelCount = zeros(1, 10);
            
            % Counter of elements inserted in set
            elemCount = 0;
            
            % Used to track the location in the dataset.
            index = zeros(1, 60000);
            
            % Fill the set (of set_dim size)
            while elemCount < set_dim
                
                % Select a random image from the dataset
                randPos = floor((60000-1) .* rand(1)+1);
                
                % Check has not already been inserted
                if index(randPos) == 0
                    % Check not to exceed the allowed number of images, of
                    % a class, to be inserted
                    if labelCount(matLabel(randPos)+1) < partition
                        % Update support variables
                        labelCount(matLabel(randPos)+1) = labelCount(matLabel(randPos)+1)+1;
                        elemCount = elemCount+1;
                        index(randPos) = 1;
                        
                        % Insert elem in set (images and label)
                        image_set(elemCount,:) = matImages(randPos,:);
                        label_set(elemCount,matLabel(randPos)+1) = 1;
                    end
                end
            end
        end
        
        function noisyImage = addNoise(inputSet, v, noiseType)
            % addNoise
            % Adds the noise type 'noiseType' with destruction percentage
            % 'v' to the images contained in 'inputSet'
            %
            % INPUT PARAMS:
            % - inputSet: set of images to which to apply noise (MNIST)
            % - v: destruction percentage
            % - noiseType: type of noise to be applied.
            %
            % OUTPUT PARAMS:
            % - noisyImage: set of images to which noise has been applied
            %
            % Supported noise:
            %  # "Standard": as in refferal paper
            %  # "GaussianStandard": provided by MatLab (avg 0, stdev 0.01)
            %  # "GaussianManual": Gaussian noise with intensity v/1000
            %  # "SaltnPepper" salt-and-pepper noise with intensity v/1000
            % 
          
            % Copy input set.
            noisyImage = inputSet;
            
            % Dimension of input set
            set_dim = size(inputSet, 1);
            
            % Calculare how many pixels for each image have to apply noise  
            % in relation to the number of pixels in the image and
            % percentage of destruction.
            dirtyPixel = floor((v * 784) / 100);
            
            % Switch-case to manage supported types of noise.
            switch noiseType
                case 'Standard'
                % For each image generates as much 0 as the percentage
                % of destruction.
                    for i = 1 : set_dim
                        
                        % Array of dirty pixel.
                        dirtyPx_arr=zeros(1, 784);
                        
                        % Dirty pixel counter.
                        dirtyPx_count=0;
                        
                        % Iterates until the number of pixels to be dirty
                        % is reached.
                        while dirtyPx_count < dirtyPixel
                            
                            % Choice randomly pixel to be dirty.
                            randPx = floor((784 - 1) .* rand(1) + 1);
                            
                            % Check that the pixel is not already dirty, 
                            % if you skip this cycle and find another one.
                            if dirtyPx_arr(randPx) == 0
                                
                                % Dirty the pixel.
                                noisyImage(i, randPx) = 0;
                                
                                % Marks the pixel that has been soiled.
                                dirtyPx_arr(randPx) = 1;
                                
                                % Update counter.
                                dirtyPx_count = dirtyPx_count + 1;
                            end
                        end
                    end
                    
                case 'GaussianStandard'
                % Provided by MatLab: average=0, standard deviation=0.01
                
                    % For each image in range [1, set_dim].
                    for i=1:set_dim
                            % Reshape image
                            digit = reshape(inputSet(i, :), [28, 28]);

                            % Applying noise to the image ('gaussian' type).
                            noisyDigit = imnoise(digit, 'gaussian');

                            % Add noisy immage to return set.
                            noisyImage(i, :) = reshape(noisyDigit, [784, 1]);     
                    end

                case 'GaussianManual'
                % Intensity = v/1000 (can be set as preferred).
                   
                    % For each image in range [1, set_dim]. 
                    for i = 1 : set_dim
                        
                        % Reshape image
                        digit = reshape(inputSet(i, :), [28, 28]);
                        
                        % Applying noise to the image (as descripted).
                        noisyDigit = double(digit) + (v/1000) * randn(size(digit));
                        
                        % Add noisy immage to return set.
                        noisyImage(i, :) = reshape(noisyDigit, [784, 1]);     
                    end

                case 'SaltnPepper'
                % Intensity = v/1000
                    
                    % For each image in range [1, set_dim].
                    for i=1:set_dim
                        
                        % Reshape image
                        digit = reshape(inputSet(i, :), [28, 28]);
                        
                        % Applying noise to the image.
                        noisyDigit = imnoise(digit, 'salt & pepper', v/1000);
                        
                         % Add noisy immage to return set.
                        noisyImage(i, :) = reshape(noisyDigit, [784, 1]);     
                    end
                    
                otherwise
                fprintf("\n[ERROR]: Type of noise not supported!\nPlease check the 'addNoise' function in Utils\nfor the list of supported noises.");
            end
        end
        
        function [images, labels] = loadData(imagePath, labelPath)
            % loadData:
            % Loading the MNIST dataset into the set images and labels.
            %
            % INPUT PARAMS:
            % - imagePath: path that contains the whole MNIST image dataset
            % - labelPath: path that contains the whole MNIST labels
            %              dataset (of the corresponding images)
            %
            % OUTPUT PARAMS:
            % - images: 60000x784 matrix of images loaded from the path
            % - labels: 60000x1 array of labels loaded from the path
            %

            fp = fopen(imagePath, 'rb');
            assert(fp ~= -1, ['Could not open ', imagePath, '']);

            magic = fread(fp, 1, 'int32', 0, 'ieee-be');
            assert(magic == 2051, ['Bad magic number in ', imagePath, '']);

            numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
            numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
            numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

            images = fread(fp, inf, 'unsigned char');
            images = reshape(images, numCols, numRows, numImages);
            images = permute(images,[2 1 3]);

            fclose(fp);

            % Reshape to #pixels x #examples
            images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
            
            % Convert to double and rescale to [0,1]
            images = (double(images) / 255)';
            
            fp = fopen(labelPath, 'rb');
            assert(fp ~= -1, ['Could not open ', labelPath, '']);

            magic = fread(fp, 1, 'int32', 0, 'ieee-be');
            assert(magic == 2049, ['Bad magic number in ', labelPath, '']);

            numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

            labels = fread(fp, inf, 'unsigned char');

            assert(size(labels,1) == numLabels, 'Mismatch in label count');

            fclose(fp);
        end     
   end
end