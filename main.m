% main.m:
% Autoencoder construction and training, using MNIST dataset
% Each HyperParam is specified here (and is editable as desired).
%
% After the training phase, there is testing and printing of the results
%(images with noise and reconstructed images) 

% removes all variables from the currently active workspace
%(For clean execution)
clearvars;

tic % start stopwatch timer

% add the dataset path to the workspace (for easy access)
addpath('./dataset_mnist/'); 

trainingSet_dim = 1000; % size of the training set
testSet_dim = 250; % size of the test set
validationSet_dim = 250; % size of validation set


% Defines type of noise to apply on the dataset.
% - Standard: as used in the article
% - GaussianStandard: Standard Gaussian noise (provided by MATLAB)
% - GaussianManual: Gaussian noise (intensity: v/1000)
% - SaltnPepper: classic Salt 'N Pepper noise (intensity: v/1000)
noise_type = "Standard";
 
v = 25; % noise percentage

% loading the dataset (with print of size for checking)
[mnist_images, mnist_labels] = Utils.loadData('./dataset_mnist/train-images-idx3-ubyte', './dataset_mnist/train-labels-idx1-ubyte');

% All datasets (Training, Validation and Test) are loaded randomly in order
% to have heterogeneous inputs.
% Functions such as rand() are used and, when loading a dataset, 
% duplicates are suitably discarded

% NOTA BENE: The MNIST dataset has 10 classes
% Dividing the datasets into 10 partitions of the same size
trainingSet_dim = floor(trainingSet_dim/10)*10;
testSet_dim = floor(testSet_dim/10)*10;
validationSet_dim = floor(validationSet_dim/10)*10;

% Building training, validation and test set from MNIST
[trainset_images, trainset_labels] = Utils.createSet(mnist_images, mnist_labels, trainingSet_dim);
[valset_images,valset_labels] = Utils.createSet(mnist_images, mnist_labels, validationSet_dim);
[testset_images,testset_labels] = Utils.createSet(mnist_images, mnist_labels, testSet_dim);

% Applying noise to training, validation and test set
noisy_trainset = Utils.addNoise(trainset_images,v,noise_type);
noisy_valset = Utils.addNoise(valset_images,v,noise_type);
noisy_testset = Utils.addNoise(testset_images,v,noise_type);


% First net, that serves as a basic autoencoder, parameters and
% iper-parameters
n_features_AutoEncoder = 784;
hiddenNodes_AutoEncoder = 128;
nEpochs_AutoEncoder = 50;
minWeight_Autoencoder = -0.09;
maxWeight_Autoencoder = 0.09;
n_layers_Autoencoder = 2;
eta_AutoEncoder = 0.0002; % used for 1000 dataset
%eta_AutoEncoder = 0.00002; % used for 10000 dataset

% Second net, that serves as a basic autoencoder, parameters and
% iper-parameters
n_features_Denoiser = 784;
hiddenNodes_Denoiser = 32;
nEpochs_Denoiser = 50;
minWeight_Denoiser = -0.09;
maxWeight_Denoiser = 0.09;
n_layers_Denoiser = 4;
eta_Denoiser = 0.00002; % used for 1000 dataset
%eta_Denoiser = 0.000002; % used for 10000 dataset

% Defining autoencoder's layers details
layers_Autoencoder(1) = struct('n_nodes', hiddenNodes_AutoEncoder, 'act_function', "relu");
layers_Autoencoder(2) = struct('n_nodes', n_features_AutoEncoder, 'act_function', "sigmoid");

% Building first autoencoder. 
Autoencoder = Net(784, layers_Autoencoder, minWeight_Autoencoder, maxWeight_Autoencoder, @Utils.sumOfSquare, @Utils.sumOfSquare_der, 2.5); %GL era 2.5

% Setting second layer weights equal to first layer weights to apply tied
% weights technique
Autoencoder.layer{2}.W = Autoencoder.layer{1}.W';

% Training the Autoencoder
fprintf("\n AutoEncoder training started. ");
[Autoencoder, trainingError_Autoencoder, validationError_Autoencoder, totEpochs_Autoencoder] = Autoencoder.training(nEpochs_AutoEncoder, noisy_trainset, noisy_valset, trainset_images, valset_images, eta_AutoEncoder);

% Defining Denoiser's layers details
layers_Denoiser(1) = struct('n_nodes', hiddenNodes_AutoEncoder, 'act_function', "relu");
layers_Denoiser(2) = struct('n_nodes', hiddenNodes_Denoiser, 'act_function', "relu");
layers_Denoiser(3) = struct('n_nodes', hiddenNodes_AutoEncoder, 'act_function', "relu");
layers_Denoiser(4) = struct('n_nodes', n_features_Denoiser , 'act_function', "sigmoid");

% Building the "denoiser" neural network that will inherit weights from the
% encoder
Denoiser = Net(784, layers_Denoiser, minWeight_Denoiser, maxWeight_Denoiser, @Utils.sumOfSquare, @Utils.sumOfSquare_der, 0.8); % GL era 0.8

% Tied weights
Denoiser.layer{1}.W = Autoencoder.layer{1}.W;
Denoiser.layer{3}.W = Denoiser.layer{2}.W';
Denoiser.layer{4}.W = Denoiser.layer{1}.W';

% Training the Denoiser
[Denoiser, trainingError_Denoiser, validationError_Denoiser, totEpochs_Denoiser] = Denoiser.training(nEpochs_Denoiser,noisy_trainset, noisy_valset, trainset_images, valset_images, eta_Denoiser);

% Printing results.
figure('Name','Grafico errore');
plot(trainingError_Denoiser(1:totEpochs_Denoiser));
hold
plot(validationError_Denoiser(1:totEpochs_Denoiser));
legend('Errore rispetto al training set', 'Errore rispetto al validation set');

% Testing
Denoisertest = Denoiser.forwardPropagation(noisy_testset);

% Printing recostructed images
figure('units', 'normalized', 'outerposition', [0 0 1 1], 'Name', 'Input - output');
colormap(gray);

for i = 1 : 5
    subplot(4, 5, i+5)
	digit = reshape(Denoisertest.X(i, :), [28, 28]);
    imagesc(digit)
    axis off
    
    subplot(4, 5, i+10)
    digit2 = reshape(Denoisertest.layer{Denoiser.nrHiddenLevels+1}.z(i, :), [28, 28]);
    imagesc(digit2)
    axis off
end

% Print of configuration used for building Denoiser, training and time occurred.
fprintf("\n Denoising Autoencoder details:\n ");
fprintf("\n nodes for each layers : [784| %d| %d| %d| 784]", hiddenNodes_AutoEncoder, hiddenNodes_Denoiser, hiddenNodes_AutoEncoder)
fprintf("\n eta : %f\n Hidden layers : %d", eta_Denoiser , n_layers_Denoiser);
fprintf("\n Epochs : %d" , nEpochs_Denoiser);
fprintf("\n Epochs performed before the stop criteria: %d\n", totEpochs_Denoiser);

pause(1);

fprintf("\nTotal time occurred: %.2f sec", toc);