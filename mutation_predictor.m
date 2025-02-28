% rng(42) is used to set the random number generator (RNG) seed to ensure reproducibility of random number generation.

rng(42);

% Define the input and target data

load "path to the .mat file for the input Raman spectra";

% after loading the Raman spectra into 'data' matrix, we need to normalize all the spectra in the range of 0 to 1.
data =  normalize(data,'range');

% labels are the number of particular base in that sample, herein A represent the number of adenine. The labels are also normalized in min-max scenario.
labels = A/max(A);

% In order to cut down the computational cost and training time, PCA is utilised to reduce the dimensions and top 35 component were selected as an input to the ANN model.
[coeff score] = pca(data);
zz = coeff(:,1:35);

% we need to randomise the data and divide it into training and testing. The training set is later divided into training and validation.
idx = randi([1, 1094], 1094, 1);
x_train = zz(:,idx(1:930,:));
t_train = labels(idx(1:930,:));
x_test = zz(:,idx(931:end,:));
t_test = labels(idx(931:end,:));

x = x_train';
t = t_train';


% Possible training functions
trainingFunctions = {'trainlm', 'trainbr', 'trainscg'};

% Generate a range of hidden layer configurations with different neurons in each layer
maxHiddenLayers = 5;  % You can increase this number to explore more layers
maxNeuronsPerLayer = 40;  % Maximum neurons per layer
hiddenLayerConfigs = {};

for numLayers = 1:maxHiddenLayers
    for layerConfig = combnk(1:maxNeuronsPerLayer, numLayers)'
        hiddenLayerConfigs{end+1} = layerConfig';
    end
end

% Initialize variables for tracking the best model
bestNet = [];
minTestLoss = inf;
bestIteration = 0;
bestConfig = '';

% Create a directory to save models
saveDir = 'path to directory to save the model';
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

% Open a file to save all losses
allLossesFile = fullfile(saveDir, 'all_test_losses.txt');
fid_allLosses = fopen(allLossesFile, 'w');

iteration = 0;

% Iterate over all configurations of hidden layers and training functions
for i = 1:length(trainingFunctions)
    for j = 1:length(hiddenLayerConfigs)
        % Update iteration count
        iteration = iteration + 1;

        % Get current training function and hidden layer configuration
        trainFcn = trainingFunctions{i};
        hiddenLayerSize = hiddenLayerConfigs{i);

        % Create and configure the network
        net = fitnet(hiddenLayerSize, trainFcn);
        net.input.processFcns = {'removeconstantrows', 'mapminmax'};
        net.output.processFcns = {'removeconstantrows', 'mapminmax'};
        net.divideFcn = 'dividerand';  % Divide data randomly
        net.divideMode = 'sample';  % Divide up every sample
        net.divideParam.trainRatio = 82.366;
        net.divideParam.valRatio = 100-82.366;
        net.performFcn = 'mse';  % Mean Squared Error
        net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', ...
            'plotregression', 'plotfit'};

        % Train the Network
        [net, tr] = train(net, x, t);

        % Test the Network
        y_test = net(x_test);
        testLoss = perform(net, t_test, y_test);

        % Save the current model
        configName = sprintf('config_%d_%s_%s', iteration, trainFcn, mat2str(hiddenLayerSize));
        modelFilename = fullfile(saveDir, sprintf('%s_net.mat', configName));
        save(modelFilename, 'net');
        
        functionFilename = fullfile(saveDir, sprintf('%s_myNeuralNetworkFunction', configName));
        genFunction(net, functionFilename, 'MatrixOnly', 'yes');
        
        % Append the current test loss to the all_losses file
        fprintf(fid_allLosses, 'Iteration %d: Config = %s, Test loss = %.6f\n', iteration, configName, testLoss);
        
        % Check if the current model is the best so far
        if testLoss < minTestLoss
            minTestLoss = testLoss;
            bestNet = net;
            bestIteration = iteration;
            bestConfig = configName;
            bestModelFilename = modelFilename;
            bestFunctionFilename = functionFilename;
            fprintf('Iteration %d: Test loss improved to %.6f. Model saved.\n', iteration, testLoss);
        else
            fprintf('Iteration %d: Test loss did not improve (%.6f).\n', iteration, testLoss);
        end
    end
end

% Close the all_losses file
fclose(fid_allLosses);

% Save the best model separately
if ~isempty(bestNet)
    save(fullfile(saveDir, 'best_net.mat'), 'bestNet');
    copyfile([bestFunctionFilename '.m'], fullfile(saveDir, 'best_myNeuralNetworkFunction.m'));
    fprintf('Best model saved: Iteration %d, Config = %s, Test loss = %.6f.\n', bestIteration, bestConfig, minTestLoss);
end

% Final message
fprintf('Training completed. Best test loss: %.6f.\n', minTestLoss);

% To use a saved function
% y = myNeuralNetworkFunction_iterX(x_test);  % Replace X with the desired iteration number
