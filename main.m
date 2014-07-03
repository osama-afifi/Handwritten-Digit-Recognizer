%% Osama M. Afifi
%% 29/06/2013
%% Digit Recognition Labeling using Neural Networks

%  Procedure
%  ------------
% 
%	1.1  Load and Manipulate the Data.
%	1.2  Data Synthesis (Optional).
%	2    Initializing Parameters
%	3.1  Train Neural Network
%	3.2  Find Suitable Reg. Parameter
%	3.3  Find Suitable Iteration Limit
%	4	 Visualize Hidden Layer Weights
%	5	 Predict Labels and Calculate Accuracy Perc.

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this program
input_layer_size  = 784;  % 28x28 Input Images of Digits
hidden_layer_size = 50;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped 0 -> 10)
						  
%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('.................... Phase 1 .......................\n')
fprintf('Loading Data File ...\n')
Data = load('Data/train.csv');
fprintf('Setting up Label Vector ...\n')
y = Data(:,1);
y( y==0 )= 10; % Mapping 0 into 10
fprintf('Setting up Feature Matrix ...\n')
feature_columns = [2 : size(Data,2)];
X = Data(:,feature_columns);
size(X,1)
size(X,2)
m = size(X, 1);
% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
fprintf('Visualize Data ...\n')
displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Initializing Parameters ================
%  A two layer neural network that classifies digits. we will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)
fprintf('.................... Phase 2 .......................\n')
warning('off', 'Octave:possible-matlab-short-circuit-operator');
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =================== Part 3: Training NN ===================
%  To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.

fprintf('.................... Phase 3 .......................\n')
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 200);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Part 4: Visualize Weights =================
%  "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('.................... Phase 4 .......................\n')
fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Part 5: Predict =================
%  After training the neural network, we would like to use it to predict the labels of the training set. This lets
%  you compute the training set accuracy.

fprintf('.................... Phase 5 .......................\n')
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%% ================= Part 6: Predict Testing Data =================
%  After training the neural network, we would like to use it to predict the labels of the tesring data

fprintf('.................... Phase 6 .......................\n')

XTest = load('Data/test.csv');
predTest = predict(Theta1, Theta2, XTest);
predTest( predTest == 10 )= 0; % Mapping 0 into 10
numLables = ([1:size(predTest,1)])';
predTest = [numLables predTest];
csvwrite ('predTest.csv', predTest);
