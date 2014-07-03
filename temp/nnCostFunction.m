function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1)); % remember s(j+1) x s(j)+1

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


Y = eye(num_labels)(y,:); % [5000, 10]

a1 = [ones(m, 1) X]; % 5000 * 401
z2 = a1*Theta1';  % 5000x25
a2 = sigmoid(z2); % 5000x25
a2 = [ones(m, 1) a2]; %5000x26
z3 = a2*Theta2'; %5000x10
hypo = sigmoid(z3); %5000x10
% output hypothesis calculated
y1 = -Y .* log(hypo);
y0 = -(1 - Y) .* log(1 - hypo);
temp = (y1 + y0);
J =  (1/m) * sum(temp(:));
% Cost calculated
% Yet still not regularized

%  regularization
modifiedTheta1 = Theta1(:,2:end);
modifiedTheta2 = Theta2(:,2:end);
%should not change involve theta0

regular = (lambda / (2*m)) * (sumsq(modifiedTheta1(:)) + sumsq(modifiedTheta2(:)));
J = J + regular;
% finished regularization

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

Delta1 = 0;
Delta2 = 0;

% Theta1 25 x 401
% Theta2 10 x 26

for t = 1:m;
	% Step 1 - Forward Propagation
	a1 = [1; X(t,:)']; % 401 x 1
	z2 = Theta1 * a1; % 25 x 1
	a2 = sigmoid(z2); % 25 x 1
	a2 = [1; a2]; % 26 x 1
	z3 = Theta2 * a2; % 10 x 1
	a3 = sigmoid(z3); % 10 x 1 i.e hypothesis of the output
	% Step 2 - Calculating Error of Hypothesis Output
	DeltaError3 = a3 - Y(t,:)' ; % 10x1
	% Step 3 - Calculating Error of Hidden Layer 1 
	DeltaError2 = (modifiedTheta2' * DeltaError3) .* sigmoidGradient(z2); % 25x1
	% Step 4 - Accumulating Delta Value 
	Delta1 = Delta1 + (DeltaError2 * a1'); %   25 x 401
	Delta2 = Delta2 + (DeltaError3 * a2'); %  10 x 25
endfor

% Step 5 - Averaging the Deltas 
Theta1_grad = (1/m) * Delta1; 
Theta2_grad = (1/m) * Delta2;

%%%%%%%%%%%%%%%%%%%%%5
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Regularzation of gradients
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda / m) * modifiedTheta1);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda / m) * modifiedTheta2);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
