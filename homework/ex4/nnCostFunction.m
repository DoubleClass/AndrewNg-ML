
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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); 

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
realy = zeros(m, num_labels);
for n = 1:m
    realy(n,y(n)) = 1;
    
end
realy;

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
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
G1 = zeros( size(Theta1) );
G2 = zeros( size(Theta2) );
for t = 1:m
    a_1 = [1 X(t,:)];   % a_1的形状是 1×401  Theta1 25×401 形状
    z_2 = a_1 * Theta1';         % z_2的形状是 1×25
    a_2 = [1 sigmoid(z_2)];
    z_3 = a_2 * Theta2';           %Theta2的形状是 10×26
    a_3 = sigmoid(z_3);            % 1×10  
    err3 = a_3' - realy(t,:)';
    temp = Theta2' * err3;

    err2 = temp(2:end) .* sigmoidGradient(z_2');% 25×1 ;
    
    G1 = G1 +err2 * a_1;
    G2 = G2 + err3 * a_2;
end
Theta1_grad = G1 / m + lambda * [zeros(hidden_layer_size, 1) Theta1(:, 2:end)]/m;
Theta2_grad = G2 / m + lambda * [zeros(num_labels, 1) Theta2(:, 2:end)] / m;

X = [ones(m, 1) X];
oX = X;
X = [ones(m, 1) sigmoid(X * Theta1')];
X = sigmoid(X * Theta2');
for n = 1:m
    J = J - log(X(n,:)) * realy(n,:)' - log(1 - X(n,:)) * (1 - realy(n,:)');
end

Theta1 = Theta1.^2;
Theta2 = Theta2.^2;
temp1 = sum(Theta1);
temp2 = sum(Theta2);

sum1  = sum(Theta1(:));
sum2 = sum(Theta2(:));


J = J/m;

J = J + (sum1 + sum2 - temp1(1) - temp2(1)) * lambda/(2*m);
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
