function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

J = 1/m*sum(- y.*log(sigmoid(X * theta)) - (1-y).*log(1-sigmoid(X * theta)));
% for n = 1:m
%     % hh = X(n,:)
%     % vee = grad
%     % X(n) * grad
%     % sigmoid(X(n) * grad)
%     % y(n)*log10(sigmoid(X(n) * grad))
%     J = J - y(n)*log(sigmoid(X(n,:) * grad)) - (1-y(n))*log(1-sigmoid(X(n,:) * grad));
% end
% J = 1/m  * J;

grad(1) = 1/m * X(:,1)'*(sigmoid(X*theta) - y);
grad(2) = 1/m * X(:,2)'*(sigmoid(X*theta) - y);
grad(3) = 1/m * X(:,3)'*(sigmoid(X*theta) - y);
  









% =============================================================

end
