function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
alpha = 1;%Test
% You need to return the following variables correctly 
J = 0;
h = 0;
s = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
%for iter = 1:m
    h = X*theta;
    s = sigmoid(h);
    J = -(sum(y.*log(s) + (1-y).*log(1 - s)))/m;
    %J = (-y'*log(s)-(1-y)'*log(1-s))/m;
    grad = X'*(sigmoid(h)-y)/m;
   
   % 	theta(1,1) = theta(1,1) - (alpha/m) * sum((s - y).*X(:,1));
   %	theta(2,1) = theta(2,1) - (alpha/m) * sum((s - y).*X(:,2));
   %    theta(3,1) = theta(3,1) - (alpha/m) * sum((s - y).*X(:,3));
   % 	grad = [theta(1,1);theta(2,1);theta(3,1)];
	
    % ============================================================

%end







% =============================================================


