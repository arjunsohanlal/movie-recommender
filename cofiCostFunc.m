function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% Initializations
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% Computing the cost function and gradient for collaborative
% filtering, including regularization terms.

% Output details:
% J - Cost function
% X_grad - num_movies x num_features matrix, containing the 
%          partial derivatives w.r.t. to each element of X
% Theta_grad - num_users x num_features matrix, containing the 
%              partial derivatives w.r.t. to each element of Theta
%

h_x = X*Theta';
cost_matrix = R.*(h_x - Y).^2;
J = (1/2)*sum(sum(cost_matrix)) + (lambda/2)*(sum(sum(Theta.^2)) + sum(sum(X.^2)));

for i = 1:num_movies
	X_grad(i,:) = (((X(i,:)*Theta') - Y(i,:)).*R(i,:))*Theta;
	X_grad(i,:) += lambda*X(i,:);
end

for j = 1:num_users
	Theta_grad(j,:) = (((Theta(j,:)*X') - Y(:,j)').*R(:,j)')*X;
	Theta_grad(j,:) += lambda*Theta(j,:);
end

grad = [X_grad(:); Theta_grad(:)];

end
