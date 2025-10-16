function [J, grad] = cost_function(params, X, y, lambda, ...
                   input_layer_size, hidden_layer_size, ...
                   output_layer_size)

  % params -> vector containing the weights from the two matrices
  %           Theta1 and Theta2 in an unrolled form (as a column vector)
  % X -> the feature matrix containing the training examples
  % y -> a vector containing the labels (from 1 to 10) for each
  %      training example
  % lambda -> the regularization constant/parameter
  % [input|hidden|output]_layer_size -> the sizes of the three layers

  % J -> the cost function for the current parameters
  % grad -> a column vector with the same length as params
  % These will be used for optimization using fmincg

  % TODO: cost_function implementation

  % TODO1: get Theta1 and Theta2 (from params). Hint: reshape

  % Reshape the unrolled params vector into Theta1 and Theta2
  lt1 = hidden_layer_size * (input_layer_size + 1);
  Theta1 = reshape(params(1 : lt1), hidden_layer_size, input_layer_size + 1);
  Theta2 = reshape(params(lt1 + 1 : end), output_layer_size, hidden_layer_size + 1);

  % TODO2: Forward propagation
  [m,n] = size(X);
  a1 = [ones(m,1), X];
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 = [ones(m,1), a2];
  z3 = a2 * Theta2';
  a3 = sigmoid(z3);

  % Convert y to one-hot encoding
  Y = zeros(m,output_layer_size);
  for i = 1 : m
    Y(i,y(i)) = 1;
  endfor

  % TODO3: Compute the error in the output layer and perform backpropagation
  cost = sum(sum(-Y .* log(a3) - (1 - Y) .* log(1 - a3))) / m;
  reg = lambda / (2 * m) * (sum(sum(Theta1(:, 2 : end) .^ 2)) + sum(sum(Theta2(:, 2 : end) .^ 2)));
  J = cost + reg;

  % Backpropagation
  Err3 = a3 - Y;
  delta2 = Err3' * a2;
  Err2 =  (Err3 * Theta2)(:, 2 : end) .* (sigmoid(z2) .* (1 - sigmoid(z2)));
  delta1 = Err2' * a1;

  % TODO4: Determine the gradients
  grad1 = delta1 / m;
  grad2 = delta2 / m;
  grad1(:, 2 : end) += (lambda / m) * Theta1(:, 2 : end);
  grad2(:, 2 : end) += (lambda / m) * Theta2(:, 2 : end);

  % TODO5: Final J and grad
  grad = [grad1(:); grad2(:)];
end
