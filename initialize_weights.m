function [matrix] = initialize_weights(L_prev, L_next)
  % L_prev -> the number of units in the previous layer
  % L_next -> the number of units in the next layer

  % matrix -> the matrix with random values

  % TODO: initialize_weights implementation

  % calculate the range for the weight matrix
  epsilon = sqrt(6) / sqrt(L_next + L_prev);

  % generate random number in the given interval
  matrix = rand(L_next,L_prev + 1) * 2 * epsilon - epsilon;
end
