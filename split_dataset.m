function [X_train, y_train, X_test, y_test] = split_dataset(X, y, percent)
  % X -> the loaded dataset with all training examples
  % y -> the corresponding labels
  % percent -> fraction of training examples to be put in training dataset

  % X_[train|test] -> the datasets for training and test respectively
  % y_[train|test] -> the corresponding labels

  % Example: [X, y] has 1000 training examples with labels and percent = 0.85
  %           -> X_train will have 850 examples
  %           -> X_test will have the other 150 examples

  % TODO: split_dataset implementation

  [m,nx] = size(X);
  [~,ny] = size(y);

  %randomize the dataset so the model trains on all the numbers
  p = randperm(m);
  X = X(p, :);
  y = y(p);

  idx = floor(percent * m);
  X_train = X(1 : idx, :);
  X_test = X(idx + 1 : end, :);
  y_train = y(1 : idx, :);
  y_test = y(idx + 1 : end, :);
end
