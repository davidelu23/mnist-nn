# mnist
Repository containing files for training and evaluating a simple feedforward neural
network loop for classification using the mnist dataset split into training and
test subsets. The network supports a single hidden layer, and weights are
optimized using backpropagation. The solution to this problem follows the basic
method for solving classification problems proposed in the problem statement,
the only improvement I tried was using the softmax function instead of the
sigmoid in the output layer to present the predictions as probabilities, but
after some tries I came to the conclussion that there was no significant
improvement, so I scrapped the idea.

Functions:

~load_dataset(path): Loads a file from the specified path and returns the
feature matrix X and corresponding label vector y.

~split_dataset(X, y, percent): Shuffles using randperm and splits the dataset
into training and test sets. The proportion percent of examples is used for
training, the rest for testing.

~initialize_weights(L_prev, L_next): Initializes weights randomly in the
range (−ϵ, ϵ) between two layers with L_prev and L_next neurons.

~cost_function(params, X, y, lambda, input_layer_size, hidden_layer_size,
output_layer_size): Computes the neural network cost and gradient using forward
and backward propagation. Returns both the scalar cost J and unrolled gradients
grad.

~predict_classes(X, weights, input_layer_size, hidden_layer_size,
output_layer_size): Predicts class labels for each row in X using trained
weights and a single forward pass through the network (~90% accuracy, could
be better given more iterations, but ~90% is acceptable given the context).
