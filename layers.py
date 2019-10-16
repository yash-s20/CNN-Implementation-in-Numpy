import numpy as np


def clip_exp(x):
    return np.exp(np.clip(x, -100, 100))


class FullyConnectedLayer:
    def __init__(self, in_nodes, out_nodes, activation):
        # Method to initialize a Fully Connected Layer
        # Parameters
        # in_nodes - number of input nodes of this layer
        # out_nodes - number of output nodes of this layer
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.activation = activation
        # Stores the outgoing summation of weights * feautres
        self.data = None

        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0, 0.1, (in_nodes, out_nodes))
        self.biases = np.random.normal(0, 0.1, (1, out_nodes))

    ###############################################
    # NOTE: You must NOT change the above code but you can add extra variables if necessary

    def forwardpass(self, X):
        # print('Forward FC ',self.weights.shape)
        # Input
        # activations : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer

        n = X.shape[0]  # batch size
        # INPUT activation matrix  		:[n X self.in_nodes]
        # OUTPUT activation matrix		:[n X self.out_nodes]
        assert self.weights.shape[0] == X.shape[1]

        ###############################################
        # TASK 1 - YOUR CODE HERE
        sum_l = X @ self.weights + self.biases
        if self.activation == 'relu':
            self.data = relu_of_X(sum_l)
            return self.data
        elif self.activation == 'softmax':
            self.data = softmax_of_X(sum_l)
            return self.data
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit(1)

    ###############################################

    def backwardpass(self, lr, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev

        # Update self.weights and self.biases for this layer by backpropagation
        n = activation_prev.shape[0]  # batch size

        ###############################################
        # TASK 2 - YOUR CODE HERE
        inp_delta = None
        if self.activation == 'relu':
            inp_delta = delta * gradient_relu_of_X(self.data)
        elif self.activation == 'softmax':
            inp_delta = gradient_softmax_of_X(self.data, delta)
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        del_e_w = np.mean(activation_prev[:, :, np.newaxis] @ inp_delta[:, np.newaxis, :], axis=0)
        del_e_p = inp_delta @ self.weights.T
        self.biases -= lr * np.mean(inp_delta, axis=0).reshape([1, -1])
        self.weights -= lr * del_e_w
        return del_e_p
    ###############################################


class ConvolutionLayer:
    def __init__(self, in_channels, filter_size, numfilters, stride, activation):
        # Method to initialize a Convolution Layer
        # Parameters
        # in_channels - list of 3 elements denoting size of input for convolution layer
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer
        # numfilters  - number of feature maps (denoting output depth)
        # stride	  - stride to used during convolution forward pass
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride
        self.activation = activation
        self.out_depth = numfilters
        self.out_row = int((self.in_row - self.filter_row) / self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col) / self.stride + 1)

        # Stores the outgoing summation of weights * features
        self.data = None
        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0, 0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))
        self.biases = np.random.normal(0, 0.1, self.out_depth)

    def forwardpass(self, X):
        # print('Forward CN ',self.weights.shape)
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        n = X.shape[0]  # batch size
        # INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
        # OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

        ###############################################
        # TASK 1 - YOUR CODE HERE
        sum_l = np.zeros([n, self.out_depth, self.out_row, self.out_col])
        for i in range(self.out_row):
            for j in range(self.out_col):
                part_x = X[:, :, i*self.stride:i*self.stride + self.filter_row, j*self.stride:j*self.stride + self.filter_col]
                flattened_part = part_x.reshape([part_x.shape[0], part_x.shape[1], -1])
                sum_l[:, :, i, j] = np.sum(np.einsum('nid,oid->nod', flattened_part,
                                                     self.weights.reshape(list(self.weights.shape[:2]) + [-1])), axis=-1)
        sum_l += self.biases[np.newaxis, :, np.newaxis, np.newaxis]
        if self.activation == 'relu':
            self.data = relu_of_X(sum_l.reshape([n, -1])).reshape(sum_l.shape)
            return self.data
        elif self.activation == 'softmax':
            self.data = softmax_of_X(sum_l.reshape([n, -1])).reshape(sum_l.shape)
            return self.data
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

    ###############################################

    def backwardpass(self, lr, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev

        # Update self.weights and self.biases for this layer by backpropagation
        n = activation_prev.shape[0]  # batch size

        ###############################################
        # TASK 2 - YOUR CODE HERE
        grad = np.zeros_like(self.weights)
        back_delta = np.zeros((n, self.in_depth, self.in_row, self.in_col))
        inp_delta = None
        if self.activation == 'relu':
            inp_delta = (delta.reshape([n, -1]) * gradient_relu_of_X(self.data.reshape([n, -1]))).reshape(self.data.shape)
        elif self.activation == 'softmax':
            inp_delta = gradient_softmax_of_X(self.data.reshape([n, -1]), delta.reshape([n, -1])).reshape(self.data.shape)
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        s = self.stride
        r = self.filter_row
        c = self.filter_col
        for i in range(self.out_row):
            for j in range(self.out_col):
                conv_back = np.einsum('no,oirc->nirc', inp_delta[:, :, i, j], self.weights)
                back_delta[:, :, i*s:i*s+r, j*s:j*s+c] += conv_back

        for i in range(self.out_row):
            for j in range(self.out_col):
                self.weights -= lr * np.mean(np.einsum('no,nirc->noirc', inp_delta[:, :, i, j], activation_prev[:, :, i*s:i*s+r, j*s:j*s+c]), axis=0)
        self.biases -= lr * np.mean(np.sum(inp_delta.reshape([n, self.out_depth, -1]), axis=-1), axis=0)
        return back_delta
    ###############################################


def my_convolve_2d(single_channel, conv_matrix, stride, out_shape):
    # might need these - tile, transpose, max, argmax, repeat, dot, matmul
    sub_shape = conv_matrix.shape
    view_shape = tuple(np.subtract(single_channel.shape, sub_shape) // stride + 1) + sub_shape
    s0, s1 = single_channel.strides
    strides = (stride * s0, stride * s1, s0, s1)
    sub_matrices = np.lib.stride_tricks.as_strided(single_channel, view_shape, strides)
    m = np.einsum('ij,ijkl->kl', conv_matrix, sub_matrices.T).T
    return m


class AvgPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # in_channels - list of 3 elements denoting size of input for max_pooling layer
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row) / self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col) / self.stride + 1)

    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        n = X.shape[0]  # batch size
        # INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
        # OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]
        ###############################################
        # TASK 1 - YOUR CODE HERE
        activation = np.zeros([n, self.out_depth, self.out_row, self.out_col])
        avg_matrix = np.ones([self.filter_row, self.filter_col]) / (self.filter_row * self.filter_col)
        # for i in range(self.out_depth):
        #     # same as in_depth
        #     activation[i, :, :] = my_convolve_2d(X[i, :, :], avg_matrix, self.stride, (self.out_row, self.out_col))
        s = self.stride
        r = self.filter_row
        c = self.filter_col
        for i in range(self.out_row):
            for j in range(self.out_col):
                part_x = X[:, :, i*self.stride:i*self.stride + self.filter_row, j*self.stride:j*self.stride + self.filter_col]
                activation[:, :, i, j] = np.sum(np.sum(part_x * avg_matrix[np.newaxis, np.newaxis, :, :], axis=-1), axis=-1)
        return activation

    ###############################################

    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev

        n = activation_prev.shape[0]  # batch size

        ###############################################
        # TASK 2 - YOUR CODE HERE
        avg_matrix = np.ones([self.filter_row, self.filter_col]) / (self.filter_row * self.filter_col)
        back_delta = np.zeros([n, self.in_depth, self.in_row, self.in_col])
        s = self.stride
        r = self.filter_row
        c = self.filter_col
        for i in range(self.out_row):
            for j in range(self.out_col):
                conv_back = delta[:, :, i:i+1, j:j+1] * avg_matrix[np.newaxis, np.newaxis, :, :]
                back_delta[:, :, i*s:i*s+r, j*s:j*s+c] += conv_back
        return back_delta
    ###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass

    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Function for the activation and its derivative
def relu_of_X(X):
    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu
    return (X > 0) * X


def gradient_relu_of_X(X):
    # Input
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu amd during backwardpass
    return 1.0 * (X > 0)


def softmax_of_X(X):
    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax
    exp_X = clip_exp(X)
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)


def gradient_softmax_of_X(X, delta):
    # Input
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax amd during backwardpass
    # Hint: You might need to compute Jacobian first
    identity = np.eye(X.shape[1])
    neg_x = identity[np.newaxis, :, :] - X[:, :, np.newaxis]
    jacob = X[:, :, np.newaxis] * np.swapaxes(neg_x, 1, 2)
    return (delta[:, np.newaxis, :] @ np.swapaxes(jacob, 1, 2)).squeeze(axis=1)
