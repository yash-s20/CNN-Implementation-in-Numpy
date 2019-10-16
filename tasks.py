import nn
import numpy as np
import sys

from util import *
from visualize import *
from layers import *


# XTrain - List of training input Data
# YTrain - Corresponding list of training data labels
# XVal - List of validation input Data
# YVal - Corresponding list of validation data labels
# XTest - List of testing input Data
# YTest - Corresponding list of testing data labels

def init_constant_nn(layer_list, in_nodes, out_nodes, hidden_nodes, alpha, batch_size, epochs):
    nn1 = nn.NeuralNetwork(out_nodes, alpha, batch_size, epochs)
    if len(layer_list) == 1:
        nn1.addLayer(FullyConnectedLayer(in_nodes, out_nodes, layer_list[0]))
    else:
        for i, layer in enumerate(layer_list):
            if i == 0:
                nn1.addLayer(FullyConnectedLayer(in_nodes, hidden_nodes, layer))
            elif i == len(layer_list) - 1:
                nn1.addLayer(FullyConnectedLayer(hidden_nodes, out_nodes, layer))
            else:
                nn1.addLayer(FullyConnectedLayer(hidden_nodes, hidden_nodes, layer))
    return nn1


def taskSquare(draw):
    XTrain, YTrain, XVal, YVal, XTest, YTest = readSquare()
    # Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
    # nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)
    # Add layers to neural network corresponding to inputs and outputs of given data
    # Eg. nn1.addLayer(FullyConnectedLayer(x,y))
    ###############################################
    # TASK 2.1 - YOUR CODE HERE
    epochs = 50
    alpha = 0.01
    batch_size = 20
    hidden_nodes = 7

    nn1 = init_constant_nn(['relu', 'softmax'], XTrain.shape[1], YTrain.shape[1], hidden_nodes, alpha, batch_size, epochs)

    ###############################################
    nn1.train(XTrain, YTrain, XVal, YVal, False, True)
    pred, acc = nn1.validate(XTest, YTest)
    print('Test Accuracy ', acc)
    # Run script visualizeTruth.py to visualize ground truth. Run command 'python3 visualizeTruth.py 2'
    # Use drawSquare(XTest, pred) to visualize YOUR predictions.
    if draw:
        drawSquare(XTest, pred)
    return nn1, XTest, YTest


def taskSemiCircle(draw):
    XTrain, YTrain, XVal, YVal, XTest, YTest = readSemiCircle()
    # Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
    # nn1 = nn.NeuralNetwork(out_nodes, alp, batchSize, epochs)
    # Add layers to neural network corresponding to inputs and outputs of given data
    # Eg. nn1.addLayer(FullyConnectedLayer(x,y))
    ###############################################
    # TASK 2.2 - YOUR CODE HERE
    # raise NotImplementedError
    epochs = 50
    alpha = 0.01
    batch_size = 20
    hidden_nodes = 8
    nn1 = init_constant_nn(['relu', 'softmax'], XTrain.shape[1], YTrain.shape[1], hidden_nodes, alpha, batch_size, epochs)
    ###############################################
    nn1.train(XTrain, YTrain, XVal, YVal, False, True)
    pred, acc = nn1.validate(XTest, YTest)
    print('Test Accuracy ', acc)
    # Run script visualizeTruth.py to visualize ground truth. Run command 'python3 visualizeTruth.py 4'
    # Use drawSemiCircle(XTest, pred) to vnisualize YOUR predictions.
    if draw:
        drawSemiCircle(XTest, pred)
    return nn1, XTest, YTest


def taskMnist():
    XTrain, YTrain, XVal, YVal, XTest, YTest = readMNIST()
    # Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
    # nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)
    # Add layers to neural network corresponding to inputs and outputs of given data
    # Eg. nn1.addLayer(FullyConnectedLayer(x,y))
    ###############################################
    # TASK 2.3 - YOUR CODE HERE

    ### DO NOT TOUCH THIS - gives good accuracy for some seed - 91.25% ###
    epochs = 10
    alpha = 1e-2
    batch_size = 50
    hidden_nodes = 20
    nn1 = init_constant_nn(['relu', 'relu', 'softmax'], XTrain.shape[1],
                           YTrain.shape[1], hidden_nodes, alpha, batch_size, epochs)
    ###############################################
    nn1.train(XTrain, YTrain, XVal, YVal, False, True)
    pred, acc = nn1.validate(XTest, YTest)
    print('Test Accuracy ', acc)
    return nn1, XTest, YTest


def taskCifar10():
    XTrain, YTrain, XVal, YVal, XTest, YTest = readCIFAR10()

    XTrain = XTrain[0:5000, :, :, :]
    XVal = XVal[0:1000, :, :, :]
    XTest = XTest[0:1000, :, :, :]
    YVal = YVal[0:1000, :]
    YTest = YTest[0:1000, :]
    YTrain = YTrain[0:5000, :]

    modelName = 'model.npy'
    # # Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
    # # nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)
    # # Add layers to neural network corresponding to inputs and outputs of given data
    # # Eg. nn1.addLayer(FullyConnectedLayer(x,y))
    # ###############################################
    # # TASK 2.4 - YOUR CODE HERE
    epochs = 100
    alpha = 1e-2
    batch_size = 50
    kernel1 = (5, 5)
    kernel2 = (4, 4)
    filters = 32
    conv_out = 10
    stride1 = 3
    stride2 = 2
    hidden_nodes = filters * 4 * 4


    nn1 = nn.NeuralNetwork(YTrain.shape[1], alpha, batch_size, epochs)
    # current precision without CNN is close to 29% on test.

    # 3 x 32 x 32 to 32 x 10 x 10
    nn1.addLayer(ConvolutionLayer(XTrain.shape[1:], kernel1, filters, stride1, 'relu'))

    # 32 x 10 x 10 to 32 x 4 x 4
    nn1.addLayer(AvgPoolingLayer((filters, conv_out, conv_out), kernel2, stride2))

    # 32 x 4 x 4 to 512
    nn1.addLayer(FlattenLayer())

    # 512 to 10
    nn1.addLayer(FullyConnectedLayer(hidden_nodes, YTrain.shape[1], 'softmax'))
    ###################################################
    nn1.train(XTrain, YTrain, XVal, YVal, True, True, loadModel=False, saveModel=True, modelName=modelName)
    pred, acc = nn1.validate(XTest, YTest)
    print('Test Accuracy ', acc)
    return nn1,  XTest, YTest, modelName # UNCOMMENT THIS LINE WHILE SUBMISSION
