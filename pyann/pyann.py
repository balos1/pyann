"""
.. moduleauthor:: Cody Balos <cjbalos@gmail.com>
"""

WEIGHTSDIR = '../weights/'
TRAINFILE = '../nnTrainData.txt'
TESTFILE = '../nnTestData.txt'
CLASSES = ['Africa', 'America', 'Antarctica', 'Asia', 'Australia', 'Europe', 'Arctic', 'Atlantic', 'Indian', 'Pacific']
EPOCHS = 50  # optimal perceptron setting = 50, optimal multilayer = 500
RATE = 0.1   # optimal setting for both perceptron and multilayer

import argparse
import csv
import numpy as np
import os
import pprint as pp
from neuron import Neuron
from layer import Layer
from layertype import LayerType
from ann import Ann

def main():
    """Main function.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'test'], action='store', help='command to run')
    parser.add_argument('network', choices=['perceptron', 'multi-layer'], action='store', help='what type of ANN to run')
    parser.add_argument('--weights', action='store', dest='weights', help='load a weights file instead of rng weights')
    args = parser.parse_args()

    network = None
    if args.network == 'perceptron':
        if args.weights:
            weights = np.load(args.weights)
        else:
            weights = np.random.randn(10, 2)
        network = perceptron((2, 10), weights)
    else:
        if args.weights:
            weights = np.load(args.weights)
        else:
            weights = [np.random.randn(10, 4), np.random.randn(4, 2)]
        network = multi_layer_network((2, 4, 10), weights)
    if args.command == 'train':
        print('-------------------TRAINING---------------------')
        train(network, TRAINFILE, CLASSES, EPOCHS, RATE)
    else:
        print('---------------INFERENCE RESULTS----------------')
        test(network, TESTFILE, CLASSES)


def train(network, trainfile, classes, epochs, rate):
    """Train the perceptron network for the project2 problem
    """
    (inputs, outputs) = read_data(trainfile, classes)
    inputs = normalize(inputs, [90, 180])
    for i in range(epochs):
        # if i > 100:
        #     drate = 0.5*rate
        # else:
        #     drate = rate
        (avgloss, weights) = network.train(rate, inputs, outputs)
        print('EPOCH %d: avg. loss = %.9f' % (i, avgloss))
        if i+1 > 0 and (i+1) % 50 == 0:
            with open(os.path.join(WEIGHTSDIR, 'weights_%d.npy' % (i+1)), 'wb') as fi:
                np.save(fi, weights)


def test(network, testfile, classes):
    """Test the perceptron network for the project2 problem
    """
    (inputs, outputs) = read_data(testfile, classes)
    counters = [InferenceCounter(clas, len(inputs)) for clas in classes]
    for index, X in enumerate(inputs):
        inferred = network.infer(X)
        expected = outputs[index]
        for uid, counter in enumerate(counters):
            infer = 1.0 if inferred[uid] > 0.49 else 0.0
            if infer == expected[uid]:
                counter.correct()
                if infer == 1.0:
                    counter.true_positive()
                else:
                    counter.true_negative()
            else:
                if infer == 1.0:
                    counter.false_positive()
                else:
                    counter.false_negative()
    for counter in counters:
        metrics(counter)


def perceptron(size, weights):
    """Creates a perceptron network
    """
    inputLayer = Layer(LayerType.INPUT, size[0], activation=None)
    outputLayer = Layer(LayerType.OUTPUT, size[1], activation=Neuron.HEAVISIDE)
    pp.pprint(weights)
    inputLayer.connect(outputLayer, weights)
    return Ann({'input': inputLayer,
                'hidden': [],
                'output': outputLayer})


def multi_layer_network(size, weights):
    """Creates a 3 layer network (1 hidden)
    """
    inputLayer = Layer(LayerType.INPUT, size[0], activation=None)
    hiddenLayer = Layer(LayerType.HIDDEN, size[1], activation=Neuron.LOGISTIC)
    outputLayer = Layer(LayerType.OUTPUT, size[2], activation=Neuron.LOGISTIC)
    inputLayer.connect(hiddenLayer, weights[1])
    hiddenLayer.connect(outputLayer, weights[0])
    return Ann({'input': inputLayer,
                'hidden': [hiddenLayer],
                'output': outputLayer})


def deep_network(size, weights):
    """Creates a 5 layer network (3 hidden)
    """
    inputLayer = Layer(LayerType.INPUT, size[0], activation=None)
    hiddenLayer1 = Layer(LayerType.HIDDEN, size[1], activation=Neuron.LOGISTIC)
    hiddenLayer2 = Layer(LayerType.HIDDEN, size[2], activation=Neuron.LOGISTIC)
    hiddenLayer3 = Layer(LayerType.HIDDEN, size[3], activation=Neuron.LOGISTIC)
    outputLayer = Layer(LayerType.OUTPUT, size[4], activation=Neuron.LOGISTIC)
    inputLayer.connect(hiddenLayer1, weights[3])
    hiddenLayer1.connect(hiddenLayer2, weights[2])
    hiddenLayer2.connect(hiddenLayer3, weights[1])
    hiddenLayer3.connect(outputLayer, weights[0])
    return Ann({'input': inputLayer,
                'hidden': [hiddenLayer1, hiddenLayer2, hiddenLayer3],
                'output': outputLayer})


def normalize(inputs, maxima):
    """Normalizes input to be between 0 and 1
    """
    return [[x/maxima[index] for index, x in enumerate(X)] for X in inputs]


def read_data(filename, classes):
    """Read a data file for training or testing.
    """
    with open(filename, 'r') as fi:
        reader = csv.reader(fi, delimiter='\t')
        inputs = []
        outputs = []
        for row in reader:
            inputs.append([float(row[0]), float(row[1])])
            correct = [0] * len(classes)
            correct[classes.index(row[2])] = 1.0
            outputs.append(correct)
        return (inputs, outputs)


class InferenceCounter():
    """Implements various counters used for measuring network performance.
    """
    def __init__(self, description, num_inputs):
        self.description = description
        self.num_inputs = num_inputs
        self.corrects = 0
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def correct(self):
        """Increments the corrects counter
        """
        self.corrects += 1
        return self.corrects

    def true_positive(self):
        """Increments the true_positives counter
        """
        self.true_positives += 1
        return self.true_positives

    def true_negative(self):
        """Increments the true_negatives counter
        """
        self.true_negatives += 1
        return self.true_negatives

    def false_positive(self):
        """Increments the false_positive counter
        """
        self.false_positives += 1
        return self.false_positives

    def false_negative(self):
        """Increments the false_negative counter
        """
        self.false_negatives += 1
        return self.false_negatives

    def percent_correct(self):
        return self.corrects/self.num_inputs*100.0

    def percent_true_positive(self):
        return self.true_positives/self.num_inputs*100.0

    def percent_true_negative(self):
        return self.true_negatives/self.num_inputs*100.0

    def percent_false_positive(self):
        return self.false_positives/self.num_inputs*100.0

    def percent_false_negative(self):
        return self.false_negatives/self.num_inputs*100.0


def metrics(counter):
    """Calculates percentage correct, true positives, true negatives, false positives, false negatives
    """
    print('Neuron: %s' % (counter.description))
    print('    Correct: %.2f%%' % (counter.percent_correct()))
    print('    True Positives: %.2f%%' % counter.percent_true_positive())
    print('    True Negatives: %.2f%%' % counter.percent_true_negative())
    print('    False Positives: %.2f%%' % counter.percent_false_positive())
    print('    False Negatives: %.2f%%' % counter.percent_false_negative())


if __name__ == "__main__":
    main()
