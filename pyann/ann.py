"""
.. module:: ann
.. moduleauthor:: Cody Balos <cjbalos@gmail.com>
"""

import numpy as np
import pprint as pp

class Ann():
    """Implements an artificial neural network
    """

    def __init__(self, arch):
        self.input_layer = arch['input']
        self.hidden_layers = arch['hidden']
        self.output_layer = arch['output']

    def train(self, rate, inputs, outputs):
        """Trains the ANN.
        """
        losses = []
        weights = []
        for index, X in enumerate(inputs):
            self.activate(X)
            if len(self.hidden_layers):
                results = self.output_layer.propagate(rate, outputs[index])
                l,w = map(list, zip(*results))
                losses.append(np.sum(l))
                weights.append(w)
                for layer in self.hidden_layers:
                    results = layer.propagate(rate, outputs[index])
                    l, w = map(list, zip(*results))
                    losses.append(np.sum(l))
                    weights.append(w)
            else:
                results = self.output_layer.learn(rate, outputs[index])
                l, w = map(list, zip(*results))
                losses.append(np.sum(l))
                weights = w
        avgloss = np.sum(losses)/len(inputs)
        return (avgloss, weights)

    def infer(self, X):
        """Uses ANN to infer results.
        """
        return self.activate(X)

    def activate(self, X):
        """Activates network neurons.
        """
        self.input_layer.activate(X)
        for layer in self.hidden_layers:
            layer.activate()
        return self.output_layer.activate()
