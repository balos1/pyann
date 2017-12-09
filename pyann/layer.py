"""
.. module:: Layer
.. moduleauthor:: Cody Balos <cjbalos@gmail.com>
"""

from enum import Enum
from neuron import Neuron
from layertype import LayerType


class Layer():
    """Class that implements a layer in a neural network.
    """

    def __init__(self, layerType, size=0, activation=None):
        self.type = layerType
        self.size = size
        self.neurons = [Neuron(self.type, activation) for _ in range(size)]

    def activate(self, X=None):
        """Activates the neurons in the layer and returns the outputs.

        :param x: the input vector into the layer if it is not from another layer (i.e. it is an input layer)
        """
        # if X is not None, then this is an input layer
        if X is not None:
            if len(X) != self.size:
                raise ValueError('Input vector size and layer size must be the same')
            return [neuron.activate(X[uid]) for uid, neuron in enumerate(self.neurons)]
        else:
            return [neuron.activate() for neuron in self.neurons]

    def connect(self, toLayer, weights):
        """Connects a layer to another layer

        :param to: :class:`Layer` to connect to
        """
        if len(weights) != len(toLayer.neurons):
            raise ValueError('NxM w/eights matrix required for connecting size N layer to size M layer')
        # connect each neuron in from layer to all neurons in to layer
        for i, to in enumerate(toLayer.neurons):
            if len(weights[i]) != len(self.neurons):
                raise ValueError('NxM weights matrix required for connecting size N layer to size M layer')
            for j, frm in enumerate(self.neurons):
                frm.connect(to, weights[i][j])

    def learn(self, rate, correct):
        """Uses perceptron learning model to learn.
        """
        return [neuron.learn(rate, correct[index]) for index, neuron in enumerate(self.neurons)]

    def propagate(self, rate, correct):
        """Uses back propagation for learning.
        """
        return [neuron.propagate(rate, correct[index]) for index, neuron in enumerate(self.neurons)]
