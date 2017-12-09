"""
.. module:: Neuron
.. moduleauthor:: Cody Balos <cjbalos@gmail.com>
"""

import numpy as np
import pprint as pp
from activation import heaviside, logistic, relu
from edge import Edge
from layertype import LayerType

class Neuron():
    """Class which implements an artificial neuron in a neuron network.
    """

    @staticmethod
    def HEAVISIDE(x, derivative=False):
        return heaviside(0, x, derivative)

    @staticmethod
    def LOGISTIC(x, derivative=False):
        return logistic(0, x, derivative)

    @staticmethod
    def RELU(x, derivative=False):
        return relu(0, x, derivative)


    #: static member used to assign neuron IDs
    count = 0

    def __init__(self, layerType, activation=None):
        self.uid = Neuron.count
        self.type = layerType
        self.edges = {
            'inputs': [],
            'outputs': []
        }
        self.activation = activation if not None else Neuron.HEAVISIDE
        self.dotproduct = 0
        self.error = 0
        self.output = 0
        Neuron.count += 1

    def activate(self, x=None):
        """Activates the neuron and returns the output.

        :param x: the input into the neuron if it is not from another neuron (i.e. it is an input layer neuron)
        """
        # if x is not none, then this is an input neuron
        if x is not None:
            self.output = x
            return self.output
        self.dotproduct = sum([edge.weight*edge.frm.output for edge in self.edges['inputs']])
        self.output = self.activation(self.dotproduct)
        # print('Neuron %d was activated, and the output is %f' % (self.uid, self.output))
        return self.output

    def connect(self, to, weight):
        """Connects a neuron to another neuron

        :param to: :class:`Neuron` to connect to
        """
        edge = Edge(self, to, weight)
        self.edges['outputs'].append(edge)
        to.edges['inputs'].append(edge)

    def learn(self, rate, correct):
        """Uses perceptron learning model to learn.
        """
        weights = []
        losses = []
        for edge in self.edges['inputs']:
            edge.weight = edge.weight + rate*(correct-self.output)*edge.frm.output
            # print('%.4f = %.4f + %.4f*(%.4f - %.4f)*%.4f' % (edge.weight, old_weight, rate, correct, self.output, edge.frm.output))
            # print('updated weight from neuron %d to neuron %d to %.4f from %.4f' % (edge.frm.uid, self.uid, edge.weight, old_weight))
            losses.append(float(correct == self.output))
            weights.append(edge.weight)
        return (np.sum(losses), weights)

    def propagate(self, rate, correct):
        """Uses back propagation for learning.
        """
        if self.type == LayerType.OUTPUT:
            self.error = self.activation(self.dotproduct, True)*(correct-self.output)
        else:
            self.error = sum([edge.weight*edge.to.error for edge in self.edges['outputs']])
            self.error = self.activation(self.dotproduct, True)*self.error
        weights = []
        losses = []
        for edge in self.edges['inputs']:
            edge.weight = edge.weight + rate*self.output*self.error
            losses.append((correct-self.output)**2)
            weights.append(edge.weight)
        # print(weights)
        return (np.sum(losses), weights)
