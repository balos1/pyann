"""
.. module:: Edge
.. moduleauthor:: Cody Balos <cjbalos@gmail.com>
"""

class Edge():
    """Class which implements an edge in a neural network.
    """

    count = 0

    def __init__(self, frm, to, weight):
        self.uid = Edge.count
        # The from :class:`Neuron`
        self.frm = frm
        # The to :class:`Neuron`
        self.to = to
        # The current weight for this edge
        self.weight = weight
        Edge.count += 1
