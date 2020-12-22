 #!/usr/bin/env python3
"""Create a Neuron """
import numpy as np


class Neuron:
    """defines a single neuron performing binary classification"""
    def __init__(self, nx):
        """Class Initialization """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.W = np.random.randn(1, self.nx)
        self.A = 0
        self.b = 0
 