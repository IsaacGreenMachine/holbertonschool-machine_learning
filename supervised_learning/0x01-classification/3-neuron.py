import numpy as np
import numpy
class Neuron:
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(0, 1, size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        z = numpy.matmul(self.W, X) + self.b
        self.__A = self.sigmoid_act(z)
        return self.A

    def sigmoid_act(self, x):
        return (1 / (1 + np.exp(x)) )

    def cost(self, Y, A):
        L = -(Y * np.log(A)) - (1 - Y ) * np.log(1.0000001 - A)
        return (1/A.shape[1]) * np.sum(L)
