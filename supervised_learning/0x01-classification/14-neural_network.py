"""module for neural network class"""
import numpy as np


class NeuralNetwork:
    """implements a neural network with a single layer"""
    def __init__(self, nx, nodes):
        """sets attributes for neural network"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros(shape=(nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """getter for w1"""
        return self.__W1

    @property
    def b1(self):
        """getter for b1"""
        return self.__b1

    @property
    def A1(self):
        """getter for a1"""
        return self.__A1

    @property
    def W2(self):
        """getter for w2"""
        return self.__W2

    @property
    def b2(self):
        """getter for b2"""
        return self.__b2

    @property
    def A2(self):
        """getter for a2"""
        return self.__A2

    def forward_prop(self, X):
        """forward propagation for neural network"""
        z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.matmul(self.W2, self.__A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.A1, self.A2

    def cost(self, Y, A):
        """returns cost of neural network"""
        L = -((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        return (1/A.shape[1]) * np.sum(L)

    def evaluate(self, X, Y):
        """returns info about nn using forward prop and cost funcs"""
        return (self.forward_prop(X)[1].round().astype(int),
                self.cost(Y, self.A2))

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """implements back propagation for neural network"""
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T)/m
        db2 = np.sum(dz2, axis=1, keepdims=True)/m
        z1 = np.matmul(self.W1, X) + self.b1
        dz1 = np.matmul(self.W2.T, dz2) * (A1 * (1 - A1))
        dw1 = np.matmul(dz1, X.T)/m
        db1 = np.sum(dz1, axis=1, keepdims=True)/m
        self.__W2 = self.__W2 - (alpha * dw2)
        self.__b2 = self.__b2 - (alpha * db2)
        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b1 = self.__b1 - (alpha * db1)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """trains neural network x iterations"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for _ in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)
        return self.evaluate(X, Y)
