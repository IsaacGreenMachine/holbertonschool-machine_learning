"""Module for neuron class"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """neuron with sigmoid activation"""
    def __init__(self, nx):
        """sets values for neuron"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(0, 1, size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """get W value"""
        return self.__W

    @property
    def b(self):
        """get b value"""
        return self.__b

    @property
    def A(self):
        """get A value"""
        return self.__A

    def forward_prop(self, X):
        """implements single run of forward propagation"""
        z = np.matmul(self.W, X) + self.b
        self.__A = 1 / (1 + np.exp(-z))
        return self.A

    def cost(self, Y, A):
        """returns cost function of network"""
        L = -((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        return (1/A.shape[1]) * np.sum(L)

    def evaluate(self, X, Y):
        """run of forward propagation and evaluates cost func"""
        return self.forward_prop(X).round().astype(int), self.cost(Y, self.A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """implements backpropagation with grad. desc."""
        m = Y.shape[1]
        dz = A - Y
        dw = np.dot(X, dz.T) / m
        db = np.sum(dz) / m
        self.__W = self.__W - (alpha * dw.T)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """trains a neuron for x iterations"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if graph or verbose:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if graph:
            graphx = []
            graphy = []
            # ISSUE WITH 0th ITERATION.
            # HOW TO CALC COST FOR 0 WITHOUT FORWARD PROP SETTING A?
        for i in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A, alpha)
            if (verbose or graph) and i % step == 0:
                currCost = self.cost(Y, self.A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, currCost))
                if graph:
                    graphx.append(i)
                    graphy.append(currCost)
        if graph:
            plt.plot(graphx, graphy)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)
