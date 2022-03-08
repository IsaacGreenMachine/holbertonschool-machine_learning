#!/usr/bin/env python3
"""module for deep neural network class"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork():
    """Defines a deep neural network performing binary classification."""
    def __init__(self, nx, layers):
        """implements a deep nerual network with many layers"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        elif type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        else:
            self.__L = len(layers)
            self.__cache = {}
            self.__weights = {}
            prev = nx
            for i in range(len(layers)):
                if type(layers[i]) is not int or layers[i] < 1:
                    raise TypeError(
                        'layers must be a list of positive integers')
                w = np.random.randn(layers[i], prev) * np.sqrt(2/prev)
                prev = layers[i]
                self.__weights['W{}'.format(i + 1)] = w
                dim = len(self.weights['W{}'.format(i + 1)])
                self.__weights['b{}'.format(i + 1)] = np.zeros((dim, 1))

    @property
    def L(self):
        """getter for L"""
        return self.__L

    @property
    def cache(self):
        """getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """single run of forward propagation (multiclass) for deep nn"""
        self.__cache.update({"A0": X})
        for layer in range(1, self.L+1):
            w = self.weights["W{}".format(layer)]
            a = self.cache["A{}".format(layer-1)]
            b = self.weights["b{}".format(layer)]
            z = np.matmul(w, a) + b
            if layer == self.L:
                t = np.exp(z)
                self.__cache["A{}".format(layer)] = t/np.sum(t, axis=0)
            else:
                self.__cache["A{}".format(layer)] = 1 / (1 + np.exp(-z))
        return self.cache["A{}".format(self.L)], self.cache

    def cost(self, Y, A):
        """returns cost of deep nn (multiclass)"""
        m = Y.shape[1]
        return np.sum(-Y * np.log(A)) / m

    def evaluate(self, X, Y):
        """evaluates nn using forward prop and cost funcs (multiclass)"""
        a = self.forward_prop(X)[0]
        a_idxs = np.argmax(self.cache["A3"], axis=0)
        a_idxs.reshape(a_idxs.size, 1)
        count = np.arange(a.shape[1])
        count.reshape(count.size, 1)
        hard_max = np.zeros_like(a)
        hard_max[a_idxs, count] = 1
        return (hard_max.astype(int),
                self.cost(Y, self.cache["A{}".format(self.L)]))

    def gradient_descent(self, Y, cache, alpha=0.05):
        """implements back propagation for deep nn"""
        m = Y.shape[1]
        for layer in range(self.L, 0, -1):
            if layer == self.L:
                dz = (cache["A{}".format(layer)] - Y)
            else:
                a = cache["A{}".format(layer)]
                dz = da * a * (1 - a)
            dw = np.matmul(dz, cache["A{}".format(layer-1)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            da = np.matmul(self.weights["W{}".format(layer)].T, dz)
            new_w = self.weights["W{}".format(layer)] - (alpha * dw)
            new_b = self.weights["b{}".format(layer)] - (alpha * db)
            self.__weights["W{}".format(layer)] = new_w
            self.__weights["b{}".format(layer)] = new_b

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """trains deep nn for x iterations"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
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
            self.gradient_descent(Y, self.cache, alpha)
            if (verbose or graph) and i % step == 0:
                currCost = self.cost(Y, self.cache["A{}".format(self.L)])
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

    def save(self, filename):
        """serializes instance to file filename using pickle"""
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load(filename):
        """deserializes instance from file filename using pickle"""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
