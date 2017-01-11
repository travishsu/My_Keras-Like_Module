import numpy as np

class Neuron:
    
    def __init__(self):
        pass
    
    def update(self, stepsize):
        pass

class Add(Neuron):
    
    def forward(self, a, b):
        self.a = a
        self.b = b
        return a + b
    
    def backward(self, loss):
        self.da = loss
        self.db = loss
        return {'da': self.da, 'db':self.db}

class Multiply(Neuron):
    
    def forward(self, a, b):
        self.a = a
        self.b = b
        return a * b
    
    def backward(self, loss):
        self.da= loss * self.b
        self.db= loss * self.a
        return {'da': self.da, 'db':self.db}

class Sum(Neuron):
    
    def forward(self, numlist):
        self.numlist = numlist
        return np.sum(numlist)
    
    def backward(self, loss):
        self.dnumlist = loss * np.ones(len(self.numlist)) 

class Product(Neuron):
    
    def forward(self, numlist):
        self.numlist = numlist
        return np.prod(numlist)
    
    def backward(self, loss):
        self.dnumlist = loss * np.prod(self.numlist) / self.numlist

class Sine(Neuron):

    def forward(self, inputs):
        self.prev_in = inputs
        return np.sin(inputs)

    def backward(self, loss):
        return loss * np.cos(self.prev_in)

class Cosine(Neuron):

    def forward(self, inputs):
        self.prev_in = inputs
        return np.cos(inputs)

    def backward(self, loss):
        return - loss * np.sin(self.prev_in)
