import numpy as np

class Neuron:
    
    def __init__(self):
        pass
    
    def update(self, stepsize):
        pass

class Dense(Neuron):
    
    def __init__(self, output_dim, input_dim):
        self.W = 0.01 * np.random.randn(output_dim, input_dim)
        self.bias = 0.01 * np.random.randn(output_dim, 1)
    
    def forward(self, inputs):
        self.prev_in = np.array(inputs).astype(float).reshape(-1, 1)
        return  self.W.dot(inputs) + self.bias
    
    def backward(self, loss):
        self.dW = loss.dot(self.prev_in.T)
        self.dbias = loss
        return loss.T.dot(self.W).T
    
    def update(self, stepsize):
        self.W -= (stepsize/np.linalg.norm(self.dW)) * self.dW
        self.bias -= (stepsize/np.linalg.norm(self.dbias)) * self.dbias

class Dropout(Neuron):
    
    def __init__(self, p, input_dim):
        self.p   = p
        self.input_dim = input_dim
        self.idx = np.random.binomial(1, 1-p, size=input_dim).reshape(-1, 1)
        self.proportion = sum(self.idx) / float(self.input_dim)
    
    def forward(self, inputs):
        self.prev_in = inputs
        return self.idx * inputs / self.proportion
    
    def backward(self, loss):
        self.didx = self.idx / self.proportion
        return loss * self.didx
    
    def update(self, stepsize):
        self.idx = np.random.binomial(1, 1-self.p, size=self.input_dim).reshape(-1, 1)
        self.proportion = sum(self.idx) / float(self.input_dim)

class Activation(Neuron):
    
    def __init__(self, atype):
        self.atype = atype
    
    def forward(self, inputs):
        if self.atype == 'sigmoid':
            firing_rate = 1.0 / (1.0 + np.exp(-inputs))
            self.firing_rate = firing_rate
            return firing_rate
        if self.atype == 'tanh':
            self.tanh_in = np.tanh(inputs)
            return self.tanh_in
        if self.atype == 'softmax':
            self.expon = np.exp(inputs)
            self.sumexp = np.sum( self.expon )
            return self.expon / self.sumexp
        if self.atype == 'relu':
            self.relu = np.maximum(0, inputs)
            return self.relu
        
    def backward(self, loss):
        if self.atype == 'sigmoid':
            return loss * (1 - self.firing_rate) * self.firing_rate
        if self.atype == 'tanh':
            return loss * (1 - self.tanh_in**2)
        if self.atype == 'softmax':
            return loss * self.expon * (self.sumexp - self.expon) / self.sumexp**2
        if self.atype == ' relu':
            return np.array( [(1 if s>0 else 0) for s in self.relu] ).reshape(self.relu.shape)

class SquareError(Neuron):
    
    def forward(self, inputs, targets):
        self.targets = np.reshape(targets, (-1, 1))
        self.prev_in = inputs.reshape(-1, 1)
        return (self.prev_in - self.targets).T.dot(self.prev_in - self.targets)
    
    def backward(self, loss):
        return 2*(self.prev_in - self.targets)

class CrossEntropy(Neuron):
    '''
    CE now has the issue that there is singularity when prev_in very small.
    '''
    def forward(self, inputs, targets):
        self.targets = np.reshape(targets, (-1, 1))
        self.prev_in = inputs.reshape(-1, 1)
        return - np.sum( self.targets * np.log(self.prev_in) )
    
    def backward(self, loss):
        summation = 0
        for i in xrange(self.targets.shape[0]):
            if self.targets[i]>0:
                summation += self.targets[i] * (1. / self.prev_in[i])
        return - summation
        #return - self.targets.T.dot( 1. / self.prev_in )

class Sequential:
    
    def __init__(self):
        self.nodes = []
    
    def add(self, node):
        self.nodes.append(node)
    
    def compile(self, evaluate_node):
        self.lossfun = evaluate_node
    
    def evaluate_single(self, sample):
        L = sample
        for node in self.nodes:
            L = node.forward(L)
        return L
    
    def fit_single(self, sample, targets, stepsize=1e-5):
        L = sample
        for node in self.nodes:
            L = node.forward(L)
        L = self.lossfun.forward(L, targets)
        
        dfun = L
        dfun = self.lossfun.backward(dfun)
        self.nodes.reverse()
        for node in self.nodes:
            dfun = node.backward(dfun)
            node.update(stepsize)
        self.nodes.reverse()

    def fit(self, X, y, stepsize=1e-5):
        assert X.shape[0] == y.shape[0]
        n_sample = X.shape[0]
        L_sum = 0
        for i in xrange(n_sample):
            L = X[i]
            L = L.reshape(-1, 1)
            targets = y[i]

            for node in self.nodes:
                L = node.forward(L)
            L = self.lossfun.forward(L, targets) / n_sample
            L_sum += L

            dfun = L
            dfun = self.lossfun.backward(dfun)
            self.nodes.reverse()
            for node in self.nodes:
                dfun = node.backward(dfun)
                node.update(stepsize)
            self.nodes.reverse()

        print("Loss: {} in {} samples.".format(L_sum, n_sample))
            

