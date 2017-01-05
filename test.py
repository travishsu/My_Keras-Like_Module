from mydl import Dense, Activation, SquareError, Sequential 
import numpy as np

ij = np.random.randn(32,1)
n2 = Dense(1, 5)
n1 = Dense(5, 10)
n = Dense(10, 32)

# Try node's function
n1.forward(n.forward(ij))
L = n1.forward(n.forward(ij))
n.backward(n1.backward(L))

# Try model's function
model = Sequential()
model.add(n)
model.add(Activation('sigmoid'))
model.add(n1)
model.add(Activation('sigmoid'))
model.add(n2) # NN structure: act ([1 x 5] (act[5 x 10] (act([10 x 32] [32 x 1])))) = [1 x 1]
model.compile(SquareError())
for i in range(10000):
    model.fit_single(ij, 5, .0001) # the target is 5, learning rate is .0001.
