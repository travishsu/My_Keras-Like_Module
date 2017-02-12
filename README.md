# keras-like_NN
Build stacks of neural network like Keras module

## Quickstart

 - Import
        
        from mydl import *
 
 - Build a feed-forward neural network
        
        model = Sequential()
 
 - Add layer to model
        
        model.add(new_layer)

 - We can evaluate some sample and see how the results behaviour now

        model.evaluate(samples)

 - Compile with a kind of loss function layer

        model.compile(loss_layer)

 - Train with bunch of training data

        model.fit(X, y, stepsize=1e-3)

## Neurons

 - Add
 - Multiply
 - Sum
 - Product
 - Sine
 - Cosine
 - Dense
 - Activation
 - Dropout
 - SquareError
 - CrossEntropy
 
## Models

 - Sequential
 - 
 
## Basic Components A Neuron Must Have

 - Forward propagation
 - Backward propagation
