from random import seed
from random import random
from math import  exp

def init_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs+1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer= [{'weights': [random() for i in range(n_hidden+1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

def transfer(activation):
    return 1.0/(1.0 + exp(-activation))

def forword_propagate(inputs, network):
    for layer in network:
        new_inputs = []
        for neuron in  layer:
            activation = activate(neuron['weights'],inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def transfer_derivative(output):
    return output*(1.0 - output)


network = init_network(2,6,5)
outputs = forword_propagate([1,2],network)
for layer in network:
    print(layer)
print(outputs)

