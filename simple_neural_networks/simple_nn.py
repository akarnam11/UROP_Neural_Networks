from dataclasses import dataclass # for neural network construction
import pickle # to import MNIST data
import gzip # to import MNIST data
import random # to initialize weights and biases
import numpy as np # for all needed math
from PIL import Image, ImageOps # for image file processing
from time import time # for performance measurement


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def cost_derivative(output_activations, y):
    return (output_activations - y)

@dataclass
class Network:
    num_layers: int
    biases: list
    weights: list

def init_network(layers):

    return Network(
        len(layers),

        # input layer doesn't have biases
        [np.random.randn(y, 1) for y in layers[1:]],

        # there are no (weighted) conections into input layer or out of the output layer
        [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
    )

def feedforward(nn, a):
    for b, w in zip(nn.biases, nn.weights):
        a = sigmoid(np.dot(w, a) + b)
    return a

def evaluate(nn, test_data):
    test_results = [(np.argmax(feedforward(nn, x)), y) for (x, y) in test_data]
    
    return sum(int(x==y) for (x, y) in test_results)

def learn(nn, training_data, epochs, mini_batch_size, learning_rate, test_data = None):
    n = len(training_data)

    for j in range(epochs):
        random.shuffle(training_data) # that's where "stochastic" comes from

        mini_batches = [
            training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)
        ]
        
        for mini_batch in mini_batches:
            stochastic_gradient_descent(nn, mini_batch, learning_rate) # that's where learning really happes

        if test_data:
            print('Epoch {0}: accuracy {1}%'.format(f'{j + 1:2}', 100.0 * evaluate(nn, test_data) / len(test_data)))
        else:
            print('Epoch {0} complete.'.format(f'{j + 1:2}'))

def stochastic_gradient_descent(nn, mini_batch, eta):
    # "nabla" is the gradient symbol
    nabla_b = [np.zeros(b.shape) for b in nn.biases]
    nabla_w = [np.zeros(w.shape) for w in nn.weights]

    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = backprop(nn, x, y) # compute the gradient
        
        # note that here we call the return values 'delta_nabla', while in the
        # backprop function we call them 'nabla'

        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

    nn.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(nn.weights, nabla_w)]
    nn.biases  = [b - (eta / len(mini_batch)) * nb for b, nb in zip(nn.biases, nabla_b)]

def backprop(nn, x, y):
    nabla_b = [np.zeros(b.shape) for b in nn.biases]
    nabla_w = [np.zeros(w.shape) for w in nn.weights]

    # feedforward
    activation = x # first layer activation is just its input
    activations = [x] # list to store all activations, layer by layer
    zs = [] # list to store all z vectors, layer by layer

    for b, w in zip(nn.biases, nn.weights):
        z = np.dot(w, activation) + b # calculate z for current layer
        zs.append(z) # store
        activation = sigmoid(z) # layer output
        activations.append(activation) # store

    # backward pass

    # 1. starting from the output layer
    delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) 
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    # 2. continue bac to the input layer (i is the layer index)
    for i in range(2, nn.num_layers): # starting from the next-to-last layer
        z = zs[-i]
        sp = sigmoid_prime(z)
        delta = np.dot(nn.weights[-i + 1].transpose(), delta) * sp
        
        nabla_b[-i] = delta
        nabla_w[-i] = np.dot(delta, activations[-i - 1].transpose())
        
    return (nabla_b, nabla_w)

def load_data():
    f = gzip.open('mnist.pk1.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="bytes")
    f.close()

    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [one_hot_encode(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    
    return (list(training_data), list(validation_data), list(test_data))

def one_hot_encode(j):
    e = np.zeros((10, 1))
    e[j] = 1.0

    return e

def print_shape(name, data):
    print('Shape of {0}: {1}'.format(name, data.shape))

training_data, validation_data, test_data = load_data_wrapper() # load data

nn = init_network([784, 30, 10])

for l in range(0, nn.num_layers - 1):
    print('\nNetwork layer {0}'.format(l + 2)) # disregard the input layer
    print_shape('weights', nn.weights[l])
    print_shape('biases', nn.biases[l])
    
# hyper parameters
epochs = 15
mini_batch_size = 10
learning_rate = 3.0
    
print('\nLearning process started...\n')

time_start = time()

learn(nn, training_data, epochs, mini_batch_size, learning_rate, test_data)

time_end = time()

time_elapsed = time_end - time_start

print('\nLearning process complete in {0} seconds ({1} seconds per epoch)!\n'.format(f'{time_elapsed:.0f}', f'{time_elapsed / epochs:.1f}'))

print('Validation (with yet unseen data): accuracy {0}%'.format(100.0 * evaluate(nn, validation_data) / len(validation_data)))

def load_image(file_name):
    digit = Image.open(file_name)

    # invert, so that background is black (zeros)
    digit = ImageOps.invert(digit)

    pixels = digit.load()

    return np.array(digit).reshape((784, 1)) / 255

def recognize_image(path, file):
    x = load_image(path.format(file))

    y = feedforward(nn, x)

    bitmap = x.reshape((28, 28))

    file_num = int(file)
    result = y.argmax()

    if file_num == result:
        ev = 'correctly'
    else:
        ev = 'incorrectly'
    print(file, 'was', ev, 'recognized as', result)

print('Non-MNIST digits:\n')

for file in range(0, 10):
    recognize_image('./non-MNIST-digits/{0}.png', file)