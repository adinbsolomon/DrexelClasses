
import abc
from copy import deepcopy
import logging
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from MyPython import CS383, MyNumpy, MyInspect

# Logging
#logging.basicConfig(level=logging.DEBUG)

# Constants 
DEFAULT_SEED = 0
DEFAULT_INITIALIZING_RANGE = 10 ** -4
EPSILON = 10 ** -10

# Layer Base Class
DEFAULT_BACKPROP_UPDATE = True
class Layer(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self.Y = None
        self.Y_hat = None
    def CheckForYhat(self): 
        if self.Y_hat is None: raise Exception("Y_hat must be initialized!")
    # X can one row or multiple rows
    @abc.abstractmethod
    def forwardPropagate(self, X): pass
    # gradient with respect to the input
    @abc.abstractmethod
    def gradient(self): self.CheckForYhat()
    # backpropagation for all components
    @abc.abstractmethod
    def backwardsPropagate(self, gradient, learning_rate, update=DEFAULT_BACKPROP_UPDATE): pass
    # reinitialize mostly for connected layers
    @abc.abstractmethod
    def reinit(self): pass
    @staticmethod
    def is_layer(l):
        return isinstance(l, Layer)

# Input Layers
class InputLayer(Layer):
    def __init__(self, X_train=None) -> None:
        if X_train is not None:
            _, self.mean, self.std = \
                CS383.standardize(X_train, return_stats=True)
        else: self.mean, self.std = 0, 1
    def forwardPropagate(self, X):
        return CS383.standardize(X, self.mean, self.std)
    def gradient(self): return super().gradient()
    def backwardsPropagate(self, gradient, learning_rate, update=DEFAULT_BACKPROP_UPDATE):
        return super().backwardsPropagate(gradient, learning_rate)
    def reinit(self): super().reinit()
 
# Connection Layers
class ConnectionLayer(Layer, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()
    @abc.abstractmethod
    def forwardPropagate(self, X): super().forwardPropagate(None)
    @abc.abstractmethod
    def gradient(self): super().gradient()
    @abc.abstractmethod
    def backwardsPropagate(self, gradient, learning_rate, update=DEFAULT_BACKPROP_UPDATE): super().backwardsPropagate(gradient, learning_rate, update=update)
    @abc.abstractmethod
    def reinit(self): super().reinit()
    @staticmethod
    def is_connection_layer(l):
        return isinstance(l, ConnectionLayer)
class FullyConnectedLayer(ConnectionLayer):
    def __init__(self, features_in, features_out, **kwargs) -> None:
        super().__init__()
        self.features_in, self.features_out = features_in, features_out
        self.init_range = kwargs.get('init_range', DEFAULT_INITIALIZING_RANGE)
        self.seed = kwargs.get('seed', DEFAULT_SEED)
        self.rs = np.random.RandomState(seed=self.seed)
        self.initialize_weights()
        self.initialize_biases()
        self.h = None
        self.hold = False
    def initialize_weights(self):
        # Given weight generation does +/- 0.5 * 10^-4
        #self.W = (self.rs.rand(features_in,features_out) - 0.5) * init_range
        #self.b = (self.rs.rand(1, features_out) - 0.5) * init_range
        if self.features_in == -1 or self.features_out == -1:
            self.W = None
        else:
            self.W = (self.rs.rand(self.features_in,self.features_out) * 2 - 1) * self.init_range
    def initialize_biases(self):
        if self.features_out == -1:
            self.b = None
        else:
            self.b = (self.rs.rand(1, self.features_out) * 2 - 1) * self.init_range
    def CheckForH(self):
        if self.h is None: raise Exception("h must be initialized!")
    def forwardPropagate(self, X):
        if not self.is_init(): raise Exception("Not initialized!")
        self.h = X.copy()
        #print(X.shape, self.W.shape, self.b.shape)
        self.Y_hat = (X @ self.W) + self.b
        return self.Y_hat
    def gradient(self): # with respect to input
        super().gradient()
        return self.W.T
    def gradient_weights(self):
        super().gradient()
        self.CheckForH()
        return self.h.T
    def gradient_biases(self):
        super().gradient()
        return np.ones(shape=(self.Y_hat.shape[0],1)).T
    def backwardsPropagate(self, gradient, learning_rate, update=DEFAULT_BACKPROP_UPDATE):
        #print(self.W.shape, self.b.shape)
        #print(self.gradient().shape, \
        #    self.gradient_weights().shape, \
        #    self.gradient_biases().shape)
        #print(gradient.shape)
        #update_w = self.gradient_weights() @ gradient
        #print(update_w.shape)
        #update_b = self.gradient_biases() @ gradient
        #print(update_b.shape)
        if update and not self.hold:
            dW = (learning_rate / self.W.shape[0]) * (self.gradient_weights() @ gradient)
            #print(f'{dW.shape = }')
            self.W = self.W - dW
        #print(self.gradient_biases().shape)
        if update and not self.hold:
            dB = (learning_rate / self.W.shape[0]) * (self.gradient_biases() @ gradient)
            #print(f'{dB.shape = }')
            self.b = self.b - dB
        return gradient @ self.gradient()
    def get_connection_layer(self, i):
        return [l for l in self.Layers if l is ConnectionLayer][i]
    def reinit(self):
        super().reinit()
        self.initialize_weights()
        self.initialize_biases()
    def is_init(self) -> bool:
        return self.W is not None and self.b is not None
DEFAULT_MOMENTUM_DECAY = 0.9
DEFAULT_RMSPROP_DECAY = 0.999
DEFAULT_DELTA = 10**-8
class OptimizedFullyConnectedLayer(FullyConnectedLayer):
    def __init__(self, features_in, features_out, **kwargs) -> None:
        super().__init__(features_in, features_out, **kwargs)
        self.p1 = kwargs.get('momentum_decay', DEFAULT_MOMENTUM_DECAY)
        self.p2 = kwargs.get('RMSProp_decay', DEFAULT_RMSPROP_DECAY)
        self.delta = kwargs.get('delta', DEFAULT_DELTA)
        self.epoch_num = 0
        self.s, self.r = 0, 0
    def initialize_weights(self):
        self.init_range = np.sqrt(6 / (self.features_in + self.features_out))
        super().initialize_weights()
    def initialize_biases(self):
        self.init_range = np.sqrt(6 / (self.features_in + self.features_out))
        super().initialize_biases()
    def backwardsPropagate(self, gradient, learning_rate, update=DEFAULT_BACKPROP_UPDATE):
        # compute this before making changes
        ret = gradient @ self.gradient()
        if (not update) or (self.hold): return ret
        # Update biases as normal
        dB = (self.gradient_biases() @ gradient)
        self.b = self.b - (learning_rate / self.W.shape[0]) * dB
        # Updating weights is a bit more complicated
        g = self.gradient_weights() @ gradient
        self.epoch_num += 1
        self.s = self.p1*self.s + (1-self.p1)*g
        self.r = self.p2*self.r + (1-self.p2)*g*g
        dW = (self.s / (1-self.p1**self.epoch_num)) / (self.delta + np.sqrt(self.r / (1-self.p2**self.epoch_num)))
        self.W = self.W - (learning_rate / self.W.shape[0]) * dW
        return ret
    def reinit(self):
        super().reinit()
        self.epoch_num = 0
        self.s, self.r = 0, 0

# Activation Layers
class ActivationLayer(Layer, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()
    @abc.abstractmethod
    def forwardPropagate(self, X): super().forwardPropagate(None)
    @abc.abstractmethod
    def gradient(self): super().gradient()
    def backwardsPropagate(self, gradient, learning_rate, update=DEFAULT_BACKPROP_UPDATE): return gradient * self.gradient()
    def reinit(self): super().reinit()
class LinearLayer(ActivationLayer):
    def __init__(self) -> None:
        super().__init__()
    def forwardPropagate(self, X):
        self.Y_hat = X.copy()
        return self.Y_hat
    def gradient(self):
        super().gradient()
        return np.ones(shape=self.Y_hat.shape)
class ReLuLayer(ActivationLayer):
    def __init__(self) -> None:
        super().__init__()
    def forwardPropagate(self, X):
        self.Y_hat = np.vectorize(lambda z: max(0, z))(X)
        return self.Y_hat
    def gradient(self):
        super().gradient()
        return (self.Y_hat >= 0).astype(int)
class SigmoidLayer(ActivationLayer): # AKA Logistic Layer 
    def __init__(self) -> None:
        super().__init__()
    def g(self, X):
        return 1 / (1 + np.exp(-1 * X))
    def forwardPropagate(self, X):
        self.Y_hat = self.g(X)
        if MyNumpy.is_not_prob(self.Y_hat):
            print(self.Y_hat)
            raise Exception("ruh roh")
        return self.Y_hat
    def gradient(self):
        super().gradient()
        return self.Y_hat * (1 - self.Y_hat)
class HyperbolicTangentLayer(ActivationLayer):
    def __init__(self) -> None:
        super().__init__()
    def forwardPropagate(self, X):
        p, n = np.exp(X), np.exp(-1 * X)
        self.Y_hat = (p - n) / (p + n)
        return self.Y_hat
    def gradient(self):
        super().gradient()
        raise NotImplementedError
class SoftmaxLayer(ActivationLayer):
    def __init__(self) -> None:
        super().__init__()
    def forwardPropagate(self, X):
        X = np.exp(X - np.max(X))
        s = np.sum(X, axis=1).reshape((X.shape[0],1))
        self.Y_hat = X / s
        return self.Y_hat
    def gradient(self):
        super().gradient()
        raise NotImplementedError
class SoftmaxLayer2(ActivationLayer):
    def __init__(self) -> None:
        super().__init__()
    def g(self, X):
        X = np.exp(X)
        #MyNumpy.savetxt(X, 'testyboi.csv')
        s = (MyNumpy.enforce_column(np.sum(X, axis=1)) + EPSILON)
        #print(s.T)
        return X / s
    def forwardPropagate(self, X):
        self.Y_hat = self.g(X)
        return self.Y_hat
    def gradient(self):
        super().gradient()
        return self.Y_hat * (1 - self.Y_hat)

# Output Layers
class OutputLayer(Layer, metaclass=abc.ABCMeta):
    def __init__(self, Y) -> None:
        super().__init__()
        self.Y = Y
        self.InitValid(Y)
    def InitValid(self, Y):
        self.Y = MyNumpy.enforce_column(Y)
    def forwardPropagate(self, X):
        super().forwardPropagate(None)
        self.Y_hat = deepcopy(X)
        return self.eval()
    @abc.abstractmethod
    def eval(self): self.CheckForYhat()
    @abc.abstractmethod
    def gradient(self): super().gradient()
    def backwardsPropagate(self, gradient, learning_rate, update=DEFAULT_BACKPROP_UPDATE):
        assert (gradient is None), "gradient argument for OutputLayer.backwardsPropagate() must be None"
        return self.gradient()
    def reinit(self): super().reinit()
class LeastSquaresLayer(OutputLayer):
    # Root Mean Squared Error (RMSE)
    def eval(self):
        super().eval()
        comp = (self.Y - self.Y_hat) * (self.Y - self.Y_hat)
        return np.sqrt(np.average(comp))
    def gradient(self):
        super().gradient()
        return 2 * (self.Y_hat - self.Y)
class LogLossLayer(OutputLayer):
    def InitValid(self, Y):
        pass
    def eval(self):
        super().eval()
        A = self.Y
        B = self.Y_hat
        comp = -1 * (A*np.log(B+EPSILON) + (1-A)*np.log(1-B+EPSILON))
        return np.average(comp)
    def gradient(self):
        super().gradient()
        A = (1-self.Y)/(1-self.Y_hat+EPSILON)
        B = self.Y/(self.Y_hat+EPSILON)
        return A - B
class CrossEntropyLayer(OutputLayer):
    def __init__(self, Y) -> None:
        super().__init__(Y)
        print("Warning: there's an issue with CrossEntropyLayer!!!")
    def InitValid(self, Y):
        pass
    def eval(self):
        super().eval()
        #print(self.Y.shape, self.Y_hat.shape)
        comp = -1 * self.Y * np.log(self.Y_hat + EPSILON)
        #print(comp)
        #print(comp.shape)
        ret = np.average(comp, axis=0)
        #print(ret)
        return np.average(ret)
    def gradient(self):
        super().gradient()
        return -1 * self.Y / (self.Y_hat+EPSILON)
class MeanAbsolutePercentLayer(OutputLayer):
    def eval(self):
        super().eval()
        comp = np.abs((self.Y - self.Y_hat) / self.Y)
        return np.average(comp)
    def gradient(self):
        raise NotImplementedError

# Ghost Layer for Mid-Training Processing
DEFAULT_GHOST_FUNC = lambda *args, **kwargs : None
class GhostLayer(Layer):
    def __init__(self, **funcs) -> None:
        super().__init__()
        self.funcs = funcs
    def get_func(self): return self.funcs.get(MyInspect.func_name(1), DEFAULT_GHOST_FUNC)
    def forwardPropagate(self, X):
        return self.get_func()(X)
    def gradient(self):
        # May not need to CheckForYhat
        # super().gradient()
        return self.get_func()
    def backwardsPropagate(self, gradient, learning_rate, update):
        return self.get_func()(gradient, learning_rate, update)
    def reinit(self):
        return self.get_func()()


# Layer-Organizing Structure
DEFAULT_NO_STD = False
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 10 ** -4
DEFAULT_LEARNING_FUNC = lambda epoch, lr, grad: lr
DEFAULT_MINIBATCH_FUNC = lambda n, N, RS: range(N) if n==N else RS.randint(N, size=n) - 1
GRAPH_COLORS = ["red", "blue", "green", "black"]
class Model:
    def __init__(self, **kwargs) -> None:
        # Get data
        self.Data = kwargs.get('Data', dict())
        self.validate_data()
        # Set up architecture
        self.Layers = kwargs.get('Layers',[])
        il = None
        if kwargs.get('no_std',DEFAULT_NO_STD): il = InputLayer()
        else: il = InputLayer(self.Data['X_train'])
        self.Layers = [ il ] + self.Layers
        self.OutputLayer_type = kwargs.get('OutputLayer_type', None)
        self.validate_output_layer_type()
    def validate_data(self):
        for k in ['X_train', 'Y_train', 'X_test', 'Y_test']:
            assert (k in self.Data), f"Model initialization requires {k}"
    def validate_layers(self):
        assert (self.Layers is list), f"self.Layers must be a list, not {type(self.Layers)}!"
        for i, l in enumerate(self.Layers):
            assert (l is Layer), f"All elements of self.Layers must be Layers! 'Layer' {i} is {type(l)}"
    def validate_output_layer_type(self):
        assert (issubclass(self.OutputLayer_type, OutputLayer)), "OutputLayer_type must be an of OutputLayer"
    def make_output_layer(self, X, Y):
        assert (X.shape[0] == Y.shape[0]), "X and Y must have the same number of observations!"
        return self.OutputLayer_type(Y)
    def train(self, **kwargs):
        # get args
        kwargs = self.get_training_kwargs(**kwargs)
        # start gradient descent
        loop_data = self.training_loop(**kwargs)
        for i, (label, (x, y)) in enumerate(loop_data.items()):
            plt.plot(x, y, label=label) #, color=GRAPH_COLORS[i])
        plt.legend()
        plt.show()
    def get_training_kwargs(self, **kwargs):
        kwargs['seed'] = kwargs.get('seed', DEFAULT_SEED)
        kwargs['RS'] = np.random.RandomState(seed=kwargs['seed'])
        kwargs['epochs'] = kwargs.get('epochs', DEFAULT_EPOCHS)
        kwargs['minibatch'] = kwargs.get('minibatch', self.Data['X_train'].shape[0])
        kwargs['minibatch_func'] = kwargs.get('minibatch_func', lambda: DEFAULT_MINIBATCH_FUNC(kwargs['minibatch'],self.Data['X_train'].shape[0],kwargs['RS']))
        kwargs['learning_rate'] = kwargs.get('learning_rate', DEFAULT_LEARNING_RATE)
        kwargs['learning_rate_func'] = kwargs.get('learning_func', lambda: DEFAULT_LEARNING_FUNC(None,kwargs['learning_rate'],None))
        return kwargs
    def training_loop(self, **kwargs) -> dict:
        O_train, O_test = list(), list()
        for _ in trange(kwargs['epochs']):
            # Test on whole test data
            O_test.append(self.forwardPropagate_test())
            # Train on minibatch or whole train data
            o_train, OutputLayer_batch_train = self.forwardPropagate_train(kwargs['minibatch_func'])
            O_train.append(o_train)
            # Backpropagate given training batch
            self.backwardsPropagate(OutputLayer_batch_train, kwargs['learning_rate_func']())
        x_O = list(range(kwargs['epochs']))
        return {
            'Training Objective Function': (x_O, O_train),
            'Testing Objective Function': (x_O, O_test)
        }
    def forwardPropagate(self, X, OutputLayer:OutputLayer=None):
        for layer in self.Layers:
            X = layer.forwardPropagate(X)
        if OutputLayer is not None: return OutputLayer.forwardPropagate(X)
        else: return X
    def forwardPropagate_train(self, minibatch_func) -> tuple:
        indexes = minibatch_func()
        X_batch_train = self.Data['X_train'][indexes, :]
        Y_batch_train  = self.Data['Y_train'][indexes, :]
        OutputLayer_batch_train = self.make_output_layer(X_batch_train, Y_batch_train)
        return (self.forwardPropagate(X_batch_train, OutputLayer_batch_train), OutputLayer_batch_train)
    def forwardPropagate_test(self) -> float:
        OutputLayer_test = self.make_output_layer(self.Data['X_test'], self.Data['Y_test'])
        return self.forwardPropagate(self.Data['X_test'], OutputLayer_test)
    def backwardsPropagate(self, OutputLayer:OutputLayer, learning_rate, update=DEFAULT_BACKPROP_UPDATE):
        grad = OutputLayer.backwardsPropagate(None, learning_rate, update)
        for layer in reversed(self.Layers):
            grad = layer.backwardsPropagate(grad, learning_rate, update)
    def get_connection_layer(self, i):
        if not self.has_connection_layer(): raise Exception("No ConnectionLayers here!")
        return [l for l in self.Layers if ConnectionLayer.is_connection_layer(l)][i]
    def has_connection_layer(self) -> bool:
        return True in [ConnectionLayer.is_connection_layer(l) for l in self.Layers]
    def reinit(self):
        for l in self.Layers: l.reinit()
DEFAULT_GRAPH_ACCURACY = True
DEFAULT_ACCURACY_FREQUENCY = 10
class MultiClassModel(Model):
    # ensures one-hot encoding of Ys and can evaluate accuracy
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.has_connection_layer():
            l = self.get_connection_layer(-1)
            l.features_out = self.Data['Y_train'].shape[1]
        self.reinit()
    def validate_data(self):
        super().validate_data()
        if not MultiClassModel.is_one_hot(self.Data['Y_train']):
            self.Data['Y_train'] = MultiClassModel.one_hot(self.Data['Y_train'])
        if not MultiClassModel.is_one_hot(self.Data['Y_test']):
            self.Data['Y_test']  = MultiClassModel.one_hot(self.Data['Y_test'])
        assert (self.Data['Y_train'].shape[1] == self.Data['Y_test'].shape[1]), "Train and Test sets don't have the same number of classes!"
    def get_training_kwargs(self, **kwargs):
        kwargs = super().get_training_kwargs(**kwargs)
        kwargs['graph_accuracy'] = kwargs.get('graph_accuracy', DEFAULT_GRAPH_ACCURACY)
        kwargs['accuracy_frequency'] = kwargs.get('accuracy_frequency', DEFAULT_ACCURACY_FREQUENCY)
        return kwargs
    def training_loop(self, **kwargs):
        O_train, O_test, A_train, A_test = list(), list(), list(), list()
        for e in trange(kwargs['epochs']):
            # only measure accuracy on interval
            if kwargs['graph_accuracy'] and e % kwargs['accuracy_frequency'] == 0:
                a_train, a_test = self.accuracy()
                A_test.append(a_test)
                A_train.append(a_train)
            # test on whole test data
            O_test.append(self.forwardPropagate_test())
            # Train on minibatch or whole train data
            o_train, OutputLayer_batch_train = self.forwardPropagate_train(kwargs['minibatch_func'])
            O_train.append(o_train)
            # Backpropagate given training batch
            self.backwardsPropagate(OutputLayer_batch_train, kwargs['learning_rate_func']())
        x_O = list(range(kwargs['epochs']))
        x_A = list(range(0,kwargs['epochs'],kwargs['accuracy_frequency']))
        return {
            'Training Objective Function': (x_O, O_train),
            'Testing Objective Function': (x_O, O_test) #,
            #'Training Accuracy': (x_A, A_train),
            #'Testing Accuracy': (x_A, A_test)
        }
    def accuracy(self):
        pred_train = self.forwardPropagate(self.Data['X_train'])
        a_train = np.average((np.argmax(pred_train,axis=1)==np.argmax(self.Data['Y_train'],axis=1)).astype(int))
        pred_test = self.forwardPropagate(self.Data['X_test'])
        a_test = np.average((np.argmax(pred_test,axis=1)==np.argmax(self.Data['Y_test'],axis=1)).astype(int))
        return a_train, a_test
    @staticmethod
    def one_hot(Y, classes=None):
        Y = MyNumpy.enforce_row(Y)
        classes = classes or np.unique(Y).shape[0]
        assert (np.max(Y) < classes), f"Y has some non-contiguous int class representations: {np.max(Y)}"
        assert (np.min(Y) >= 0),      f"Y has some non-contiguous int class representations: {np.min(Y)}"
        return np.eye(classes)[Y.astype(int)][0]
    @staticmethod
    def is_one_hot(Y, classes=0):
        if classes: assert (Y.shape[1] == classes), f"Y is of shape {Y.shape}"
        if len(Y.shape) != 2: return False
        check = np.sum(Y, axis=1)
        return np.sum(check)==check.shape[0] and np.min(check)==np.max(check)
DEFAULT__ = None
class GenerativeAdversarialModel:
    class GenerativeAdversarialLayer(OutputLayer):
        def eval(self):
            super().eval()
            return None
        def gradient(self):
            super().gradient()
            return -1 * np.log(self.Y_hat + EPSILON)
    def __init__(self, **kwargs) -> None:
        # Get Real Data
        # Only needs X_train (all of one class)
        if kwargs.get("no_std", False):
            self.Data, self.Mean, self.Std = kwargs.get("Data"), kwargs.get("DataMean"), kwargs.get("DataStd")
        else: 
            self.Data, self.Mean, self.Std = CS383.standardize(kwargs.get("Data"), return_stats=True)
        self.Bounds = kwargs.get("Bounds", [])
        # Get Generator (output layer is always GAN Layer)
        self.GeneratorLayers = kwargs.get("GeneratorLayers")
        if isinstance(self.GeneratorLayers[-1], OutputLayer):
            if not isinstance(self.GeneratorLayers[-1], GenerativeAdversarialModel.GenerativeAdversarialLayer):
                raise TypeError("GeneratorLayers doesn't need an output layer (it will be GA Layer)")
        # Get Descriminator (output layer is always LogLoss)
        self.DescriminatorLayers = kwargs.get("DescriminatorLayers")
        if isinstance(self.DescriminatorLayers[-1], OutputLayer):
            if not isinstance(self.DescriminatorLayers[-1], LogLossLayer):
                raise TypeError("DescriminatorLayers doesn't need an output layer (it will be LogLoss)")
    def Generator_forwardPropagate(self, X):
        for l in self.GeneratorLayers:
            X = l.forwardPropagate(X)
        return X
    def Descriminator_forwardPropagate(self, X):
        for l in self.DescriminatorLayers:
            X = l.forwardPropagate(X)
        return X
    def Generator_backwardsPropagate(self, OutputLayer, learning_rate):
        # should be GenerativeAdversarialLayer
        gradient = OutputLayer.backwardsPropagate(None, learning_rate)
        # backprop through descriminator with no update
        for l in reversed(self.DescriminatorLayers):
            gradient = l.backwardsPropagate(gradient, learning_rate, update=False)
        # backprop through generator with update
        for l in reversed(self.GeneratorLayers):
            gradient = l.backwardsPropagate(gradient, learning_rate, update=True)
    def Descriminator_backwardsPropagate(self, OutputLayer, learning_rate):
        gradient = OutputLayer.backwardsPropagate(None, learning_rate)
        for l in reversed(self.DescriminatorLayers):
            gradient = l.backwardsPropagate(gradient, learning_rate)
    def bound(self, X): 
        if self.Bounds: return MyNumpy.bound(X, *self.Bounds)
        else: return X
    def make_noise(self, RS, minibatch_size):
        return RS.normal(loc=self.Mean, scale=self.Std, size = (minibatch_size, self.Data.shape[1]))
    def get_random(self, RS, minibatch_size):
        return self.Data[RS.randint(self.Data.shape[0], size=minibatch_size), :]
    def postprocess(self, X):
        return self.bound(CS383.unstandardize(X, self.Mean, self.Std).astype(np.uint8))
    def train(self, epochs, learning_rate, minibatch_size, seed=DEFAULT_SEED):
        # Set up Output Layers
        Y_RealAndFake = np.array([([1]*minibatch_size)+([0]*minibatch_size)]).T
        Y_Fake = np.array([[0]*minibatch_size]).T
        Descriminator_OutputLayer_RealAndFake = LogLossLayer(Y = Y_RealAndFake)
        Descriminator_OutputLayer_Fake = LogLossLayer(Y = Y_Fake)
        Generator_OutputLayer = GenerativeAdversarialModel.GenerativeAdversarialLayer(Y = Y_Fake)
        # Training Loop
        best_samples = []
        RS = np.random.RandomState(seed=seed)
        for i in trange(epochs):
            # Get real and fake samples
            noise = self.make_noise(RS, minibatch_size)
            fake = self.Generator_forwardPropagate(noise)
            real = self.get_random(RS, minibatch_size)
            # Forward propagate minibatch
            minibatch = np.vstack((real, fake))
            Y_hat = self.Descriminator_forwardPropagate(minibatch)
            _ = Descriminator_OutputLayer_RealAndFake.forwardPropagate(Y_hat)
            # Back propagate in Descriminator
            self.Descriminator_backwardsPropagate(Descriminator_OutputLayer_RealAndFake, learning_rate)
            # Forward propagate just fake data
            Y_hat = self.Descriminator_forwardPropagate(fake)
            _ = Descriminator_OutputLayer_Fake.forwardPropagate(Y_hat)
            best_samples.append(self.postprocess(fake[np.argmax(Y_hat),:]))
            # Back propagate through Descriminator and Generator
            self.Generator_backwardsPropagate(Descriminator_OutputLayer_Fake, learning_rate)
        return best_samples
    def new_data(self, X, Mean, Std):
        self.Data, self.Mean, self.Std = X, Mean, Std


if __name__ == "__main__": 
    pass

    