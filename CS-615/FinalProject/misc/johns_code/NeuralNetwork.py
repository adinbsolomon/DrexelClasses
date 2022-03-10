# Author: John George
# Drexel University CS 615 - Deep Learning
# Homework 6
# August 06, 2021

import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import gzip
import PIL.Image as pil

def separate_features_target(data):
    #Isolate the feature values in the training data
    features = data[:,0:data.shape[1] - 1] #strip the value column
    targets = data[:,data.shape[1] - 1:]
    return (features, targets)

def find_mean_std(data):
    mean = np.mean(data, axis=0)
    standardDeviation = np.std(np.array(data), axis=0, ddof=0)
    return (mean, standardDeviation)

def zScore_data(dataMatrix, meanArr, stdArr):
    standardizedData = (np.matrix(dataMatrix) - meanArr) / (stdArr + 0.000000001)
    return standardizedData

def normalize_data(dataMatrix, maxValue):
    normalizedData = dataMatrix / maxValue
    return normalizedData

def collapse_probability_distribution(data):
    classCount = len(data[0].A1)

    distributionCollapse = []
    for prediction in data:
        maxPos = 0
        maxProb = 0
        predictionArr = prediction.A1
        for i in range(len(predictionArr)):
            prob = predictionArr[i]
            if prob > maxProb:
                maxProb = prob
                maxPos = i
        collapsedProbability = np.zeros(classCount)
        collapsedProbability[maxPos] = 1
        distributionCollapse.append(collapsedProbability)
    return np.matrix(distributionCollapse)

def array_chunkify(data, size):
    chunks = [np.matrix(data[i:i + size]) for i in range(0, len(data), size)]

    remainder = len(data) % size

    if remainder > 0 and remainder < len(data) / 2:
        return chunks[0:len(chunks) - 1]

    return chunks

def sfold(X, Y, foldCount):
    # Shuffle the Training data and Targets in the same way
    permutation = np.random.permutation(len(X))
    XShuffled = X[permutation]
    YShuffled = Y[permutation]

    dataSets = []
    spread = math.ceil(len(X) / foldCount)

    i = 0
    j = spread

    while j < len(X):
        xLow = XShuffled[0:i]
        xValidation = XShuffled[i:j]
        xHigh = XShuffled[j:]

        yLow = YShuffled[0:i]
        yValidation = YShuffled[i:j]
        yHigh = YShuffled[j:]

        xTrain = np.append(xLow, xHigh, axis=0)
        yTrain = np.append(yLow, yHigh, axis=0)

        dataSets.append((xTrain, yTrain, xValidation, yValidation))

        i = j
        j = j + spread

    return dataSets

def log_loss_accuracy(target, predictionDistribution):
    prediction = collapse_probability_distribution(predictionDistribution)

    trainingCorrect = 0
    for i in range(len(prediction)):
        predictionArr = prediction[i].A1
        targetArr = target[i].A1

        # Scan through the target and prediction index by index
        # If there is a position where both values == 1
        # then the prediction was correct.
        for j in range(len(predictionArr)):
            if(predictionArr[j] == 1 and targetArr[j] == 1):
                trainingCorrect += 1
                break

    return trainingCorrect

def least_squares_accuracy(target, prediction):
    prediction = collapse_probability_distribution(prediction)

    trainingCorrect = 0
    for i in range(len(prediction)):
        predictionArr = prediction[i].A1
        targetArr = target[i].A1

        # Scan through the target and prediction index by index
        # If there is a position where both values == 1
        # then the prediction was correct.
        for j in range(len(predictionArr)):
            if(predictionArr[j] == 1 and targetArr[j] == 1):
                trainingCorrect += 1
                break

    return trainingCorrect

def plotCharacter(characterMatrix):
    pixels = characterMatrix.reshape(28, 28).T
    pixels = np.asarray(pixels * 255, dtype=np.uint8)

    image = pil.fromarray(pixels, 'L')
    image = image.resize((280,280))

    return image

def import_data(targetClass):
    np.random.seed(0)

    # Read in the training data
    print('Importing training data...')
    f = gzip.open('../emnist-digits-train-images-idx3-ubyte.gz', 'r')
    magicNumber = int.from_bytes(f.read(4), 'big')
    imageCount = int.from_bytes(f.read(4), 'big')
    rowCount = int.from_bytes(f.read(4), 'big')
    columnCount = int.from_bytes(f.read(4), 'big')
    imageSize = rowCount * columnCount
    imageData = f.read(imageCount * imageSize)
    data = np.frombuffer(imageData, dtype=np.uint8).astype(np.float32)
    trainData = data.reshape(imageCount, imageSize)

    # Training Labels
    f = gzip.open('../emnist-digits-train-labels-idx1-ubyte.gz', 'r')
    magicNumber = int.from_bytes(f.read(4), 'big')
    imageCount = int.from_bytes(f.read(4), 'big')
    imageLabels = f.read(imageCount)
    trainLabels = np.frombuffer(imageLabels, dtype=np.uint8).astype(np.int32)
    print(f'Training data imported. {len(trainData)} observations')

    # Convert the training values to numeric
    print('Preparing training data...')

    #Separate the features and targets for the training data
    XTrain = []
    YTrain = []
    for pos in range(len(trainLabels)):
        trainLabel = trainLabels[pos]
        if trainLabel == targetClass:
            XTrain.append(trainData[pos])
            YTrain.append(trainLabel)
    XTrain = np.matrix(XTrain) / 255
    YTrain = np.array(YTrain)
    print('Training data prepared')

    # Read in the validation data
    print('Importing validation data...')
    f = gzip.open('../emnist-digits-test-images-idx3-ubyte.gz', 'r')
    magicNumber = int.from_bytes(f.read(4), 'big')
    imageCount = int.from_bytes(f.read(4), 'big')
    rowCount = int.from_bytes(f.read(4), 'big')
    columnCount = int.from_bytes(f.read(4), 'big')
    imageSize = rowCount * columnCount
    imageData = f.read(imageCount * imageSize)
    data = np.frombuffer(imageData, dtype=np.uint8).astype(np.float32)
    testData = data.reshape(imageCount, imageSize)

    f = gzip.open('../emnist-digits-test-labels-idx1-ubyte.gz', 'r')
    magicNumber = int.from_bytes(f.read(4), 'big')
    imageCount = int.from_bytes(f.read(4), 'big')
    imageLabels = f.read(imageCount)
    testLabels = np.frombuffer(imageLabels, dtype=np.uint8).astype(np.int32)
    print(f'Validation data imported. {len(testData)} observations')

    # Convert the validation values to numeric
    print('Preparing validation data...')
    shufflePermutation = np.random.permutation(imageCount)
    testData = testData[shufflePermutation]
    testLabels = testLabels[shufflePermutation]

    # #Separate the features and targets for the testing data
    XTest = []
    YTest = []
    for pos in range(len(testLabels)):
        testLabel = testLabels[pos]
        if testLabel == targetClass:
            XTest.append(testData[pos])
            YTest.append(testLabel)
    XTest = np.matrix(XTest) / 255
    YTest = np.array(YTest)
    print('Validation data prepared')

    return (XTrain, YTrain, XTest, YTest)

def plot_results(epochArr, trainLossArr, validationLossArr, xLabel, yLabel, title):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(epochArr, trainLossArr, color='orangered', linewidth=2, label='Training Data')
    ax.plot(epochArr, validationLossArr, color='blue', linewidth=2, label='Validation Data')
    ax.legend(loc='upper right', fontsize='x-large')
    ax.set_xlabel(xLabel, fontsize='x-large')

    ax.set_ylabel(yLabel, fontsize='x-large')
    ax.set_title(title, fontsize='x-large')

    lastPointTraining = (epochArr[len(epochArr) - 1], trainLossArr[len(trainLossArr) - 1])

    ax.scatter(lastPointTraining[0], lastPointTraining[1], color='r')
    ax.annotate(f'J = {round(lastPointTraining[1], 2)}', (lastPointTraining[0] - 30, lastPointTraining[1] + 0.5))

    plt.show()   

class Layer:
    def forwardPropagate(self, data):
        pass

    def gradient(self):
        pass

    def backPropagate(self, gradientAccum, isupdateWeights=True):
        pass

    def eval(self, target):
        pass

class InputLayer(Layer):
    def __init__(self):
        pass

    def forwardPropagate(self, data):
        return data

class NoiseLayer(Layer):
    def __init__(self, noiseSize):
        self.NoiseSize = noiseSize

    def forwardPropagate(self, data):
        shape = (len(data), self.NoiseSize)
        noise = np.matrix(np.random.rand(shape[0], shape[1]))
        return noise

class InputLayerZScore(Layer):
    def __init__(self, data):
        meanAndStd = find_mean_std(data)
        self.MeanArr = meanAndStd[0]
        self.StdArr = meanAndStd[1]

    def forwardPropagate(self, data):
        zData = zScore_data(data, self.MeanArr, self.StdArr)
        return zData

class InputLayerNormalize(Layer):
    def __init__(self, maxValue):
        self.MaxValue = maxValue

    def forwardPropagate(self, data):
        zData = normalize_data(data, self.MaxValue)
        return zData

class ReLuLayer(Layer):
    def __init__(self):
        self.ForwardData = []

    def forwardPropagate(self, data):
        forwardData = np.matrix([[0 if (val < 0) else val for val in row.A1] for row in data])
        self.ForwardData = forwardData
        return forwardData

    def gradient(self):
        return np.matrix([[1 if val > 0 else 0 for val in row.A1] for row in self.ForwardData])

    def backPropagate(self, gradientAccum, isUpdateWeights=False):
        return np.multiply(self.gradient(), gradientAccum)

class SigmoidLayer(Layer):
    def __init__(self):
        self.ForwardData = []

    def forwardPropagate(self, data):
        forwardData = 1 / (1 + np.exp(-1 * data))
        self.ForwardData = forwardData
        return forwardData

    def gradient(self):
        return np.multiply(self.ForwardData, (1 - self.ForwardData))

    def backPropagate(self, gradientAccum, isUpdateWeights=False):
        return np.multiply(self.gradient(), gradientAccum)

class SoftmaxLayer(Layer):
    def __init__(self):
        self.ForwardData = []

    def forwardPropagate(self, data):
        try:
            dataExp = np.matrix(np.exp(data))
            softmaxDenominator = np.matrix(np.sum(dataExp, axis=1))
            forwardData = np.divide(dataExp, softmaxDenominator)
            self.ForwardData = forwardData
            return forwardData
        except:
            print('Overflow error')
            print(data)
            raise('Overflow error')

    def gradient(self):
        return np.multiply(self.ForwardData, (1 - self.ForwardData))

    def backPropagate(self, gradientAccum, isUpdateWeights=False):
        return np.multiply(self.gradient(), gradientAccum)

class HyperbolicTangentLayer(Layer):
    def __init__(self):
        self.ForwardData = []

    def forwardPropagate(self, data):
        exp = np.exp(data)
        expNeg = np.exp(-1 * data)
        forwardData = (exp - expNeg) / (exp + expNeg)
        self.ForwardData = forwardData
        return forwardData

    def gradient(self):
        return 1 - np.multiply(self.ForwardData, self.ForwardData)

    def backPropagate(self, gradientAccum, isUpdateWeights=False):
        return np.multiply(self.gradient(), gradientAccum)

class FullyConnectedLayer(Layer):
    def __init__(self, featureCount, outputCount, initialLearningRate):
        self.Weights = (np.random.rand(featureCount, outputCount) * 2 - 1) / 100
        self.BiasArr = np.matrix([(np.random.random() * 2 - 1) / 100 for x in range(outputCount)])
        self.OutputCount = outputCount
        self.ForwardData = []
        self.InitialLearningRate = initialLearningRate

    def forwardPropagate(self, data):
        forwardData = (data @ self.Weights) + self.BiasArr
        self.ForwardData = np.matrix(forwardData)
        self.InputData = data
        return forwardData

    def gradient(self):
        return self.Weights.T

    def backPropagate(self, gradientAccum, isUpdateWeights=True):
        inputSize = len(self.InputData)
        returnGradient = gradientAccum @ self.gradient()

        if isUpdateWeights:
            multipliedGradient = (self.InputData.T @ gradientAccum) / inputSize
            self.Weights = self.Weights - self.InitialLearningRate * multipliedGradient
            
            ones = np.matrix(np.ones(inputSize))
            biasDelta = (ones @ gradientAccum) / inputSize
            self.BiasArr = self.BiasArr - self.InitialLearningRate * biasDelta

        return returnGradient

class FullyConnectedLayerAdam(Layer):
    def __init__(self, featureCount, outputCount, initialLearningRate):
        self.Weights = (np.random.rand(featureCount, outputCount) * 2 - 1) / 100
        self.BiasArr = np.array([(np.random.random() * 2 - 1) / 100 for x in range(outputCount)])
        self.OutputCount = outputCount
        self.ForwardData = []

        self.DecayMomentum = 0.9
        self.DecayRms = 0.999
        self.Stability = 0.00000001

        self.MomentumAccumulator = 0
        self.RmsAccumulator = 0
        self.InitialLearningRate = initialLearningRate

    def forwardPropagate(self, data):
        forwardData = (data @ self.Weights) + self.BiasArr
        self.ForwardData = forwardData
        self.InputData = data
        return forwardData

    def gradient(self):
        return self.Weights.T

    def backPropagate(self, gradientAccum, isUpdateWeights=True):
        inputSize = len(self.InputData)
        returnGradient = gradientAccum @ self.gradient()

        multipliedGradients = (self.InputData.T @ gradientAccum) / inputSize

        self.MomentumAccumulator = (
            self.DecayMomentum * self.MomentumAccumulator 
            + (1 - self.DecayMomentum) * multipliedGradients
        )

        self.RmsAccumulator = (
            self.DecayRms * self.RmsAccumulator
            + (1 - self.DecayRms) * np.square(multipliedGradients)
        )

        if isUpdateWeights:
            self.Weights = self.Weights - self.InitialLearningRate * (
                (
                    self.MomentumAccumulator / (1 - self.DecayMomentum)
                )
                / (
                    np.sqrt(self.RmsAccumulator / (1 - self.DecayRms)) + self.Stability
                )
            )
            
            ones = np.matrix(np.ones(inputSize))
            biasDelta = (ones @ gradientAccum) / inputSize
            self.BiasArr = self.BiasArr - self.InitialLearningRate * biasDelta

        return returnGradient

class OutputLeastSquares(Layer):
    def __init__(self):
        self.AccuracyMethod = least_squares_accuracy
        pass
    
    def forwardPropagate(self, data):
        self.ForwardData = data
        return data

    def eval(self, target):
        diff = self.ForwardData - target
        return np.sum(np.sqrt(np.multiply(diff, diff)))

    def gradient(self, target):
        return 2 * (self.ForwardData - target)

    def countCorrect(self, target, prediction):
        return self.AccuracyMethod(target, prediction)

class OutputLogLoss(Layer):
    def __init__(self):
        self.AccuracyMethod = log_loss_accuracy
    
    def forwardPropagate(self, data):
        self.ForwardData = data
        return data

    def eval(self, target):
        return -1 * np.sum(
            np.multiply(target, np.log(self.ForwardData + 0.00000001))
            + np.multiply((np.ones(target.shape) - target), np.log(np.ones(self.ForwardData.shape) - self.ForwardData + 0.00000001))
        )

    def gradient(self, target):
        return (
            (self.ForwardData - target) 
            / (np.multiply(self.ForwardData, 1 - self.ForwardData) + 0.00000001)
        )

    def countCorrect(self, target, prediction):
        return self.AccuracyMethod(target, prediction)

class OutputCrossEntropy(Layer):
    def __init__(self):
        pass
    
    def forwardPropagate(self, data):
        self.ForwardData = data
        return data

    def eval(self, target):
        evaluation = np.sum(np.negative(np.multiply(target, np.log(self.ForwardData + 0.00000001))))
        return evaluation

    def gradient(self, target):
        return np.negative(target / (self.ForwardData + 0.00000001))

    def countCorrect(self, target, prediction):
        return self.AccuracyMethod(target, prediction)

class OutputNegativeLog(Layer):
    def __init__(self):
        self.AccuracyMethod = log_loss_accuracy

    def forwardPropagate(self, data):
        self.ForwardData = data
        return data

    def eval(self, target):
        return -1 * np.log(self.ForwardData)

    def gradient(self, target):
        return -1 * np.divide(1, self.ForwardData)

    def countCorrect(self, target, prediction):
        return self.AccuracyMethod(target, prediction)

class NeuralNetwork:
    def __init__(self, layers):
        self.Layers = layers
        self.EpochArr = []
        self.ValidationLossArr = []
        self.TrainLossArr = []
        self.AnimationImages = []

    def evalLoss(self, data, target):
        self.forwardPropagate(data)
        return self.Layers[len(self.Layers) - 1].eval(target)

    def evalInput(self, data):
        return self.forwardPropagate(data)

    def forwardPropagate(self, data):
        returnData = data
        for layer in self.Layers:
            returnData = layer.forwardPropagate(returnData)

        return returnData
    
    def backPropagate(self, target):
        self.Layers.reverse()
        gradientAccum = None
        layerCount = 0
        for layer in self.Layers:
            if layerCount == 0:
                gradientAccum = layer.gradient(target)
            else:
                gradientAccum = layer.backPropagate(gradientAccum)

            layerCount = layerCount + 1

        self.Layers.reverse()
        return gradientAccum

    def countCorrect(self, target, prediction):
        return self.Layers[len(self.Layers) - 1].countCorrect(target, prediction)

class GenerativeAdversarialNetwork:
    def __init__(self, generator, discriminator):
        self.Generator = generator
        self.Discriminator = discriminator
        self.EpochArr = []
        self.ValidationLossArr = []
        self.TrainLossArr = []
        self.AnimationImages = []

    def forwardPropagateGenerator(self, data):
        # noise = np.matrix(np.random.rand(shape[0], shape[1]))
        fakeInput = self.Generator.forwardPropagate(data)
        # avgs = np.average(fakeInput, axis=1)
        # maxs = np.max(fakeInput, axis=1)
        # fakeInput = np.multiply(fakeInput, (np.matrix(np.ones(fakeInput.shape) / 2) / (fakeInput + 0.000000001)))
        # avgs2 = np.average(fakeInput, axis=1)
        # fakeInput = np.around(fakeInput, decimals=0)
        # fakeInput = np.matrix([[1 if f > avgs[pos].A1 else 0 for f in fakeInput[pos].A1] for pos in range(len(fakeInput))])
        # fakeInput = np.matrix([[f if f < 255 else 255 for f in ob.A1] for ob in fakeInput])
        # fakeInput = fakeInput / 255
        return fakeInput

    def forwardPropagateDiscriminator(self, data):
        discriminatorOutput = self.Discriminator.forwardPropagate(data)
        return discriminatorOutput

    def evalLoss(self, data, throwaway):
        target = np.matrix(np.ones(len(data)) / 2).T
        return self.Discriminator.evalLoss(data, target)

    def evalInput(self, data):
        return self.Discriminator.forwardPropagate(data)

    def forwardPropagate(self, data):
        # Initialize the noise input for the generator
        fakeTargets = np.zeros((len(data), 1))
        realTargets = np.ones((len(data), 1))
        self.Targets = np.append(realTargets, fakeTargets, axis=0)

        # Generate noise and pass it through all the generator layers
        fakeInput = self.forwardPropagateGenerator(data)

        # Concatenate the fake data to the real data
        # The real data is first
        discriminatorData = np.append(data, fakeInput, axis=0)

        # Pass the training and fake data through the discriminator layers
        discriminatorData = self.forwardPropagateDiscriminator(discriminatorData)

        self.FakeData = fakeInput
        self.FakePredictions = discriminatorData[0:len(fakeTargets)]
        return discriminatorData

    def backPropagate(self, throwaway):
        # Backpropagate through the Discriminator, updating its weights and biases
        discriminatorGradientAccum = self.Discriminator.backPropagate(self.Targets)

        # Backpropagate through the Generator. Part of the backpropagation is through
        # the Discriminator, but we want to leave the Discriminator weights and biases
        # unchanged this time. We want to update only the Generator's weights and biases.
        # We are only backpropagating the fake data.
        # 1. Begin with the the gradient from the Generator's objective function
        # 2. Backpropagate through the Discriminator's hidden layers, without updating weights and biases
        # 3. Backpropagate through the Generator's hidden layers, do update the weights and biases.

        # Pass just the fake data through the discriminator layers
        # Preparing for the generator backpropatation with the fake predictions data
        self.forwardPropagateDiscriminator(self.FakeData)

        # Backpropagate through the Generator's objective function
        generatorObjectiveLayer = self.Generator.Layers[len(self.Generator.Layers) - 1]
        generatorObjectiveLayer.forwardPropagate(self.FakePredictions)
        generatorGradientAccum = generatorObjectiveLayer.gradient(self.FakePredictions)

        # Backpropagate through the discriminator layers
        self.Discriminator.Layers.reverse() # Discriminator Reversed
        for layerIndex in range(1, len(self.Discriminator.Layers) - 1):
            layer = self.Discriminator.Layers[layerIndex]
            generatorGradientAccum = layer.backPropagate(generatorGradientAccum, isUpdateWeights=False)
        self.Discriminator.Layers.reverse() # Discriminator Normal

        # Backpropagate through the Generator's hidden layers
        self.Generator.Layers.reverse() # Generator Reversed
        for layerIndex in range(1, len(self.Generator.Layers) - 1):
            layer = self.Generator.Layers[layerIndex]
            generatorGradientAccum = layer.backPropagate(generatorGradientAccum, isUpdateWeights=True)

        self.Generator.Layers.reverse()

    def countCorrect(self, throwaway, prediction):
        target = np.matrix(np.ones(len(prediction)) / 2).T
        return self.Discriminator.Layers[len(self.Discriminator.Layers) - 1].countCorrect(target, prediction)

class Strategy:
    def train_epochs(self, network, epochs, XTrain, YTrain, XTest, YTest):
        pass

class NormalStrategy:
    def train_epochs(self, network, epochs, XTrain, YTrain, XTest, YTest):
        epochArr = []
        trainLossArr = []
        validationLossArr = []

        print('Beginning epochs...')
        epochLimit = epochs
        for epoch in range(epochLimit):
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch} / {epochLimit}')

            # Evaluate the Validation data.
            network.forwardPropagate(XTest)
            evalValidation = network.evalLoss(YTest)

            # Train the stochastic chunk
            network.forwardPropagate(XTrain)
            network.backPropagate(YTrain)

            # Evaluate with the entire training set.
            eval = network.evalLoss(YTrain)

            epochArr.append(epoch)
            validationLossArr.append(evalValidation / len(XTest))
            trainLossArr.append(eval / len(XTrain))

            if (epoch + 1) % 25 == 0:
                trainPredictionDistribution = network.evalInput(XTrain)
                validationPredictionDistribution = network.evalInput(XTest)

                # Count the correct predictions
                trainingCorrect = network.countCorrect(YTrain, trainPredictionDistribution)
                validationCorrect = network.countCorrect(YTest, validationPredictionDistribution)

                print(f'Training Data Prediction Accuracy: {trainingCorrect / len(XTrain) * 100}')
                print(f'Validation Data Prediction Accuracy: {validationCorrect / len(XTest) * 100}')

        print('Epochs complete')

        return (epochArr, trainLossArr, validationLossArr)

class StochasticStrategy(Strategy):
    def __init__(self, stochasticChunkSize):
        self.StochasticChunkSize = int(stochasticChunkSize)

    def train_epochs(self, network, epochs, XTrain, YTrain, XTest, YTest, epochFunction):
        stochasticChunkSize = self.StochasticChunkSize

        print('Beginning epochs...')
        epochLimit = epochs
        for epoch in range(epochLimit):
            if (epoch) % 10 == 0:
                print(f'Epoch {epoch} / {epochLimit}')

            # Shuffle the Training data and Targets in the same way
            permutation = np.random.permutation(len(XTrain))
            XTrain = XTrain[permutation]
            YTrain = YTrain[permutation]
            
            # Get the stochastic chunk of training data
            # XChunk = array_chunkify(XTrain, stochasticChunkSize)[0]
            # YChunk = array_chunkify(YTrain, stochasticChunkSize)[0]
            XChunk = np.matrix(XTrain[0:stochasticChunkSize])
            YChunk = np.matrix(YTrain[0:stochasticChunkSize])

            # Train the stochastic chunk
            network.forwardPropagate(XChunk)
            network.backPropagate(YChunk)

            epochFunction(epoch, network, XTrain, YTrain, XTest, YTest)

        print('Epochs complete')

class CrossValidationStrategy(Strategy):
    def __init__(self, foldCount):
        self.FoldCount = foldCount

    def train_epochs(self, network, epochs, XTrain, YTrain, XTest, YTest):
        epochArr = []
        trainLossArr = []
        validationLossArr = []

        # Concatenate the training and validation data
        X = np.append(XTrain, XTest, axis=0)
        Y = np.append(YTrain, YTest, axis=0)

        # Shuffle the Training data and Targets in the same way
        permutation = np.random.permutation(len(X))
        X = X[permutation]
        Y = Y[permutation]

        folds = sfold(X, Y, self.FoldCount)
        trainSize = len(folds[0][0])
        validationSize = len(folds[0][2])

        print('Beginning epochs...')
        epochLimit = epochs
        for epoch in range(epochLimit):
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch} / {epochLimit}')

            # Evaluate the network with the test data
            evalValidationAccum = 0
            for fold in folds:
                testData = fold[2]
                testTargets = fold[3]

                network.forwardPropagate(testData)
                evalValidationAccum += network.evalLoss(testTargets)

            # Train the network with the train data
            evalAccum = 0
            for fold in folds:
                trainData = fold[0]
                trainTargets = fold[1]

                network.forwardPropagate(trainData)
                network.backPropagate(trainTargets)

                # Evaluate with the entire training set.
                evalAccum += network.evalLoss(trainTargets)        

            epochArr.append(epoch)
            validationLossArr.append(evalValidationAccum / (self.FoldCount * validationSize))
            trainLossArr.append(evalAccum / (self.FoldCount * trainSize))

            if (epoch + 1) % 25 == 0:
                trainPredictionDistribution = network.forwardPropagate(X)

                # Count the correct predictions
                trainingCorrect = network.countCorrect(Y, trainPredictionDistribution)

                print(f'SFold Prediction Accuracy: {trainingCorrect / len(X) * 100}')

        print('Epochs complete')

        return (epochArr, trainLossArr, validationLossArr)

def ProblemRunner(network, data, epochs, strategy, xLabel, yLabel, title):
    XTrain = data[0]
    YTrain = data[1]
    XTest = data[2]
    YTest = data[3]

    results = strategy.train_epochs(
        network,
        epochs=epochs, 
        XTrain=XTrain, 
        YTrain=YTrain, 
        XTest=XTest, 
        YTest=YTest
    )

    epochArr = results[0]
    trainLossArr = results[1]
    validationLossArr = results[2]

    print('Calculating Accuracy...')
    trainPredictionDistribution = network.forwardPropagate(XTrain)
    validationPredictionDistribution = network.forwardPropagate(XTest)

    # Count the correct predictions
    trainingCorrect = network.countCorrect(YTrain, trainPredictionDistribution)
    validationCorrect = network.countCorrect(YTest, validationPredictionDistribution)

    print(f'Training Data Prediction Accuracy: {trainingCorrect / len(XTrain) * 100}')
    print(f'Validation Data Prediction Accuracy: {validationCorrect / len(XTest) * 100}')

    plot_results(
        epochArr, 
        trainLossArr, 
        validationLossArr, 
        xLabel=xLabel, 
        yLabel=yLabel, 
        title=title)

def GANRunner(network, targetClass, data, epochs, strategy, xLabel, yLabel, title):
    def GANEpochFunction(epoch, network, XTrain, YTrain, XTest, YTest):
        if (epoch) % 50 == 0:
            generatedImage = network.forwardPropagateGenerator(XTrain[0])
            print(network.evalInput(generatedImage))

            # Evaluate the Validation data.
            evalValidation = network.evalLoss(XTest, YTest)

            # Shuffle the Training data and Targets in the same way
            permutation = np.random.permutation(len(XTrain))
            evalDataSize = int(len(XTrain) / 5)
            X = (XTrain[permutation])[0:evalDataSize]
            Y = (YTrain[permutation])[0:evalDataSize]

            # Evaluate with the entire training set.
            eval = network.evalLoss(X, Y)

            network.EpochArr.append(epoch)
            network.ValidationLossArr.append(evalValidation / len(XTest))
            network.TrainLossArr.append(eval / len(X))

            # trainPredictionDistribution = network.evalInput(XTrain)
            # validationPredictionDistribution = network.evalInput(XTest)

            # # Count the correct predictions
            # trainingCorrect = network.countCorrect(YTrain, trainPredictionDistribution)
            # validationCorrect = network.countCorrect(YTest, validationPredictionDistribution)

            # print(f'Training Data Prediction Accuracy: {trainingCorrect / len(XTrain) * 100}')
            # print(f'Validation Data Prediction Accuracy: {validationCorrect / len(XTest) * 100}')

        if (epoch) % 100 == 0:
            generatedImage = network.forwardPropagateGenerator(XTrain[0])
            image = plotCharacter(generatedImage)
            network.AnimationImages.append(image)

    XTrain = data[0]
    YTrain = data[1]
    XTest = data[2]
    YTest = data[3]

    strategy.train_epochs(
        network,
        epochs=epochs, 
        XTrain=XTrain, 
        YTrain=YTrain, 
        XTest=XTest, 
        YTest=YTest,
        epochFunction=GANEpochFunction
    )

    # Save the network training animation
    if len(network.AnimationImages) > 0:
        network.AnimationImages[0].save(f'TrainAnimation{targetClass}.gif', save_all=True, append_images=network.AnimationImages[1:], duration=300, loop=0)

    # print('Calculating Accuracy...')
    # fakeInput = network.forwardPropagateGenerator((1000, XTrain.shape[1]))
    # discriminatorPrediction = network.evalInput(fakeInput)

    # Count the correct predictions
    # trainingCorrect = network.countCorrect(YTrain, trainPredictionDistribution)

    # generatedImage = network.forwardPropagateGenerator(XTrain[0])
    # plotCharacter(generatedImage)

    # plot_results(
    #     network.EpochArr, 
    #     network.TrainLossArr, 
    #     network.ValidationLossArr, 
    #     xLabel=xLabel, 
    #     yLabel=yLabel, 
    #     title=title)

def ProblemGAN(targetClass):
    data = import_data(targetClass)

    XTrain = data[0]
    inputSize = len(XTrain[0].A1)

    generator = NeuralNetwork([
        NoiseLayer(inputSize),
        FullyConnectedLayer(inputSize, inputSize, 0.001),
        ReLuLayer(),
        OutputNegativeLog()
    ])

    discriminator = NeuralNetwork([
        InputLayer(),
        FullyConnectedLayer(inputSize, 1, 0.001),
        SigmoidLayer(),
        OutputLogLoss()
    ])

    network = GenerativeAdversarialNetwork(generator=generator, discriminator=discriminator)

    GANRunner(network, targetClass, data, epochs=2000, strategy=StochasticStrategy(100), xLabel='Epoch', yLabel='Average Log Loss (J)', title='GAN')

def main():
    #Prevent scientific notation in numpy matrices
    np.set_printoptions(suppress=True)

    ####
    # Homework 6
    ####
    print(f'Homework 6\n')

    # Problem4_3()
    ProblemGAN(0)
    ProblemGAN(1)
    ProblemGAN(2)
    ProblemGAN(3)
    ProblemGAN(4)
    ProblemGAN(5)
    ProblemGAN(6)
    ProblemGAN(7)
    ProblemGAN(8)
    ProblemGAN(9)

if __name__ == "__main__":
    main()
