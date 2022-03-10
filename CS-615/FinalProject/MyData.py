
import numpy as np
import os

from MyPython import MyPickle, CS383, MyNumpy

# Paths and Picklers
this_path = os.path.abspath('')
data_path = this_path + os.path.sep + 'data' + os.path.sep
PickledData_path = data_path + 'pickled' + os.path.sep
PickledLetters_path = PickledData_path + 'letters.pickle'
PickledNumbers_path = PickledData_path + 'numbers.pickle'
PickledNumbersEMNIST_path = PickledData_path + 'numbersEMNIST.pickle'
PicklerLetters = MyPickle.Pickler(PickledLetters_path)
PicklerNumbers = MyPickle.Pickler(PickledNumbers_path)
PicklerNumbersEMNIST = MyPickle.Pickler(PickledNumbersEMNIST_path)

def pickleData():
    assert (os.path.basename(os.path.abspath("")) == "CS-615-Final"), "Make sure you're in the Final Project repo"
    # If everything is already pickled, then do nothing
    if False not in [os.path.isfile(path) for path in [PickledLetters_path, PickledNumbers_path, PickledNumbersEMNIST_path]]: return
    # Process and Pickle Everything
    print("Initial Pickling in Progress... at most 3 minutes")
    # 1. Numbers - all classifications from mnist
    data = MyNumpy.readCSV(data_path + 'mnist_train.csv')
    X, Mean, Std = dict(), dict(), dict()
    for c in range(10): X[c], Mean[c], Std[c] = CS383.standardize(data[np.where(data[:, 0]==c),1:][0,:,:], return_stats=True)
    PicklerNumbers.pickle({"X" : X, "Mean" : Mean, "Std" : Std})
    # 2. Letters - only classifications > 9 from emnist
    data = MyNumpy.readCSV(data_path + 'emnist-balanced-train.csv')
    X, Mean, Std = dict(), dict(), dict()
    for c in range(10,47): X[c], Mean[c], Std[c] = CS383.standardize(data[np.where(data[:, 0]==c),1:][0,:,:], return_stats=True)
    PicklerLetters.pickle({"X" : X, "Mean" : Mean, "Std" : Std})
    # 3. Numbers from emnist - only classiciations <= 9 from emnist
    X, Mean, Std = dict(), dict(), dict()
    for c in range(10): X[c], Mean[c], Std[c] = CS383.standardize(data[np.where(data[:, 0]==c),1:][0,:,:], return_stats=True)
    PicklerNumbersEMNIST.pickle({"X" : X, "Mean" : Mean, "Std" : Std})
    print("\tInitial Pickling is Done!")


def getLetters():
    pickleData()
    Data = PicklerLetters.unpickle()
    return Data["X"], Data["Mean"], Data["Std"]

def getNumbers(fromEMNIST = False):
    pickleData()
    Data = (PicklerNumbersEMNIST if fromEMNIST else PicklerNumbers).unpickle()
    return Data["X"], Data["Mean"], Data["Std"]


if __name__ == "__main__": pickleData()
