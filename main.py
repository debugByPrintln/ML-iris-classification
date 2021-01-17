import numpy as np
from random import *
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
plt.rc('font', family='Verdana')

from sklearn.datasets import load_iris
irisDataset = load_iris()

from sklearn.model_selection import train_test_split
XTrain, XTest, YTrain, YTest = train_test_split(irisDataset['data'], irisDataset['target'], random_state=0)

irisDataframe = pd.DataFrame(XTrain, columns=irisDataset.feature_names)

grr = pd.plotting.scatter_matrix(irisDataframe, c=YTrain, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()