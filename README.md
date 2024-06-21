# Titanic Classification - Module 13 & 14 Review

## Background
The aim of this notebook is to preprocess and run different machine learning models on the titanic dataset. The following steps will be used:

1. Import Libraries
2. Import Dataset
3. Exploratory data analysis
4. Splitting Data
5. Imputation
6. Feature Engineering
7. Standardization
8. Encoding
9. Feature Selection
10. Synthetic Sampling
11. Model Selection and Hyperparameter tuning

## Code Source
The code location is: [Click Here to view](https://github.com/jaidevkler/Module-13-14-Review)

## Files
Titanic-Classification.ipynb [Click here to view](https://github.com/jaidevkler/Module-13-14-Review/blob/main/Titanic-Classification.ipynb)<br />
titanic.csv [Click here to view](https://github.com/jaidevkler/Module-13-14-Review/blob/main/Resources/titanic.csv)

## How to run the program
Download the files and then use jupyter notebook or jupyter lab to open the Titanic-Classification.ipynb file.<br />

## Imports required
```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report
```

## Reference
* Dataset: https://www.kaggle.com/competitions/titanic