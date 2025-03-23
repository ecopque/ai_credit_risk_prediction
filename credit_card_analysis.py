# FILE: /credit_card_analysis.py

# [0. DATABASE]
# [https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients] #1:

# [1. ENVIRONMENT CONFIGURATION]
# [1. Required libs]:
# $pip install pandas numpy scikit-learn matplotlib seaborn xlrd openpyxl joblib #2:

# [2. DATA IMPORT AND EXPLORATION]
# [2. Importing libraries]: #3:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# [2. Loading the data]
# [2. Upload the xls file): #4:
df = pd.read_excel('default of credit card clients.xls', header=1)

# [2. Initial data exploration]
# [2. Check the first few lines]:
print(df.head()) #5:
# [2. Information about columns and data types]:
print(df.info()) #6:
# [2. Check for missing values]:
print(df.isnull().sum()) #7:
# [2. Descriptive statistics]:
print(df.describe()) #8:

# [2. Data Cleaning and Preprocessing]
# [2. Fill missing values ​​with mean (for numeric columns)]:
df.fillna(df.mean(), inplace=True) #9:
# [2. Encode categorical variables]
df = pd.get_dummies(df, columns=['SEX', 'EDUCATION', 'MARRIAGE'], drop_first=True) #10: