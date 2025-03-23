# FILE: /credit_card_analysis.py

# [0. DATABASE]
# Link: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients #1:

# [1. ENVIRONMENT CONFIGURATION]
# Required libs:
# $pip install pandas numpy scikit-learn matplotlib seaborn xlrd openpyxl joblib #2:

# [2. DATA IMPORT AND EXPLORATION]
# [2.1. Importing libraries]: #3:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# [2.2. Loading the data]
# [2.2.1. Upload the xls file]:
df = pd.read_excel('default of credit card clients.xls', header=1) #4:

# [2.3. Initial data exploration]
# [2.3.1. Check the first few lines]:
print(df.head()) #5:

# [2.3.2. Information about columns and data types]:
print(df.info()) #6:

# [2.3.3. Check for missing values]:
print(df.isnull().sum()) #7:

# [2.3.4. Descriptive statistics]:
print(df.describe()) #8:

# [3. DATA CLEANING AND PROCESSING]
# [3.1. Handle missing values]:
df.fillna(df.mean(), inplace=True) #9:

# [3.2. Encode categorical variable]:
df = pd.get_dummies(df, columns=['SEX', 'EDUCATION', 'MARRIAGE'], drop_first=True) #10:

# [3.3. Normalize/Standardize numeric variables]:
scaler = StandardScaler() #11:
df[['LIMIT_BAL', 'AGE']] = scaler.fit_transform(df[['LIMIT_BAL', 'AGE']]) #11:

# [3.4. Split data into training and testing]:
x = df.drop('default payment next month', axis=1) #12:
y = df['default payment next month'] #12:
# [3.4.1. Spliting]:
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) #13:

# [4. MODELING]
# 