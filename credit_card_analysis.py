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

# [2.1. Loading the data]
# Upload the xls file:
df = pd.read_excel('default of credit card clients.xls', header=1) #4:

# [2.2. Initial data exploration]
# Check the first few lines:
print(df.head()) #5:

# Information about columns and data types:
print(df.info()) #6:

# Check for missing values:
print(df.isnull().sum()) #7:

# Descriptive statistics:
print(df.describe()) #8:

# [2.3. Data Cleaning and Preprocessing]
# Handle missing values:
df.fillna(df.mean(), inplace=True) #9:

# Encode categorical variables:
df = pd.get_dummies(df, columns=['SEX', 'EDUCATION', 'MARRIAGE'], drop_first=True) #10:

# Normalize/Standardize numeric variables:
scaler = StandardScaler() #11:
df[['LIMIT_BAL', 'AGE']] = scaler.fit_transform(df[['LIMIT_BAL', 'AGE']]) #11:

# Split data into training and testing:
x = df.drop('default payment next month', axis=1) #12:
y = df['default payment next month'] #12:
# Spliting:
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) #13: