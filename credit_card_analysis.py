# FILE: /credit_card_analysis.py

#   [0. DATABASE]
# Link: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients #1:

#   [1. ENVIRONMENT CONFIGURATION]
# Required libs:
# $pip install pandas numpy scikit-learn matplotlib seaborn xlrd openpyxl joblib #2:

#   [2. DATA IMPORT AND EXPLORATION]
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

from sklearn.model_selection import cross_val_score #18:
from sklearn.model_selection import GridSearchCV #19:

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

#   [3. DATA CLEANING AND PROCESSING]
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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) #13:

#   [4. MODELING]
# [4.1. Choose a model]:
model = RandomForestClassifier(random_state=42) #14:
model.fit(x_train, y_train) #14:

# [4.2. Evaluate the model]:
y_pred = model.predict(x_test) #15:

# [4.2.1. Classification report]:
print(classification_report(y_test, y_pred)) #16:

# [4.2.2. Confusion matrix]:
print(confusion_matrix(y_test, y_pred)) #17:

# [4.3. Cross-validation]:
scores = cross_val_score(model, x, y, cv=5, scoring='accuracy') #18:
print(f'Average accuracy with cross-validation: {scores.mean()}') #18:

# [4.4. Hyperparameter tuning (optional)]: #19:
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10]
# }

# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
# grid_search.fit(x_train, y_train)
# print(f"Best parameters: {grid_search.best_params_}")

#   [5. SAVING MODEL]
# [5.1. Savind the trained model]:
joblib.dump(model, 'credit_card_default_model.pkl') #20:

# [5.2. Load the model]:
model = joblib.load('credit_card_default_model.pkl') #21:

#   [6. VIEWING RESULTS]
# [6.1. Create charts]:
importances = model.feature_importances_ #22:
feature_names = x.columns #22:

sns.barplot(x=importances, y=feature_names) #23:
plt.title('Importance of Variables') #23:
plt.show()

# [6.2. Confusion Matrix]:
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d') #24:
plt.title('Confusion Matrix')
plt.show()