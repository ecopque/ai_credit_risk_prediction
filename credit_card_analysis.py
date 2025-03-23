# FILE: /credit_card_analysis.py

# pip install pandas numpy scikit-learn matplotlib seaborn xlrd openpyxl joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

pd = pd.read_excel('default of credit card clients.xls', header=1)