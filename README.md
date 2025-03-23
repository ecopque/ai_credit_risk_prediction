# Credit Card Default Risk Analysis

This project aims to predict credit card default risk based on historical customer data. We use machine learning techniques to build a classification model that identifies whether a customer will default on their payment in the next month.

## Project Description

The dataset used in this project is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients). It contains information about credit card clients, such as credit limit, age, gender, payment history, and bill amounts. The goal is to predict the `default payment next month` variable, which indicates whether a customer will default (1) or not (0).

The project follows these steps:
1. **Data Import and Exploration**: Loading and initial analysis of the dataset.
2. **Data Cleaning and Preprocessing**: Handling missing values, encoding categorical variables, and normalizing numeric features.
3. **Modeling**: Training a `RandomForestClassifier` to predict default risk.
4. **Model Evaluation**: Using metrics like precision, recall, F1-score, and confusion matrix to evaluate model performance.
5. **Visualization**: Generating charts to interpret feature importance and the confusion matrix.

## Technologies Used

- **Python**: Primary programming language.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Scikit-learn**: Machine learning (modeling, cross-validation, metrics).
- **Matplotlib and Seaborn**: Data visualization.
- **Joblib**: Saving and loading the trained model.

## Project Structure
```
credit_card_analysis/
├── credit_card_analysis.py # Main analysis script
├── default of credit card clients.xls # Dataset
├── credit_card_default_model.pkl # Saved trained model
├── README.md # Project documentation
└── requirements.txt # Project dependencies
```

## Results

The model achieved an average accuracy of **81.7%** with cross-validation. However, the recall for class 1 (defaulters) was **36%**, indicating that the model struggles to correctly identify customers who will default. Below are some visualizations generated:

1. **Feature Importance**:

   ![Feature Importance](https://github.com/ecopque/ai_credit_risk_prediction/blob/main/prints/Screenshot%20from%202025-03-23%2019-47-38.png)

2. **Confusion Matrix**:

   ![Confusion Matrix](https://github.com/ecopque/ai_credit_risk_prediction/blob/main/prints/Screenshot%20from%202025-03-23%2019-57-24.png)

## Developer Info
```
. Developer: Edson Copque
. Website: https://linktr.ee/edsoncopque
. GitHub: https://github.com/ecopque
. Signal Messenger: ecop.01
```