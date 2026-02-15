# Early Stage Diabetes Risk Pediction

## Problem Statement
Building multiple modals to classify and predict the rist of early stage diabetes in soon to be diagnosed or normal patients. By analyzing the various featues in the dataset and health indicators like age, gender,polydispia and so on,, and hence model will predict / classify patients into "Psotive (in risk)" or "Negative" categories. 

## Dataset Description
* **Source:** [UCI Machine Learning Repository - Early Stage Diabetes Risk Prediction Dataset](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset)
* **Target Variable:** `class` (Positive/Negative)
* **Features:** The dataset consists of various predictor variables including Age, Gender, Polyuria, Polydipsia, Sudden weight loss, Weakness, Polyphagia, Genital thrush, Visual blurring, Itching, Irritability, Delayed healing, Partial paresis, Muscle stiffness, Alopecia, and Obesity.
* **Preprocessing:** Categorical variables were encoded using Label Encoding, and features were scaled using StandardScaler for distance-based algorithms.

## Models Used 
Following 6 Models were used for the classification purposes
1. **Logistic Regression** Linear model which uses sigmoid function 
2. **Decision Tree Classifier** Non-Linear model considers information gain at each branch level and splits data
3. **K-th Nearest Neighbor** Distance based algo that classifies according to the nearest data points
4. **Naive Bayes** Use probability classification based on Bayes theorem in Naive way
5. **Random Forest** Trains multiple descision trees and averages the predictions to reduce overfitting
6. **XGBoost** Builds trees sequentially and corrects the previous trees errors

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9423 | 0.9902 | 0.9833 | 0.9219 | 0.9516 | 0.8832 |
| **Decision Tree** | 0.9904 | 0.9922 | 1.0000 | 0.9844 | 0.9921 | 0.9800 |
| **KNN** | 0.9327 | 0.9697 | 0.9831 | 0.9062 | 0.9431 | 0.8653 |
| **Naive Bayes** | 0.9423 | 0.9863 | 0.9677 | 0.9375 | 0.9524 | 0.8800 |
| **Random Forest (Ensemble)** | 0.9808 | 0.9982 | 0.9841 | 0.9841 | 0.9841 | 0.9594 |
| **XGBoost (Ensemble)** | 0.9904 | 0.9996 | 1.0000 | 0.9844 | 0.9921 | 0.9800 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Provided a solid baseline with high recall, effectively identifying positive cases, though it struggled slightly with complex non-linear relationships compared to tree-based models. |
| **Decision Tree** | Achieved excellent accuracy and interpretability but showed a slight tendency towards overfitting compared to the Random Forest model. |
| **kNN** | Performance was highly dependent on feature scaling. It performed well but was computationally slower during inference compared to Logistic Regression. |
| **Naive Bayes** | Gave the lowest accuracy among the models, likely due to the strong independence assumption which may not hold true for correlated symptoms like Polyuria and Polydipsia. |
| **Random Forest (Ensemble)** | **Best Performer.** Achieved the highest accuracy and AUC score. The ensemble approach effectively reduced variance and handled feature interactions robustly. |
| **XGBoost (Ensemble)** | Performed nearly as well as Random Forest with very high precision and recall. It effectively handled the imbalanced aspects of the data through its boosting mechanism. |
