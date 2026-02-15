import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import( accuracy_score ,roc_auc_score , precision_score , recall_score, f1_score, matthews_corrcoef)


def load_and_preprocess_data(filepath='diabetes_data_upload.csv'):
    target_col = 'class'
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"{filepath} doesn't exist!!")
    
    df.column = df.columns.str.strip()

    label_encoder = LabelEncoder()
    object_columns = df.select_dtypes(include=['object']).columns
    print(f"Encoding Following columns : {object_columns.to_list()}")
    for col in object_columns:
            df[col] = label_encoder.fit_transform(df[col].astype(str))

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2 ,random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train ,X_test, y_train, y_test

def return_evaluation_metrics(model_name, y_test , y_pred, y_prob=None):
    accuracy = accuracy_score(y_test,y_pred)
    if y_prob is not None:
        auc = roc_auc_score(y_test,y_prob)
    else:
        auc = roc_auc_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    mcc = matthews_corrcoef(y_test,y_pred)
    return {"accuracy" : f"{accuracy:.4f}",
            "auc" : f"{auc:.4f}",
            "precision" : f"{precision:.4f}",
            "recall" : f"{recall:.4f}",
            "f1" : f"{f1:.4f}",
            "mcc" : f"{mcc:.4f}"}