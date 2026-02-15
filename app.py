import streamlit as sl
import pandas as pd
import matplotlib.pyplot as mpl
import seaborn as sea
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import( accuracy_score ,roc_auc_score , precision_score , recall_score, f1_score, matthews_corrcoef, confusion_matrix)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


sl.set_page_config(page_title="Diabetes Risk Classifier",layout="wide")
sl.title("Early Stage Diabetes Risk Pediction")
sl.markdown("""
This app lets users to upload the dataset csv file , train using different ML models and showcase the evaluation metrics and performance
""")

sl.sidebar.header("Config")
uploaded_file = sl.sidebar.file_uploader("Upload CSV file with data", type=["csv"])
model_name = sl.sidebar.selectbox("Choose Model",["XGBoost","Log Regression","K-th Nearest Neighbours","Naive Bayes","Random Forest","Descision Tree"])
sl.sidebar.markdown("---")
sl.sidebar.write("Model Settings")

if model_name == "K-th Nearest Neighbours":
    k_neighbors = sl.sidebar.slider("Number of Neighbors (K)",1,15,5)
if model_name == "Random Forest":
    n_estimators = sl.sidebar.slider("Number of Trees",10,200,100)
if model_name == "XGBoost":
    lr = sl.sidebar.slider("Learning Rate",0.01,0.5,0.1)

@sl.cache_data
def processing_data(file):
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        target_col = 'class'
        if target_col not in df.columns:
            return None,None,None,None,f"Error : Dataset Must contain '{target_col}' column."
        label_encoder = LabelEncoder()
        object_columns = df.select_dtypes(include=['object']).columns
        for col in object_columns:
            df[col] = label_encoder.fit_transform(df[col].astype(str))
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2 ,random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train ,X_test, y_train, y_test , None
    except Exception as e:
        return None, None,None,None, str(e)
    
if uploaded_file is not None:
    X_train , X_test, y_train , y_test, error = processing_data(uploaded_file)
    if error:
        sl.error(error)
    else:
        sl.success("Data is loaded now, and pre processed")

        if model_name == "Log Regression":
            model = LogisticRegression(random_state=42, max_iter=5000)
        elif model_name == "Descision Tree":
            model = DecisionTreeClassifier(random_state=42, criterion='entropy')
        elif model_name == "K-th Nearest Neighbours":
            k = k_neighbors
            model = KNeighborsClassifier(n_neighbors=k)
        elif model_name == "Naive Bayes":
            model = GaussianNB()
        elif model_name == "Random Forest":
            est = n_estimators
            model = RandomForestClassifier(n_estimators=est,random_state=42)
        elif model_name == "XGBoost" :
            model = XGBClassifier(n_estimators=100,learning_rate=lr,random_state=42,use_label_encoder=False,eval_metric='logloss')
        
        if sl.button("Evaluate") :
            with sl.spinner(f"Training {model_name}"):
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                try:
                    y_prob = model.predict_proba(X_test)[:,1]
                except:
                    y_prob = None
            sl.divider()
            sl.subheader(f"Results / Performance for {model_name}")
            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred,zero_division=0)
            recall = recall_score(y_test,y_pred,zero_division=0)
            f1 = f1_score(y_test,y_pred,zero_division=0)
            mcc = matthews_corrcoef(y_test,y_pred)
            try:
                auc_score = roc_auc_score(y_test,y_prob) if y_prob is not None else roc_auc_score(y_test,y_pred)
            except:
                auc_score =0.0
            m1,m2,m3 = sl.columns(3)
            m4,m5,m6 = sl.columns(3)
            m1.metric("Accuracy", f"{accuracy:.4f}")
            m2.metric("Precision" , f"{precision:.4f}")
            m3.metric("Recall" , f"{recall:.4f}")
            m4.metric("F1 Score" , f"{f1:.4f}")
            m6.metric("MCC Score" , f"{mcc:.4f}")
            m5.metric("AUC Score" , f"{auc_score:.4f}")
            sl.divider()
            col_graph,col_text = sl.columns([1,1])
            with col_graph:
                sl.write("#### Confusion Matrix")
                cm = confusion_matrix(y_test,y_pred)
                fig,ax = mpl.subplots(figsize=(6,5))
                sea.heatmap(cm,annot=True,fmt='d',cmap="Blues",ax=ax)
                ax.set_ylabel('Actual Label')
                ax.set_xlabel('Prediction Label')
                sl.pyplot(fig)
else:
    sl.info("Please upload the CSV file to start")     

