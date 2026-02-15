from xgboost import XGBClassifier
from my_util import load_and_preprocess_data, return_evaluation_metrics

def main():
    X_train , X_test , y_train , y_test = load_and_preprocess_data()
    model = XGBClassifier(n_estimators=100 ,learning_rate=0.1,random_state=42,use_label_encoder=False,eval_metric='logloss')

    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:,1]
    except IndexError:
        y_prob = None
    print(return_evaluation_metrics("XGBoost", y_test ,y_pred,y_prob))
if __name__ == "__main__":
    main()