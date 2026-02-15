from sklearn.ensemble import RandomForestClassifier
from my_util import load_and_preprocess_data, return_evaluation_metrics

def main():
    X_train , X_test , y_train , y_test = load_and_preprocess_data()
    model = RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:,1]
    except IndexError:
        y_prob = None
    print(return_evaluation_metrics("Naive Bayes Classification", y_test ,y_pred,y_prob))
if __name__ == "__main__":
    main()