from sklearn.linear_model import LogisticRegression
from my_util import load_and_preprocess_data , return_evaluation_metrics

def main():
    X_train , X_test , y_train , y_test = load_and_preprocess_data()
    model = LogisticRegression(random_state=42,max_iter=5000)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    print(return_evaluation_metrics("Logistic Regression",y_test,y_pred,y_prob))

if __name__ == "__main__":
    main()
