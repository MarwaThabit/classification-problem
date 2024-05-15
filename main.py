import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import *
from models import *
import pickle
# from Test import *


def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
 
def save_train_modes(x):
   X=pd.DataFrame(x) 
   modes = X.mode().iloc[0]

   modes_dict = modes.to_dict()
   with open('train_modes.pkl', "wb") as f:
        pickle.dump(modes_dict, f) 
        
def main():
    data = pd.read_csv("ElecDeviceRatingPrediction_Milestone2.csv")
    
    Y = data["rating"]
    X = data.drop(columns=["rating"])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False,random_state=0)
       
    X_train,y_train= preprocess(X_train,y_train)
    
    
    X_train = select_features_anova(X_train,y_train)
    
    random_forest_model=random_forest_classifier(X_train,y_train)
   
    
    save_model(random_forest_model, 'random_forest_model.pkl')
    
    SVM_model=svm_classifier(X_train,y_train)
    save_model(SVM_model, 'svm_model.pkl')
    
    SVM_Linear_model=svm_linear_classifier(X_train,y_train)
    save_model(SVM_Linear_model, 'svm_linear_model.pkl')
    
    SVM_Polynomial_model=svm_polynomial_classifier(X_train,y_train)
    save_model(SVM_Polynomial_model, 'svm_Polynomial_model.pkl')
    
    SVM_RBF_model=svm_rbf_classifier(X_train,y_train)
    save_model(SVM_RBF_model, 'svm_RBF_model.pkl')
    
    KNN_model=knn_classifier(X_train,y_train)
    save_model(KNN_model, 'knn_model.pkl')
    
    DT_model=decision_tree_classifier(X_train,y_train)
    save_model(DT_model, 'decision_tree_model.pkl')
    
    Logistic_model=logistic_regression_classifier(X_train,y_train)
    save_model( Logistic_model, 'Logistic_model.pkl')
    
    with open('x_test.pkl', 'wb') as f:
      pickle.dump(X_test, f)

    with open('Y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
        
    save_train_modes(X_train)
 
main()


