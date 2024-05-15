import pickle
import numpy as np
import pandas as pd
from  preprocessing import extract,extract_digit
from sklearn.metrics import accuracy_score

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    

def label_encoder_for_features(X):
    with open('feature_encoder.pkl', 'rb') as f:
        feature_encoder = pickle.load(f)
        
    for column, info in feature_encoder.items():
        lbl = info['encoder']
        transformed_items = []
        
        for item in X[column]:
            if item in lbl.classes_:
                transformed_item = lbl.transform([item])[0]
            else:
                 transformed_item = max(lbl.transform(lbl.classes_)) + 1
            transformed_items.append(transformed_item)
        
        X[column] = transformed_items
    return X

def normalize_with_scaler(X):
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    X_normalized = scaler.transform(X)
    X_normalized = pd.DataFrame(X_normalized, columns=X.columns)
    return X_normalized

def skewness(X):
    with open('skewness_params.pkl', 'rb') as skew_file:
        skewness_params = pickle.load(skew_file)
    for col in X.columns:
        skewness = skewness_params.get(col, 0)
        if skewness > 0.5:
            X[col] = np.log1p(X[col])
        elif skewness < -0.5:
            X[col] = X[col] ** 2
    return X

def OneHotEncoding(X):
    with open('encoded_columns.pkl', 'rb') as f:
        encoder_info = pickle.load(f)
        
    columns = ["brand","processor_brand"]
    X_encoded= pd.get_dummies(X, columns=columns)
    
    new_columns = []
    for col in X_encoded.columns:
        if col not in X.columns:
            new_columns.append(col)
            
    if set(encoder_info) != set(new_columns):
        for col in new_columns:
           if col not in encoder_info:
              X_encoded.drop(columns=[col], inplace=True)

    return X_encoded


def replace_outliers_with_mean(X):
    with open('outlier_parameters.pkl', 'rb') as params_file:
        outlier_params = pickle.load(params_file)
    for column, params in outlier_params.items():
        mean_value = params['mean']
        lower_bound = params['lower_bound']
        upper_bound = params['upper_bound']
        X.loc[X[column] < lower_bound, column] = mean_value
        X.loc[X[column] > upper_bound, column] = mean_value
    return X

def label_encoder_for_target(y):
    with open('label_encoder.pkl', 'rb') as f:
        lbl = pickle.load(f)
    y = lbl.transform(y)
    return y

def feature_selector(X):
    with open('selector_anova.pkl', 'rb') as f:
        selector_info = pickle.load(f)
    selector = selector_info['selector']
    selected_features = selector_info['selected_features']
    X = selector.transform(X[selected_features])

    return X

def fill_missing_with_mode(X):
    with open('train_modes.pkl', "rb") as f:
        modes_dict = pickle.load(f)
        
    missing_mask = X.isna()
    for i in range(X.shape[1]):
        for col in X.columns:
            X[col] = np.where(missing_mask[col], modes_dict[i], X[col])
    return X

def preprocessing(X,Y):
    # X.iloc[1, 0] = np.nan
    # print(X.isna().sum())
    X=fill_missing_with_mode(X)
    # print(X.isna().sum())
    
    # X.iloc[1, 2] = 'marwa'
    # X.iloc[3, 2] = 'marwan'
    # X.to_excel('check.xlsx', index=False)
    
    X = label_encoder_for_features(X)
    
    Y = label_encoder_for_target(Y)
 
    X = OneHotEncoding(X)
    
    # X.to_excel('check2.xlsx', index=False)
    
    if(X['warranty']=='No warranty').any():
      X['warranty'] = X['warranty'].replace("No warranty","0")
      
    digit_columns = ['warranty','ssd', 'hdd', 'graphic_card_gb', 'ram_gb']
    for col in digit_columns:
       if col in X.columns:
            X[col] = extract_digit(X[col])
    X['os'] = extract(X['os'])

    X =skewness(X)

    X = normalize_with_scaler(X)
    
    X= replace_outliers_with_mean(X)
        
    return X,Y    

def evaluate_models(X,Y,path,name):
    loaded_model = load_model(path)   
    prediction = loaded_model.predict(X)
    
    accuracy = accuracy_score(Y,prediction)
    print("Accuracy of " +name +' : '+ str(round(accuracy * 100, 2))+'%'+'\n')

def predict(x,y):
    X,Y=preprocessing(x,y)
    
    X=feature_selector(X)
    
    evaluate_models(X,Y,'random_forest_model.pkl','random_forest')
    evaluate_models(X,Y,'svm_model.pkl','SVM')
    evaluate_models(X,Y,'svm_linear_model.pkl','SVM_Linear')
    evaluate_models(X,Y,'svm_Polynomial_model.pkl','SVM_Polynomial')
    evaluate_models(X,Y,'svm_RBF_model.pkl','SVM_RBF')
    evaluate_models(X,Y,'knn_model.pkl','KNN')
    evaluate_models(X,Y,'Logistic_model.pkl','Logistic')
    evaluate_models(X,Y,'decision_tree_model.pkl','Decision_Tree')

with open('x_test.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open('Y_test.pkl', 'rb') as f:
    Y_test = pickle.load(f)  


predict(X_test,Y_test)

##################<for exam>##################
# test_data = pd.read_csv("Unseen_data.csv") 
# y=test_data['rating']
# x =test_data.drop(columns=["rating"])

# predict(x,y)
