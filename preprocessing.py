import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif


def select_features_anova(X_train, y_train, k='all'):
    selector_anova = SelectKBest(score_func=f_classif, k=k)
    X_train_anova = selector_anova.fit_transform(X_train, y_train)
    selected_features_anova = X_train.columns[selector_anova.get_support()]
    selector_info = {
        'selector': selector_anova,
        'selected_features': selected_features_anova
    }
    with open('selector_anova.pkl', 'wb') as f:
        pickle.dump(selector_info, f)
        
    return X_train_anova

def extract_digit(column):
    extracted = []
    for string in column:
        parts = string.split()
        for part in parts:
            if part.isdigit():
                extracted.append(int(part))
                break
    return extracted

def extract(column):
    extracted = []
    for string in column:
        parts = string.split("-")
        for part in parts:
            if part.isdigit():
                extracted.append(int(part))
                break
    return extracted

def Label_Encoder_for_features(X, cols):
    feature_encoder = {}
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
        feature_encoder[c] = {
            'encoder': lbl,
            'classes': lbl.classes_.tolist() 
        }
    with open( 'feature_encoder.pkl', 'wb') as f:
        pickle.dump(feature_encoder, f)
    return X

def oneHotEncoding(X, cols):
    X_encoded= pd.get_dummies(X, columns=cols)
    new_columns = []
    for col in X_encoded.columns:
        if col not in X.columns:
            new_columns.append(col)
            
    with open('encoded_columns.pkl', 'wb') as f:
        pickle.dump(new_columns, f)
    
    return X_encoded



def label_encode_for_target(y):
    lbl = LabelEncoder()
    lbl.fit(y)
    y = lbl.transform(y)
   
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(lbl, f)
    
    return y

def replace_outliers_with_mean(X, columns):
    parameters = {}
    for column in columns:
        Q1_train = X[column].quantile(0.25)
        Q3_train = X[column].quantile(0.75)
        IQR_train = Q3_train - Q1_train
        lower_bound_train = Q1_train - 1.5 * IQR_train
        upper_bound_train = Q3_train + 1.5 * IQR_train
        mean_value_train = X[column].mean()
        
        parameters[column] = {
            'lower_bound': lower_bound_train,
            'upper_bound': upper_bound_train,
            'mean': mean_value_train
        }
        with open('outlier_parameters.pkl', 'wb') as params_file:
          pickle.dump(parameters, params_file)
        
        X.loc[X[column] < lower_bound_train, column] = mean_value_train
        X.loc[X[column] > upper_bound_train, column] = mean_value_train
   
    return X

def mean_normalize_data(X):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    return X_train_scaled

def skewness(X):
    skewness_params = {}
    for col in X.columns:
        skewness = X[col].skew()
        skewness_params[col] = skewness
        if skewness > 0.5:
            X[col] = np.log1p(X[col])
        elif skewness < -0.5:
            X[col] = X[col] ** 2
    with open('skewness_params.pkl', 'wb') as skew_file:
        pickle.dump(skewness_params, skew_file)    
    return X


def preprocess(X,Y):
    cols = ('msoffice','Touchscreen','weight','ram_type','processor_gnrtn','processor_name')
  
    X = Label_Encoder_for_features(X, cols)
    Y=label_encode_for_target(Y)
  
    columns = ["brand","processor_brand"]
    X = oneHotEncoding(X,columns)
  
    X['warranty'] = X['warranty'].replace("No warranty","0")
  
    digit_columns = ['warranty','ssd', 'hdd', 'graphic_card_gb', 'ram_gb']
    for col in digit_columns:
        if col in X.columns:
            X[col] = extract_digit(X[col])
  
    X['os'] = extract(X['os'])
  
    X = skewness(X)
    X = mean_normalize_data(X)
    X = replace_outliers_with_mean(X, X.columns)
    

    return X,Y