import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import json
import os
import joblib

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def load_data(file_path, config):
    data = pd.read_csv(file_path)
    data = data[list(config['input_columns'].keys()) + [config['output_column']]]
    return data

def preprocess_data(data, config):
    encoder = OneHotEncoder()
    scaler = StandardScaler()
    
    for column, col_type in config['input_columns'].items():
        if col_type == 'categorical':
            encoded = encoder.fit_transform(data[[column]]).toarray()
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))
            data = pd.concat([data.drop(column, axis=1), encoded_df], axis=1)
        elif col_type == 'numerical':
            data[column] = scaler.fit_transform(data[[column]])

    if config['output_column'] in config['input_columns'] and config['input_columns'][config['output_column']] == 'numerical':
        data[config['output_column']] = scaler.fit_transform(data[[config['output_column']]])
    
    return data, scaler

def split_data(data, output_column, test_size=0.2, random_state=42):
    X = data.drop(output_column, axis=1)
    y = data[output_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_data(X_train, X_test, y_train, y_test, scaler, directory='data'):
    ensure_directory(directory)
    X_train.to_csv(f'{directory}/train_features.csv', index=False)
    y_train.to_csv(f'{directory}/train_labels.csv', index=False)
    X_test.to_csv(f'{directory}/test_features.csv', index=False)
    y_test.to_csv(f'{directory}/test_labels.csv', index=False)
    joblib.dump(scaler, f'{directory}/scaler.pkl')

def process_data(file_path, config_path):
    config = load_config(config_path)
    data = load_data(file_path, config)
    data, scaler = preprocess_data(data, config)
    X_train, X_test, y_train, y_test = split_data(data, config['output_column'])
    save_data(X_train, X_test, y_train, y_test, scaler)
    return X_train, X_test, y_train, y_test
