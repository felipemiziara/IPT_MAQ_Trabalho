import pandas as pd
import os

# Definindo os caminhos dos arquivos diretamente no script
TRAIN_FEATURES = 'data/train_features.csv'
TRAIN_LABELS = 'data/train_labels.csv'
TEST_FEATURES = 'data/test_features.csv'
TEST_LABELS = 'data/test_labels.csv'

def check_file_exists(file_path):
    """Verifica se um arquivo existe no caminho especificado."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado. Por favor, verifique se os dados foram carregados e normalizados corretamente.")

def load_train_features():
    """Carrega e retorna os dados de características de treinamento."""
    check_file_exists(TRAIN_FEATURES)
    return pd.read_csv(TRAIN_FEATURES)

def load_train_labels():
    """Carrega e retorna os dados de labels de treinamento."""
    check_file_exists(TRAIN_LABELS)
    data = pd.read_csv(TRAIN_LABELS)
    if len(data.columns) == 1:
        return data.squeeze()
    return data

def load_test_features():
    """Carrega e retorna os dados de características de teste."""
    check_file_exists(TEST_FEATURES)
    return pd.read_csv(TEST_FEATURES)

def load_test_labels():
    """Carrega e retorna os dados de labels de teste."""
    check_file_exists(TEST_LABELS)
    data = pd.read_csv(TEST_LABELS)
    if len(data.columns) == 1:
        return data.squeeze()
    return data
