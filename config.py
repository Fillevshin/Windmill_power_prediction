import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DATASET_PATH = os.path.join(BASE_DIR, 'data', 'test_phase.csv')
TRAIN_DATASET_PATH = os.path.join(BASE_DIR, 'data', 'train_phase.csv')
PREPARED_DATASET_PATH = os.path.join(BASE_DIR, 'data', 'test_data_prepared.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'predict', 'catboost_model.pkl')
PREDICT_VALUES_PATH = os.path.join(BASE_DIR, 'data', 'predict_values.csv')
