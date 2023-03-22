import pickle
import pandas as pd
import numpy as np
import catboost

from config import MODEL_PATH, PREDICT_VALUES_PATH


def exp_func(x: float) -> float:
    """
    Экспоненциальное преобразование обратное логарифмическому.
    Испльзуется для преобразования целевой переменной для лучшего предсказания.
    Params:
    x (float): вещественное число
    return float
    """
    return np.exp(x) - 1


def get_predict(dataset_path: str) -> None:
    """
    Загружает сохраненную ранее модель. Предсказывает значения для новых данных.
    Сохраняет предсказанные значения в файл.
    Params:
    dataset_path (str): путь к файлу с данными
    return None
    """
    data = pd.read_csv(dataset_path, sep=',')
    with open(MODEL_PATH, 'rb') as fd:
        load_model = pickle.load(fd)
    data_frame = pd.DataFrame(data.date, columns=['date'])
    data_frame['wp1'] = pd.Series(load_model.predict(data.iloc[:, 1:]))
    data_frame['wp1'] = data_frame['wp1'].apply(exp_func)
    data_frame.to_csv(PREDICT_VALUES_PATH, index=False)
