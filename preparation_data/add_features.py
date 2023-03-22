import pandas as pd
import numpy as np


def groupby_feature_df(x_groupby: pd.DataFrame, x_merge: pd.DataFrame, cat_feature: str,
                       feature: str) -> pd.DataFrame:
    """
    Создает pd.DataFrame из X_groupby с новой сгруппированной фичей.
    Params:
    X_groupby (pd.DataFrame): датафрейм из которого берем среднее
    X_merge (pd.DataFrame): датафрейм в который добавляем среднее
    cat_feature (str): фича по кооторой группируем
    feature (str): фича по которой берем среднее
    return pd.DataFrame
    """
    groupby_feature = x_groupby.groupby(cat_feature)[feature].mean().rename(
                                                                f'mean_{cat_feature}_{feature}')
    return x_merge.merge(groupby_feature, on=cat_feature, how='inner')


def add_basic_features(data: pd.DataFrame) -> pd.DataFrame:
    """"
    Добавляет фичи в датасет
    Params:
    data (pd.DataFrame): датасет с которым работаем
    return pd.DataFrame
    """
    data['date'] = pd.to_datetime(data.date)
    data['month'] = data['date'].dt.month
    data['quarter'] = data['date'].dt.quarter
    data['hour'] = data['date'].dt.hour
    data['log_ws'] = np.log(data.ws + 1)
    data['wd_direction'] = data.wd.apply(lambda x: 1 if x >= 338 and x < 360 else (
        2 if x >= 23 and x < 68 else (3 if x >= 68 and x < 113 else (
            4 if x >= 113 and x < 158 else (5 if x >= 158 and x < 203 else (
                6 if x >= 203 and x < 248 else (
                    7 if x >= 248 and x < 293 else (8 if x >= 293 and x < 338 else 1))))))))
    data['power_of_direction'] = data.wd_direction.apply(
        lambda x: 1 if x in [5, 6] else (3 if x in [2, 3] else 2))
    data['wp_is_max'] = data.wd_direction.apply(lambda x: 1 if x in [2, 3] else 0)
    data['alpha'] = data.wd.apply(lambda x: x - 90 if x > 90.0 else 360 + x - 90)
    data['cos_aplha'] = np.cos(data.alpha)
    data['sin_aplha'] = np.sin(data.alpha)
    data['abs_u'] = data.u.apply(abs)
    data['gorizont'] = data.u.map(lambda x: 'e' if x >= 0 else 'w')
    data['abs_v'] = data.v.apply(abs)
    data['vertik'] = data.v.map(lambda x: 'n' if x >= 0 else 's')
    data['direction_u_v'] = (data.vertik + data.gorizont).apply(
        lambda x: 1 if x == 'nw' else (2 if x == 'ne' else (3 if x == 'se' else 4)))
    data['gorizont'] = data.u.map(lambda x: 1 if x == 'e' else 0)
    data['vertik'] = data.v.map(lambda x: 1 if x == 'n' else 0)
    return data


def add_features(dataset_path: str, dataset_path_groupby: str) -> None:
    """
    Выполненяет функции add_basic_features и groupby_feature_df. Тем самым
    добавляет фичи в датасет и добавляет сгрупированные фичи.  Сохраняет датасет в dataset_path.
    Params:
    dataset_path (str): путь к датасет которому нужно добавить фичи
    dataset_path_groupby (str): путь к датасету по которому будем группировать
    return  None
    """
    train_data = pd.read_csv(dataset_path_groupby, sep=',')
    test_data = pd.read_csv(dataset_path, sep=',')
    train_data = add_basic_features(train_data)
    test_data = add_basic_features(test_data)
    cat_features_goupby = ["month", "hour", "quarter", "wd_direction", "direction_u_v"]
    for cat_feature in cat_features_goupby:
        test_data = groupby_feature_df(x_groupby=train_data,
                                       x_merge=test_data,
                                       cat_feature=cat_feature,
                                       feature='wp1')
    for cat_feature in cat_features_goupby:
        test_data = groupby_feature_df(x_groupby=test_data,
                                       x_merge=test_data,
                                       cat_feature=cat_feature,
                                       feature='ws')
    test_data.to_csv(dataset_path, index=False)
