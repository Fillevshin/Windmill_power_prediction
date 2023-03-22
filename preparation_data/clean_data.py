import pandas as pd


def clean_data(input_path: str, output_path: str) -> None:
    """
    Очистка данных от пустых значений и преобразование даты в формат datetime.
    Сохраняет преобразоыванные данные в файл.
    Params:
    input_path (str): путь к файлу с данными
    return None
    """
    data = pd.read_csv(input_path, sep=',')
    data.dropna(axis=0, how='all', inplace=True)
    data['date'] = pd.to_datetime(data.date)
    data.to_csv(output_path, index=False)
