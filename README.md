# Прогнозирование мощности ветрогенератора

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYzNjNDM1MjljZTQ0NzdjNzg3ZDdmOWViYmY3YTQ3ODA2M2QzMTYzMCZjdD1n/OnavQV9tvZVks/giphy.gif" alt="My Project GIF" width="900" height="500">

Данный проект имеет целью создание API-приложения для прогнозирования мощности ветрогенератора на основе данных, полученных с помощью кейса [соревнования](https://datascience.totalenergies.com/fr/challenge/19/details#tab_brief84).  

## Исследование данных и создание модели

В первой части проекта проводилось исследование датасета, его обработка, создание новых признаков, нахождение наилучшей модели для прогнозирования и её сохранение. Результаты данной работы представлены в файле [research_part.ipynb](https://github.com/Fillevshin/Windmill_power_prediction/blob/main/research_part.ipynb).  

## FastAPI приложение

Во второй части проекта было написано приложение на основе FastAPI. В HTML-форме оно принимает на вход необработанный датасет в формате CSV, дополняет его необходимыми для прогнозирования признаками, загружает сохраненную ранее модель и на основе неё генерирует датасет с прогнозами мощности. Данные прогнозов автоматически скачиваются в формате CSV.  

## Запуск проекта из Docker

Вы можете загрузить image по данному запросу:  

```docker pull fillevshin/windmill_power_prediction```  

Для запуска контейнера на 80 порту используйте:  

```docker run -p 80:80 fillevshin/windmill_power_prediction```

Затем загрузите датасет из [облака](https://disk.yandex.ru/d/li_YKGGwyCZV5Q) в форму.

## Запуск проекта из командной строки

Для запуска проекта из командной строки выполните следующие шаги:  

1. Клонируйте репозиторий с помощью команды:  
    
```git clone https://github.com/fillevshin/windmill_power_prediction.git```

2. Установите и активируйте виртуальное окружение:  

```python -m venv venv```  
```source venv/Scripts/activate``` (для Windows OS)  
или  
```source venv/bin/activate``` (для Unix OS)  

3. Установите необходимые зависимости из файла requirements.txt:  
    
```pip install -r requirements.txt```  

4. Запустите проект:

```python main.py```  

5. Перейдите по ссылке http://localhost:7777/

6. Загрузите датасет из [облака](https://disk.yandex.ru/d/li_YKGGwyCZV5Q) в форму.
