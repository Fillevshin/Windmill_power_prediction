import warnings
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uvicorn
import os
from fastapi.responses import HTMLResponse

from config import BASE_DIR, INPUT_DATASET_PATH, PREPARED_DATASET_PATH, TRAIN_DATASET_PATH, \
                   PREDICT_VALUES_PATH
from preparation_data import clean_data, add_features
from predict.get_pedict import get_predict


app = FastAPI()


@app.get('/')
async def upload_file():
    with open(os.path.join(BASE_DIR, 'form.html'), 'r') as fd:
        html_form = fd.read()
    return HTMLResponse(html_form)


@app.post("/read_and_predict/")
async def read_and_predict(file: UploadFile = File(...)):
    with open(file.filename, 'wb') as f:
        f.write(await file.read())
    os.rename(file.filename, os.path.join(BASE_DIR, 'data', 'test_phase.csv'))
    clean_data.clean_data(input_path=INPUT_DATASET_PATH, output_path=PREPARED_DATASET_PATH)
    add_features.add_features(dataset_path=PREPARED_DATASET_PATH,
                              dataset_path_groupby=TRAIN_DATASET_PATH)
    warnings.filterwarnings("ignore", category=UserWarning)
    get_predict(dataset_path=PREPARED_DATASET_PATH)
    return FileResponse(PREDICT_VALUES_PATH, media_type='text/csv', filename='predict_values.csv')


if __name__ == '__main__':
    uvicorn.run(app='main:app', host='localhost', port=7777, reload=True)
