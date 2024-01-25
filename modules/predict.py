import json
import os
import dill
import pandas as pd
from datetime import datetime


def predict():
    path = os.environ.get('PROJECT_PATH', '..')

    mod = sorted(os.listdir(f'{path}/data/models'))[-1]
    with open(f'{path}/data/models/{mod}', 'rb') as file:
        model = dill.load(file)

    df_pred = pd.DataFrame(columns=['id', 'predict'])

    test_list = os.listdir(f'{path}/data/test')

    for filename in test_list:
        with open(f'{path}/data/test/{filename}', 'r') as file2:
            form = json.load(file2)
        data = pd.DataFrame.from_dict([form])
        prediction = model.predict(data)

        dict_pred = {'id': data['id'].values[0], 'predict': prediction[0]}
        df = pd.DataFrame([dict_pred])
        df_pred = pd.concat([df, df_pred], ignore_index=True)

    now = datetime.now().strftime("%Y%m%d%H%M")
    df_pred.to_csv(f'{path}/data/predictions/{now}.csv', index=False)

    pass


if __name__ == '__main__':
    predict()
