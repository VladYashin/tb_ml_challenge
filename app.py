import pandas as pd
import numpy as np
import pickle
import time
from raw_data.test_snippet import test_df
from models.init_model import Model

DF = 'DF'
listt = 'list'
strg = 'string'

# First run jupyter to get all the models (read README.doc) --> than choose appropriate model from /models
model = Model()


def start_app():
    print('Hello Bearbeiter! \n')
    df_string = input('DataFrame, List or single string? (DF/list/string)', )

    print('Model:', model.mlmodel)

    if df_string == listt or df_string == strg:
        query_text = input('Enter data: ', )
        print('Predicting... \n')
        label = model.predict(str(query_text))
        print('Label(s) for your input: \n', label, '\n')
        print('Executed')
    elif df_string == DF:
        print('1. Import df to app.py or try on the preloaded test_df\n'
              '2. Use: model.predict(model.fit(test_df, "label")) \n'
              '3. Smash enter button')
    else:
        print('Unknown data type for me')
        print('Try again!')


if __name__ == "__main__":
    start_app()
