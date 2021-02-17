import pandas as pd
from raw_data.test_snippet import test_df
from sklearn.preprocessing import LabelEncoder

from models.data_cleaning import text_cleaning, metrics_eval
import pickle
import os

# TfiDF Vectorizer
tfidf_path = os.path.dirname(os.path.realpath(__file__))
tfidf_file = open(os.path.join(tfidf_path, "tfidf_vectorizer"), 'rb')
tfidf = pickle.load(tfidf_file)
tfidf_file.close()


class Model(object):
    """ Custom class for ML model. To choose the model use string as input.
    Possible options: knn_classifier, knn_cv_classifier, rfc_classifier
    By default --> rfc_classifier """

    def __init__(self, mlmodel='rfc_classifier'):

        # Import model from pickle file
        model_path = os.path.dirname(os.path.realpath(__file__))
        model_file = open(os.path.join(model_path, mlmodel), 'rb')
        self.mlmodel = pickle.load(model_file)
        model_file.close()

    def predict(self, query):

        if type(query) == str:
            query_c = text_cleaning(query)
            query_c = [query_c]
            query_c = tfidf.transform(query_c).toarray()
        elif type(query) == list:
            query_c = list(map(text_cleaning, query))
            query_c = tfidf.transform(query_c)
        else:
            print('Still unknown data type for me!')

        y_predicted = self.mlmodel.predict(query_c)
        output_df = pd.DataFrame(data={'text': query, 'label': y_predicted}).replace(0, 'none').replace(1,
                                                                                                        'soft').replace(
            2, 'tech')

        return output_df

    def fit(self, data, label):

        """ Arguments:
        data: Pandas DataFrame
        label: Column with labels
        X: 1D  array
        y: 1D numpy array """

        df = pd.DataFrame(data=data)
        df[label] = LabelEncoder().fit_transform(df[label])
        X_data = df.drop(columns=[label])

        X_data = X_data.applymap(text_cleaning)
        X_data = X_data.values.tolist()
        X_data = [" ".join(x) for x in X_data]

        print('Model fitted')
        return X_data

    def get_metrics(self):

        """ Function to get metrics for specific model """

        metrics_eval()


# input_string = 'Ich habe gerne fuer euch zur Vefuegung die Kenntnisse der Kraftomnibussen und Python, Java, Pandas, Mikroskopie'
#
# model = Model()
# # ddata = model.fit(test_df, 'label')
# output = model.predict(input_string)
# print(output)
