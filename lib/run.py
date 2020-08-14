from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np
import nltk
import re

if __name__ == '__main__':
#Loading data
    df_train = pd.read_csv('data/val.csv')
    df_train.drop(['city','subcategory', 'category','price','region', 'datetime_submitted'], axis='columns', inplace=True)

    df_test = pd.read_csv('/task-for-hiring-data/test_data.csv')
    df_test.drop(['city','subcategory', 'category','price','region', 'datetime_submitted'], axis='columns', inplace=True)

#Preparing data
    def standardize_text(df, text_field):
        df[text_field] = df[text_field].str.replace(r"[^А-яа-я0-9(),!?@\'\`\"\_\n]", " ")
        df[text_field] = df[text_field].str.lower()
        return df

    clean_data_train = standardize_text(df_train, "description")
    clean_data_test = standardize_text(df_test, "description")

    tokenizer = RegexpTokenizer(r'\w+')
    clean_data_train["tokens"] = clean_data_train["description"].apply(tokenizer.tokenize)
    clean_data_test["tokens"] = clean_data_test["description"].apply(tokenizer.tokenize)

    def cv(data):
        count_vectorizer = CountVectorizer()
        emb = count_vectorizer.fit_transform(data)
        return emb, count_vectorizer

    X_train = clean_data_train["description"].tolist()
    y_train = clean_data_train["is_bad"].tolist()
    X_test = clean_data_test["description"].tolist()
    y_test = clean_data_test["is_bad"].tolist()

    X_train_counts, count_vectorizer = cv(X_train)
    X_test_counts = count_vectorizer.transform(X_test)

#Creating model
    model = LogisticRegression()
    c_values = np.logspace(-2, 2, 10)
    grid = GridSearchCV(estimator=model, param_grid={'C': c_values}, scoring='roc_auc', cv = 4, n_jobs=-1, verbose=1)

    grid.fit(X_train_counts, y_train)

#Prediction and save results
    target_prediction = pd.DataFrame(grid.predict_proba(X_test_counts)[:, 1]).reset_index()
    target_prediction.columns = ["index", "prediction"]
    target_prediction.to_csv('/task-for-hiring-data/target_prediction.csv', index=False)