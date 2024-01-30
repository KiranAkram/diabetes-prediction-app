import pandas as pd
import xgboost as xgb
import pickle
from preprocessor import pipeline, preprocessor
from config import hyperparameters, threshold, feats, target
from joblib import dump, load
import joblib

import warnings


def get_clean_data(X):
    X = X.query("gender != 'Other'").reset_index(drop=True)
    X = X.drop_duplicates()

    y = X[target]

    # Fit and transform data using the pipeline
    transformed_data = pipeline.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    # Creating a DataFrame with the transformed data and feature names
    transformed_df = pd.DataFrame(
        transformed_data,
        columns=feature_names,
    )

    transformed_df[target] = y

    return transformed_df


def create_model(X):
    train_x = X[feats]
    train_y = X[target]

    # train the model
    model = xgb.XGBClassifier(**hyperparameters, objective="binary:logistic")
    model.fit(train_x, train_y)

    return model


def get_predictions(model, x):
    test_prob = model.predict_proba(x)[:, 1]
    label = [1 if i > threshold else 0 for i in test_prob]

    return label


def new_model():
    train_data = pd.read_csv("data/diabetes_prediction_dataset.csv")
    data = get_clean_data(train_data)  # if already saved just load it

    # to Save the cleaned data using joblib.dump
    # dump(data, "data/cleaned_data.joblib")

    model = create_model(data)

    # serialize model
    # with open("model.pkl", "wb") as f:
    # pickle.dump(model, f)
    return model


def get_test_data():
    test = load("data/test.joblib")
    test_data = get_clean_data(test)
    test_data = test_data[feats]

    return test_data


def main():
    # if model has not already been created, use this to create it
    model = new_model()

    # if model already created, just load it to use for predictions
    # model = joblib.load("model.pkl")

    test_data = get_test_data()

    # get predictions, since we are using threshold so,
    label = get_predictions(
        model, test_data
    )  # data will be the test_data we want to get predictions for
    print(label)


if __name__ == "__main__":
    main()
