import pandas as pd
from sklearn.model_selection import train_test_split
import csv
from utils import Utils

from Booking.utils import Utils

def predict_cancellation(train_X, train_y, test_X: pd.DataFrame, utils):
    h_booking_id = test_X['h_booking_id']
    test_X = utils.preprocess_test_data(test_X)
    pred_y, model = utils.xgBoost_model(train_X, train_y, test_X)


    result = pd.DataFrame({'h_booking_id': h_booking_id, 'cancellation': pred_y})
    with open("../../predictions/task1.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result)
    return result, model

def choose_classifier():
    utils = Utils()
    df = pd.read_csv("../Data/agoda_cancellation_train.csv")

    y = df['cancellation_datetime'].notnull().astype(int)
    X = df.drop(['cancellation_datetime'], axis=1)

    train_X, v1_validation_X, train_y, validation_y = train_test_split(X, y, test_size=0.2, random_state=42)
    train_X, train_y = utils.preprocess_train(train_X, train_y)
    # validation_X, validation_y = preprocess_train(v1_validation_X, validation_y) # TODO: remove
    validation_X = utils.preprocess_test_data(v1_validation_X)

    # --models:

    utils.adaboost_model(train_X, train_y, validation_X, validation_y)
    utils.random_forest_model(train_X, train_y, validation_X, validation_y)
    utils.logistic_regression(train_X, train_y, validation_X, validation_y)
    utils.decision_tree(train_X, train_y, validation_X, validation_y)
    utils.catboost(train_X, train_y, validation_X, validation_y)
    utils.xgBoost_model(train_X, train_y, validation_X, validation_y)