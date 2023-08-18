import csv
import sys

import numpy as np
from sklearn.metrics import mean_squared_error

from Booking.task_1 import predict_cancellation
from Booking.task_2 import predict_price
from Booking.task_3 import plotBestFeatures
from Booking.utils import Utils as Utils1
from Booking.utils_task_2 import Utils as Utils2
import pandas as pd

from Submit.task_1.code.hackathon_code.task_4 import cancellationPolicy


def main(test_1_path, test_2_path):
    # --task 1:
    utils = Utils1()
    train_X = pd.read_csv("Data/agoda_cancellation_train.csv")
    train_y = train_X['cancellation_datetime'].notnull().astype(int)
    train_X = train_X.drop(['cancellation_datetime'], axis=1)
    cancellation = train_X['cancellation_policy_code']
    train_X_1, train_y_1 = utils.preprocess_train(train_X, train_y)
    test_X_1 = pd.read_csv(test_1_path)
    result_agoda_1, model = predict_cancellation(train_X_1, train_y_1,
                                                 test_X_1, utils)

    result_agoda_1.to_csv("task1.csv", index=False)

    # --task2
    utils2 = Utils2()
    train_y = train_X['original_selling_amount']
    train_X_2, train_y_2 = utils2.preprocess_train(train_X, train_y)
    test_X_2 = pd.read_csv(test_2_path)
    h_booking_id = test_X_2['h_booking_id']
    y_predicted = predict_price(train_X_2, train_y_2, test_X_2, utils2)
    test_X_2 = utils.preprocess_test_data(test_X_2)
    test_X_2["original_selling_amount"] = y_predicted
    cancel_prediction = model.predict(test_X_2)
    y_predicted[y_predicted <= 0] = y_predicted[y_predicted > 0].mean()
    y_predicted[cancel_prediction == 0] = -1

    result_agoda_2 = pd.DataFrame(
            {'ID': h_booking_id, 'predicted_selling_amount': y_predicted})
    result_agoda_2.to_csv("task2.csv", index=False)

    # --task3
    plotBestFeatures(train_X_1, train_y_1)

    # task 4:
    cancellationPolicy(train_X_1, train_y_1, cancellation)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])