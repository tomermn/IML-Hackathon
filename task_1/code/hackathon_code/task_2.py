import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures

from Booking.utils_task_2 import Utils


def KNeighborsRegressor_regression(train_X: pd.DataFrame,
                                   train_y: pd.DataFrame,
                                   validation_X: pd.DataFrame,
                                   validation_y: pd.DataFrame):
    _reg = KNeighborsRegressor(5, weights='distance')
    _reg.fit(train_X, train_y)
    y_predicted = _reg.predict(validation_X)
    err = mean_squared_error(validation_y, y_predicted, squared=False)
    print("KNeighbors:", err)


def lasso_regression(train_X: pd.DataFrame, train_y: pd.DataFrame,
                     validation_X: pd.DataFrame, validation_y: pd.DataFrame):
    _reg = Lasso(alpha=0.1)
    _reg.fit(train_X, train_y)
    y_predicted = _reg.predict(validation_X)
    err = mean_squared_error(validation_y, y_predicted, squared=False)
    print("lasso:", err)
    return y_predicted


def polynomial_regression(train_X: pd.DataFrame, train_y: pd.DataFrame,
                          validation_X: pd.DataFrame,
                          validation_y: pd.DataFrame):
    lin_regressor = LinearRegression()

    poly = PolynomialFeatures(3)
    X_transform = poly.fit_transform(train_X)
    lin_regressor.fit(X_transform, train_y)

    X_test_transform = poly.fit_transform(validation_X)
    y_predicted = lin_regressor.predict(X_test_transform)

    err = mean_squared_error(validation_y, y_predicted, squared=False)
    print("polynomial_regression:", err)


def polynomial_lasso_regression(train_X: pd.DataFrame, train_y: pd.DataFrame,
                                validation_X: pd.DataFrame,
                                validation_y: pd.DataFrame):
    lin_regressor = Lasso(alpha=0.1)

    poly = PolynomialFeatures(3)
    X_transform = poly.fit_transform(train_X)
    lin_regressor.fit(X_transform, train_y)

    X_test_transform = poly.fit_transform(validation_X)
    y_predicted = lin_regressor.predict(X_test_transform)

    err = mean_squared_error(validation_y, y_predicted, squared=False)
    print("polynomial_lasso_regression:", err)


def choose_price():
    utils = Utils()
    df = pd.read_csv("../Data/agoda_cancellation_train.csv")
    y = df['original_selling_amount']
    X = df.drop(['original_selling_amount', 'cancellation_datetime'], axis=1)

    train_X, v1_validation_X, train_y, validation_y = train_test_split(X, y,
                                                                       test_size=0.2,
                                                                       random_state=42)
    train_X, train_y = utils.preprocess_train(train_X, train_y)
    # validation_X, validation_y = preprocess_train(v1_validation_X, validation_y) # TODO: remove
    validation_X = utils.preprocess_test_data(v1_validation_X)

    # --models:
    KNeighborsRegressor_regression(train_X, train_y, validation_X,
                                   validation_y)
    y_predicted = lasso_regression(train_X, train_y, validation_X, validation_y)
    polynomial_regression(train_X, train_y, validation_X, validation_y)
    polynomial_lasso_regression(train_X, train_y, validation_X, validation_y)
    return y_predicted

def predict_price(train_X, train_y, test_X: pd.DataFrame, utils):
        h_booking_id = test_X['h_booking_id']
        test_X = utils.preprocess_test_data(test_X)
        return utils.lasso_regression_model(train_X, train_y, test_X)
