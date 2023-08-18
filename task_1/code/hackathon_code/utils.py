import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from typing import Optional
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
import catboost as cb
import xgboost as xgb
from Booking.const import requests_list, columns_to_drop, dummies

class Utils:
    def __init__(self):
        self.train_col_by_order = []
        self.train_col_avg_dict = {}

    def parse_countries(self, X: pd.DataFrame):
        CountriesGDP = \
            pd.read_csv("../Data/CountryGDP.csv", thousands=',').set_index(
                'Country').to_dict()['2018']
    
        X["guest_nationality_country_gdp"] = X[
            "guest_nationality_country_name"].map(CountriesGDP).astype(float)
        X["guest_nationality_country_gdp"].fillna(
            X["guest_nationality_country_gdp"].mean(), inplace=True)
        return
    
    def conversions(self, X: pd.DataFrame):
        X['is_first_booking'] = X['is_first_booking'].astype(int)
        X['is_user_logged_in'] = X['is_user_logged_in'].astype(int)
    
    def parse_special_requests(self, X: pd.DataFrame):
        X[requests_list] = X[requests_list].fillna(0)
        X['special_requests_number'] = X[requests_list].sum(axis=1)
    
    
    def parse_dates(self, X: pd.DataFrame):
        X['stay_length_in_nights'] = (
                    pd.to_datetime(X['checkout_date']) - pd.to_datetime(
                X['checkin_date'])).dt.days
        X['advanced_order_days'] = (
                    pd.to_datetime(X['checkin_date']) - pd.to_datetime(
                X['booking_datetime'])).dt.days
        X['live_to_check_in_days'] = (
                    pd.to_datetime(X['booking_datetime']) - pd.to_datetime(
                X['hotel_live_date'])).dt.days
        X['night_order'] = pd.to_datetime(X['booking_datetime']).dt.hour.isin(
            range(21, 24))
        X['night_order'] = X['night_order'].astype('int')
        X['weekend_vacation'] = (pd.to_datetime(
            X['checkin_date']).dt.dayofweek.isin(range(4, 7)) & pd.to_datetime(
            X['checkin_date']).dt.dayofweek.isin(range(4, 7))) & (
                                            X['stay_length_in_nights'] < 4)
        X['weekend_vacation'] = X['weekend_vacation'].astype('int')

    def remove_outliers(self, X: pd.DataFrame):
        X = X[X['no_of_adults'].isin(
            range(1, 13))]  # maybe change the 13, maybe simply >0
        X = X[X['no_of_children'].isin(range(0, 8))]  # maybe change, even > 0
        X = X[X['no_of_extra_bed'] >= 0]
        X = X[X['no_of_room'].isin(range(1, 9))]  # maybe change the upper bound
        X = X[X['original_selling_amount'] > 0]
        X = X[X['hotel_star_rating'].isin(
            np.arange(0, 5.5, 0.5))]  # hotel rates are between 0 to 5 in 0.5 steps
        X = X[X['stay_length_in_nights'] >= 0]
        X = X[X['advanced_order_days'] >= 0]
        X = X[X['live_to_check_in_days'] >= 0]
        return X
    
    def preprocess_test_data(self, test_X: pd.DataFrame):
        self.parse_dates(test_X)
        self.parse_special_requests(test_X)
        self.parse_countries(test_X)
        self.conversions(test_X)
        self.CreateCancellationFunc(test_X)
        test_X = test_X.drop(columns_to_drop, axis=1)
        test_X = test_X.fillna(self.train_col_avg_dict) # TODO: fill Na's by mean values - check if needed
        test_X = pd.concat([test_X, pd.get_dummies(test_X["original_payment_type"])], axis=1)
        test_X = pd.concat([test_X, pd.get_dummies(test_X["charge_option"])], axis=1)
        test_X = test_X.reindex(columns=self.train_col_by_order, fill_value=0)
        return test_X
    
    def CreateCancellationFunc(self, X: pd.DataFrame):
        for j in range(1, 5):
            X[f'D{j}'] = '-1'
            X[f'P{j}'] = '-1'
        X = self.update_D_P(X)
    
        return X
    
    def calc_train_col_avg(self, train_X: pd.DataFrame):
        self.train_col_by_order = train_X.columns.tolist()
        for col in self.train_col_by_order:
            self.train_col_avg_dict[col] = train_X[col].mean()
    
    def parse_dummies(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.get_dummies(X, prefix="payment_type",
                           columns=["original_payment_type"])
        X = pd.get_dummies(X, prefix="charge_option",
                           columns=["charge_option"])
        # X = pd.get_dummies(X, prefix="accommadation_type",
        #                    columns=["accommadation_type_name"]) # TODO: add
        return X
    
    def preprocess_train(self, X: pd.DataFrame, y: pd.Series):
        """
        Clean the train DataFrame from nan anomalous values.
        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Train DataFrame
        return: X: processed train DataFrame, y: response vector (prices)
        """
        # merging data
        merged_data = X
        merged_data['is_canceled'] = y
    
        # removing duplicates
        merged_data = merged_data.drop_duplicates()
    
        # removing impossible cases
        self.parse_dates(merged_data)
        merged_data = self.remove_outliers(merged_data)
        self.parse_special_requests(merged_data)
        self.parse_countries(merged_data)
        self.conversions(merged_data)
        self.CreateCancellationFunc(merged_data)
    
        # merged_data = parse_dummies(merged_data)
    
        res_y = merged_data.is_canceled
        res_X = merged_data.drop(columns=["is_canceled"])
        res_X = res_X.drop(columns_to_drop, axis=1)
        res_X = res_X.drop(dummies, axis=1)
        self.calc_train_col_avg(res_X)
    
        return res_X, res_y
    
    def adaboost_model(self, train_X, train_y, validation_X, validation_y):
        base_estimator = DecisionTreeClassifier(max_depth=1)
        adaboost = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100)
        adaboost.fit(train_X, train_y)
        pred_y = adaboost.predict(validation_X)
        print("Adaboost model with 100 estimators: ")
        print(classification_report(pred_y, validation_y))
    
    def random_forest_model(self, train_X, train_y, validation_X, validation_y):
        rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=25)
        rf_classifier.fit(train_X, train_y)
        pred_y = rf_classifier.predict(validation_X)
        print("Random Forest model with 200 trees with maxdepth = 25: ")
        print(classification_report(pred_y, validation_y))
    
    def logistic_regression(self, train_X, train_y, validation_X, validation_y):
        logreg = LogisticRegression()
        logreg.fit(train_X, train_y)
        pred_y = logreg.predict(validation_X)
        print("Logistic Regression model: ")
        print(classification_report(pred_y, validation_y))
    
    def decision_tree(self, train_X, train_y, validation_X, validation_y):
        dtc = DecisionTreeClassifier()
        dtc.fit(train_X, train_y)
        pred_y = dtc.predict(validation_X)
        print("Decision Tree model: ")
        print(classification_report(pred_y, validation_y))

    def catboost(self, train_X, train_y, validation_X, validation_y):
        cat_model = cb.CatBoostClassifier(iterations=100, learning_rate=0.1)
        cat_model.fit(train_X, train_y)
        pred_y = cat_model.predict(validation_X)
        print("CatBoost model with 100 iterations and learning rate of 0.1: ")
        print(classification_report(pred_y, validation_y))
    
    def xgBoost_model(self, train_X, train_y, validation_X):
        xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1)
        xgb_model.fit(train_X, train_y)
        pred_y = xgb_model.predict(validation_X)
        return pred_y, xgb_model

    def CancellationPolicyParser(self, policy, stay_len):
        if policy == "UNKNOWN":
            return policy
        cancellation_policy = []
        cancellation_dict = {}
        days_or_fine = ''
        D_flag = False
        for letter in policy:
            if not D_flag:
                if letter == 'P':
                    cancellation_dict['days_before_checkin'] = '0'
                    cancellation_dict['no_show_charge'] = days_or_fine
                    days_or_fine = ''
                elif letter == 'N':
                    cancellation_dict['days_before_checkin'] = '0'
                    cancellation_dict['no_show_charge'] = str(((int(days_or_fine)/int(stay_len))*100))
                    days_or_fine = ''
                elif letter == 'D':
                    D_flag = True
                    cancellation_dict['days_before_checkin'] = days_or_fine
                    days_or_fine = ''
                elif letter == '_':
                    if days_or_fine != '':
                        cancellation_policy.append(cancellation_dict)
                        cancellation_dict = {}
                    days_or_fine = ''
                else:
                    days_or_fine += letter
            else:
                if letter == 'P':
                    cancellation_dict['no_show_charge'] = days_or_fine
                if letter == 'N':
                    cancellation_dict['no_show_charge'] = str(((int(days_or_fine) / int(stay_len)) * 100))
                elif letter == "_":
                    D_flag = False
                    if days_or_fine != '':
                        cancellation_policy.append(cancellation_dict)
                        cancellation_dict = {}
                    days_or_fine = ''
                else:
                    days_or_fine += letter
        if len(cancellation_dict) > 0:
            cancellation_policy.append(cancellation_dict)
        return cancellation_policy
    
    
    def get_D1(self, code, stay_len):
        data = self.CancellationPolicyParser(code, stay_len)
        if data == 'UNKNOWN':
            return 0
        return data[0]['days_before_checkin']
    
    
    def get_D2(self, code, stay_len):
        data = self.CancellationPolicyParser(code, stay_len)
        if data == 'UNKNOWN' or len(data) < 2:
            return 0
        return data[1]['days_before_checkin']
    
    
    def get_D3(self, code, stay_len):
        data = self.CancellationPolicyParser(code, stay_len)
        if data == 'UNKNOWN' or len(data) < 3:
            return 0
        return data[2]['days_before_checkin']
    
    
    def get_D4(self, code, stay_len):
        data = self.CancellationPolicyParser(code, stay_len)
        if data == 'UNKNOWN' or len(data) < 4:
            return 0
        return data[3]['days_before_checkin']
    
    
    def get_P1(self, code, stay_len):
        data = self.CancellationPolicyParser(code, stay_len)
        if data == 'UNKNOWN':
            return 0
        return data[0]['no_show_charge']
    
    
    def get_P2(self, code, stay_len):
        data = self.CancellationPolicyParser(code, stay_len)
        if data == 'UNKNOWN' or len(data) < 2:
            return 0
        return data[1]['no_show_charge']
    
    
    def get_P3(self, code, stay_len):
        data = self.CancellationPolicyParser(code, stay_len)
        if data == 'UNKNOWN' or len(data) < 3:
            return 0
        return data[2]['no_show_charge']
    
    
    def get_P4(self, code, stay_len):
        data = self.CancellationPolicyParser(code, stay_len)
        if data == 'UNKNOWN' or len(data) < 4:
            return 0
        return data[3]['no_show_charge']
    
    
    def update_D_P(self, X: pd.DataFrame):
        X['D1'] = X.apply(lambda row: self.get_D1(row['cancellation_policy_code'], row['stay_length_in_nights']), axis=1).astype(float)
        X['D2'] = X.apply(lambda row: self.get_D2(row['cancellation_policy_code'], row['stay_length_in_nights']), axis=1).astype(float)
        X['D3'] = X.apply(lambda row: self.get_D3(row['cancellation_policy_code'], row['stay_length_in_nights']), axis=1).astype(float)
        X['D4'] = X.apply(lambda row: self.get_D4(row['cancellation_policy_code'], row['stay_length_in_nights']), axis=1).astype(float)
        X['P1'] = X.apply(lambda row: self.get_P1(row['cancellation_policy_code'], row['stay_length_in_nights']), axis=1).astype(float)
        X['P2'] = X.apply(lambda row: self.get_P2(row['cancellation_policy_code'], row['stay_length_in_nights']), axis=1).astype(float)
        X['P3'] = X.apply(lambda row: self.get_P3(row['cancellation_policy_code'], row['stay_length_in_nights']), axis=1).astype(float)
        X['P4'] = X.apply(lambda row: self.get_P4(row['cancellation_policy_code'], row['stay_length_in_nights']), axis=1).astype(float)
        return X
