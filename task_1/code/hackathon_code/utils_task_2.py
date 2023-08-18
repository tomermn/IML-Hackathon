import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Lasso
import catboost as cb
import xgboost as xgb
from Booking.const import requests_list, dummies


class Utils:
    columns_to_drop = ['origin_country_code',
                       'language',
                       'original_payment_method',
                       'request_nonesmoke',
                       'request_latecheckin',
                       'request_highfloor',
                       'request_largebed',
                       'request_twinbeds',
                       'request_airport',
                       'request_earlycheckin',
                       'original_payment_currency',
                       'hotel_brand_code',
                       'hotel_chain_code',
                       'hotel_city_code',
                       'hotel_chain_code',
                       'hotel_country_code',
                       'hotel_area_code',
                       'h_customer_id',
                       'customer_nationality',
                       'guest_nationality_country_name',
                       "checkout_date",
                       "checkin_date",
                       "booking_datetime",
                       "h_booking_id",
                       "hotel_id",
                       "hotel_country_code",
                       "guest_is_not_the_customer",
                       "is_user_logged_in",
                       "is_first_booking",
                       "cancellation_policy_code",
                       "hotel_live_date",
                       "accommadation_type_name"]

    def __init__(self):
        self.train_col_by_order = []
        self.train_col_avg_dict = {}

    def parse_countries(self, X: pd.DataFrame):
        CountriesGDP = \
            pd.read_csv("../Data/CountryGDP.csv", thousands=',').set_index(
                'Country').to_dict()['2018']
        CountriesCodeGDP = \
            pd.read_csv("../Data/CountryGDP.csv", thousands=',').set_index(
                'Code').to_dict()['2018']

        X["guest_nationality_country_gdp"] = X[
            "guest_nationality_country_name"].map(CountriesGDP).astype(float)
        X["guest_nationality_country_gdp"].fillna(
            X["guest_nationality_country_gdp"].mean(), inplace=True)

        X["hotel_GDP"] = X["hotel_country_code"].map(CountriesCodeGDP).astype(
            float)
        X["hotel_GDP"].fillna(X["hotel_GDP"].mean(), inplace=True)
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
        X = X[
            X['no_of_room'].isin(range(1, 9))]  # maybe change the upper bound
        X = X[X['original_selling_amount'] > 0]
        X = X[X['hotel_star_rating'].isin(
            np.arange(0, 5.5,
                      0.5))]  # hotel rates are between 0 to 5 in 0.5 steps
        return X

    def preprocess_test_data(self, test_X: pd.DataFrame):
        self.parse_dates(test_X)
        self.parse_special_requests(test_X)
        self.parse_countries(test_X)
        self.conversions(test_X)
        self.CreateCancellationFunc(test_X)
        self.normalize_values(test_X)
        test_X = test_X.drop(Utils.columns_to_drop, axis=1)
        test_X = test_X.fillna(
            self.train_col_avg_dict)  # TODO: fill Na's by mean values - check if needed
        test_X = pd.concat(
            [test_X, pd.get_dummies(test_X["original_payment_type"])], axis=1)
        test_X = pd.concat([test_X, pd.get_dummies(test_X["charge_option"])],
                           axis=1)
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

    def normalize_values(self, test_X: pd.DataFrame):
        scaler = MinMaxScaler(feature_range=(0, 10))
        test_X["hotel_GDP"] = scaler.fit_transform(test_X[["hotel_GDP"]])
        test_X["live_to_check_in_days"] = scaler.fit_transform(
            test_X[["live_to_check_in_days"]])
        test_X["advanced_order_days"] = scaler.fit_transform(
            test_X[["advanced_order_days"]])
        test_X["stay_length_in_nights"] = scaler.fit_transform(
            test_X[["stay_length_in_nights"]])
        test_X["no_of_adults"] = scaler.fit_transform(test_X[["no_of_adults"]])

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
        merged_data['original_selling_amount'] = y

        # removing duplicates
        merged_data = merged_data.drop_duplicates()

        # removing impossible cases
        self.parse_dates(merged_data)
        merged_data = self.remove_outliers(merged_data)
        self.parse_special_requests(merged_data)
        self.parse_countries(merged_data)
        self.conversions(merged_data)
        self.CreateCancellationFunc(merged_data)
        self.normalize_values(merged_data)
        # merged_data = parse_dummies(merged_data)

        res_y = merged_data.original_selling_amount
        res_X = merged_data.drop(columns=["original_selling_amount"])
        res_X = res_X.drop(Utils.columns_to_drop, axis=1)
        res_X = res_X.drop(dummies, axis=1)
        self.calc_train_col_avg(res_X)

        return res_X, res_y

    def lasso_regression_model(self, train_X: pd.DataFrame,
                               train_y: pd.DataFrame,
                               validation_X: pd.DataFrame):
        _reg = Lasso(alpha=0.1)
        _reg.fit(train_X, train_y)
        return _reg.predict(validation_X)

    def predict_cancellation(self, X: pd.DataFrame):
        process_X = self.preprocess_test_data(X)
        # TODO: implement preprocess validate / test
        # TODO: predict by the chosen classifier (Tomer's work)
        # return prediction

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
                    cancellation_dict['no_show_charge'] = str(
                        ((int(days_or_fine) / int(stay_len)) * 100))
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
                    cancellation_dict['no_show_charge'] = str(
                        ((int(days_or_fine) / int(stay_len)) * 100))
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
        X['D1'] = X.apply(
            lambda row: self.get_D1(row['cancellation_policy_code'],
                                    row['stay_length_in_nights']),
            axis=1).astype(float)
        X['D2'] = X.apply(
            lambda row: self.get_D2(row['cancellation_policy_code'],
                                    row['stay_length_in_nights']),
            axis=1).astype(float)
        X['D3'] = X.apply(
            lambda row: self.get_D3(row['cancellation_policy_code'],
                                    row['stay_length_in_nights']),
            axis=1).astype(float)
        X['D4'] = X.apply(
            lambda row: self.get_D4(row['cancellation_policy_code'],
                                    row['stay_length_in_nights']),
            axis=1).astype(float)
        X['P1'] = X.apply(
            lambda row: self.get_P1(row['cancellation_policy_code'],
                                    row['stay_length_in_nights']),
            axis=1).astype(float)
        X['P2'] = X.apply(
            lambda row: self.get_P2(row['cancellation_policy_code'],
                                    row['stay_length_in_nights']),
            axis=1).astype(float)
        X['P3'] = X.apply(
            lambda row: self.get_P3(row['cancellation_policy_code'],
                                    row['stay_length_in_nights']),
            axis=1).astype(float)
        X['P4'] = X.apply(
            lambda row: self.get_P4(row['cancellation_policy_code'],
                                    row['stay_length_in_nights']),
            axis=1).astype(float)
        return X
