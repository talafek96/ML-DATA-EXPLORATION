import numpy as np
import pandas as pd
import random
import re


class DataImputer:
    def __init__(self, dataframe, features, strategy_dict, instructions_dict=dict(), missing_marker=np.NaN):
        self.strats = strategy_dict
        self.instructions_dict = instructions_dict
        self.df = dataframe.copy()
        self.missing_marker = missing_marker
        self.to_impute = features
        return

    def __median_impute(self, feature):
        median = self.df[feature].median()
        self.df[feature] = self.df[feature].fillna(median)
        return

    def __random(self, feature):
        items = list(self.df[feature].value_counts().index)
        for i in range(len(self.df[feature])):
            if self.df.loc[i, feature] is self.missing_marker:
                self.df.loc[i, feature] = random.choice(items)
        return

    def __arbitrary(self, feature, constant):
        self.df[feature] = self.df[feature].fillna(constant)
        return

    def __frequent(self, feature):
        frequent_category = self.df[feature].value_counts().index[0]
        self.df[feature] = self.df[feature].fillna(frequent_category)
        return

    def __bivariate_median(self, feature, secondary, init_bin_size=0):
        for i in range(self.df.shape[0]):
            if np.isnan(self.df.loc[i, feature]):
                tmp_secondary = self.df[secondary][i]
                tmp_bin_size = init_bin_size
                filtered_by_secondary = self.df[(self.df[secondary] >= tmp_secondary - tmp_bin_size) & (
                    self.df[secondary] <= tmp_secondary + tmp_bin_size)]
                while filtered_by_secondary[pd.notna(filtered_by_secondary[feature])].shape[0] == 0:
                    tmp_bin_size += 1
                    filtered_by_secondary = self.df[(self.df[secondary] >= tmp_secondary - tmp_bin_size) & (
                        self.df[secondary] <= tmp_secondary + tmp_bin_size)]
                bivariate_median = filtered_by_secondary[feature].median()
                self.df.loc[i, feature] = bivariate_median
        return

    def impute_data(self):
        for feature in self.to_impute:
            if self.strats[feature] == 'median':
                self.__median_impute(feature)
            elif self.strats[feature] == 'random':
                self.__random(feature)
            elif self.strats[feature] == 'arbitrary':
                self.__arbitrary(feature, self.instructions_dict[feature])
            elif self.strats[feature] == 'frequent':
                self.__frequent(feature)
            elif self.strats[feature] == 'bivariate_median' or self.strats[feature] == 'bivariate':
                self.__bivariate_median(
                    feature, self.instructions_dict[feature][0], self.instructions_dict[feature][1])
        return self.df

    @staticmethod
    def impute_blood_ohe(main_df, df_to_impute):
        frequent_blood = main_df.blood_type.value_counts().index[0]
        regex = r'([ABO]{1,2})([+-])'
        match = re.search(regex, frequent_blood)
        blood_label = 'blood_' + match.group(1)
        rh_label = 'blood_' + match.group(2)
        for i in range(df_to_impute.shape[0]):
            if df_to_impute.loc[i, 'blood_nan'] == 1:
                df_to_impute.loc[i, blood_label] = 1
                df_to_impute.loc[i, rh_label] = 1
        df_to_impute.drop(columns='blood_nan', inplace=True)
        return


def prepare_data(data, training_data):
    '''
    Returns the sum of two decimal numbers in binary digits.

            Parameters:
                    data (pandas.DataFrame): The dataframe to be cleaned.
                    training_data (pandas.DataFrame): The training set dataframe used to clean according to.

            Returns:
                    clean_data (pandas.DataFrame): A *copy* of data after it has been cleaned relatively to the provided training_data.
    '''
    pass
