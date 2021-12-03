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


class OutlierCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        return

    def __z_score_clean(self, column, lowest, highest):
        if lowest is None:
            lowest = self.df[column].mean() - 3*self.df[column].std()
        if highest is None:
            highest = self.df[column].mean() + 3*self.df[column].std()
        self.df[column] = np.where(
            self.df[column] > highest,
            highest,
            np.where(
                self.df[column] < lowest,
                lowest,
                self.df[column]
            )
        )

    def __iqr_clean(self, column, lowest, highest):
        percentile25 = self.df[column].quantile(0.25)
        percentile75 = self.df[column].quantile(0.75)
        iqr = percentile75 - percentile25
        if lowest is None:
            lowest = percentile25 - 1.5*iqr
        if highest is None:
            highest = percentile75 + 1.5*iqr
        self.df[column] = np.where(
            self.df[column] > highest,
            highest,
            np.where(
                self.df[column] < lowest,
                lowest,
                self.df[column]
            )
        )

    def z_score_filter(self, column):
        highest = self.df[column].mean() + 3*self.df[column].std()
        lowest = self.df[column].mean() - 3*self.df[column].std()
        return self.df[(self.df[column] > highest) | (self.df[column] < lowest)]

    def iqr_filter(self, column):
        percentile25 = self.df[column].quantile(0.25)
        percentile75 = self.df[column].quantile(0.75)
        iqr = percentile75 - percentile25
        highest = percentile75 + 1.5*iqr
        lowest = percentile25 - 1.5*iqr
        return self.df[(self.df[column] > highest) | (self.df[column] < lowest)]

    def clean_outliers(self, column, filter, lowest=None, highest=None):
        '''
        Cleans the outliers of the column of the dataframe managed in the class *in-place*.

                Parameters:
                        column (str): The feature's column to be cleaned from outliers.
                        filter (str): The cleaning method. Either 'z-score' or 'iqr'.
                        lowest (Optional[int]): If the filter 'iqr' was chosen and lowest was stated, the bottom bound will be set to this value.
                        highest (Optional[int]): If the filter 'iqr' was chosen and highest was stated, the upper bound will be set to this value.

                Returns:
                        The dataframe managed in the class (the original, to allow pipelining).
        '''
        if filter.lower() == 'z_score':
            self.__z_score_clean(self.df, column, lowest, highest)
        elif filter.lower() == 'iqr':
            self.__iqr_clean(self.df, column, lowest, highest)
        else:
            raise Exception("Method not supported.")
        return self.df

# def getSymptomList(train, index):
#   global symptom_set
#   global symptom_dict
#   symptoms = train_ohe.loc[index, 'symptoms']
#   if symptoms is np.NaN:
#     return [np.NaN]*len(symptom_set)
#   res = [0]*len(symptom_set)
#   symptoms = symptoms.split(';')
#   for symptom in symptoms:
#     res[symptom_dict[symptom]] = 1
#   return res


def date_to_num(date: str) -> int:
    regex = r'(\d{4})-(\d{2})-(\d{2})'
    match = re.search(regex, date)
    return int(match.group(1) + match.group(2) + match.group(3))


def prepare_data(data, training_data):
    '''
    Returns a cleaned copy of data ready to be used for prediction.

            Parameters:
                    data (pandas.DataFrame): The dataframe to be cleaned.
                    training_data (pandas.DataFrame): The training set dataframe used to clean according to.

            Returns:
                    clean_data (pandas.DataFrame): A *copy* of data after it has been cleaned relatively to the provided training_data.
    '''
    pass
