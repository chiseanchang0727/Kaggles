import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import holidays

def create_sinusoidal_transformation_year_month_day(df):

    # df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[year] * df[month] * df[day] / period)
    # df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[year] * df[month] * df[day] / period)

    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['year_sin'] = np.sin(2 * np.pi * df['year'] / 7)
    df['year_cos'] = np.cos(2 * np.pi * df['year'] / 7)
    
    return df

def create_time_features(df: pd.DataFrame, date_col='date'):
    df[date_col] = pd.to_datetime(df[date_col])

    # Time-based features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofWeek'] = df[date_col].dt.dayofweek
    df['weekend'] = np.where(df['dayofWeek']>5, 1, 0)
    df['week_of_year'] = df[date_col].dt.isocalendar().week.astype('int')

    # for country in df.country.unique():
    #     holiday_cal = holidays.CountryHoliday(country=country)
    #     df[f'{country}_holiday'] = df[date_col].apply(lambda x: x in holiday_cal).astype(int)

    df = create_sinusoidal_transformation_year_month_day(df)
    
    df['group'] = (df['year'] - 2010) * 48 + df['month'] * 4 + df['day'] // 7

    return df


def imputation(df: pd.DataFrame, group_by: list):


    # df.loc[df.index.isin(train_idx), 'num_sold'] = df.loc[df.index.isin(train_idx), 'num_sold'].fillna(0)

    # df['num_sold'] = df['num_sold'].fillna(df['num_sold'].mean())
    
    df = df.dropna()

    # df_temp = df.groupby(group_by)['num_sold'].mean().reset_index(name='avg_sold').round(0)
    # df_merge = pd.merge(df, df_temp, how='left', on=group_by)
    # df_merge['num_sold'] = np.where(df_merge['num_sold'].isna(), df_merge['avg_sold'], df_merge['num_sold'])

    return df

def target_transformation(df, target='num_sold'):
    
    df[target] = np.log1p(df[target])

    return df

def reverse_target_transformation(input):
    reversed_input = np.expm1(input) 
    return reversed_input

def encoding(df: pd.DataFrame):

    categorical_col = df.select_dtypes('object').columns.to_list()
    # df_encoded = pd.get_dummies(df, columns=categorical_col, drop_first=False, dtype=int)

    for col in categorical_col:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df


def data_splitting(df, target_col, test_size=0.2):

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    return train_test_split(X, y, test_size=test_size, shuffle=False, random_state=42)



def split_and_standardization(df, target_col, test_size):

    X_train, X_valid, y_train, y_valid = data_splitting(df, target_col, test_size)

    scaler_train = StandardScaler()
    X_train_scaled = scaler_train.fit_transform(X_train)
    X_valid_scaled = scaler_train.transform(X_valid)


    return scaler_train, X_train, X_valid, y_train, y_valid

