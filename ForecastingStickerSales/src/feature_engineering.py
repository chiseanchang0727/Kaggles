import pandas as pd
import re





def create_lag_features(data: pd.DataFrame, target_col='num_sold'):
    df = data.copy()
    df[target_col + '_lag_1'] =  df[target_col].shift(1)
    df[target_col + '_lag_2'] =  df[target_col].shift(2)
    df[target_col + '_lag_3'] =  df[target_col].shift(3)

    df[target_col + '_rolling_mean_3'] = df[target_col].rolling(window=3).mean()
    df[target_col + '_rolling_mean_5'] = df[target_col].rolling(window=5).mean()
    df[target_col + '_rolling_mean_7'] = df[target_col].rolling(window=7).mean()

    lag_cols = [col for col in df.columns if re.search('lag', col) or re.search('rolling', col)]
    for col in lag_cols:
        df[col] = df[col].fillna(df[target_col])

    return df

def generate_lag_features_by_group(df, group_cols, target_col='num_sold'):
    # Group by the specified columns and apply lag feature creation
    grouped = df.groupby(group_cols, group_keys=False)
    df_with_features = grouped.apply(create_lag_features, target_col=target_col).reset_index(drop=True)
    
    return df_with_features