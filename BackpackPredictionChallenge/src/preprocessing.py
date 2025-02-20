import pandas as pd
from sklearn.preprocessing import LabelEncoder



def encode_categorical_features(input: pd.DataFrame, cate_cols: list):
    
    df = input.copy()
    
    encoder = LabelEncoder()
    
    for col in cate_cols:
        df[col] = encoder.fit_transform(df[col])
    
    return df







def preprocessing(df):

    cate_cols = df.select_dtypes(include=['object', 'category']).columns
    df_process = encode_categorical_features(df, cate_cols)

    return df_process