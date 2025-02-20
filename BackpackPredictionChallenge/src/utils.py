from sklearn.model_selection import train_test_split


def split_data(df):

    target = 'Price'
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    return X_train, X_valid, y_train, y_valid
