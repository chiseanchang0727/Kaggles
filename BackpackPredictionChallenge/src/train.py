import joblib
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor


def get_xgb():
        return XGBRegressor(
        n_estimators=1000,      # Number of trees (increase for better performance)
        learning_rate=0.05,     # Step size shrinkage (lower values improve generalization)
        max_depth=6,            # Maximum depth of trees (higher values may lead to overfitting)
        min_child_weight=1,     # Minimum sum of instance weight in child nodes
        subsample=0.8,          # Subsample ratio of training instances
        colsample_bytree=0.8,   # Subsample ratio of columns per tree
        gamma=0,                # Minimum loss reduction required to make a split
        reg_alpha=0,            # L1 regularization term on weights
        reg_lambda=1,           # L2 regularization term on weights
        objective='reg:squarederror',  # Loss function for regression
        random_state=42,
        verbosity=1
    )



def get_model():

    model = get_xgb()


    return model



def train(X_train, X_valid, y_train, y_valid, save_model, model_path):

    model = get_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    rmse = root_mean_squared_error(y_valid, y_pred)
    
    print(f"RMSE: {rmse}")

    if save_model:
        # Save the trained model
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
