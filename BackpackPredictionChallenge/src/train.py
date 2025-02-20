import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error

def get_parameter_grids():
    """
    Returns hyperparameter grids for base models.
    """
    param_grids = {
        "xgb": {
            "n_estimators": [500, 1000],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [4, 6, 8],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9]
        },
        "lgbm": {
            "n_estimators": [500, 1000],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [4, 6, 8],
            "num_leaves": [20, 31, 40],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9]
        },
        "rf": {
            "n_estimators": [200, 500, 1000],
            "max_depth": [6, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 3, 5]
        },
        "ridge": {
            "alpha": [0.1, 1.0, 10.0]
        }
    }
    return param_grids

def tune_hyperparameters(model, param_grid, X_train, y_train):
    """
    Runs GridSearchCV to find the best hyperparameters.
    """
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def train(X_train, X_valid, y_train, y_valid, save_model, models_dir):

    models = {
        "xgb": XGBRegressor(objective='reg:squarederror', random_state=42),
        "lgbm": LGBMRegressor(random_state=42),
        "rf": RandomForestRegressor(random_state=42)
    }

    param_grids = get_parameter_grids()

    train_meta_features = np.zeros((X_train.shape[0], len(models)))
    valid_meta_features = np.zeros((X_valid.shape[0], len(models)))
    model_paths = {}

    for i, (name, model) in enumerate(models.items()):
        print(f"Hyperparameter tuning for {name}...")
        best_model = tune_hyperparameters(model, param_grids[name], X_train, y_train)

        print(f"Training best {name} model...")
        best_model.fit(X_train, y_train)

        # Generate meta-features
        train_meta_features[:, i] = model.predict(X_train)
        valid_meta_features[:, i] = model.predict(X_valid)

        # Save model if required
        if save_model:
            model_path = models_dir / f"{name}_model.pkl"
            joblib.dump(model, model_path)
            model_paths[name] = model_path
            print(f"{name} model saved to {model_path}")


    # Tune and train meta-model (Ridge Regression)
    print("Tuning Ridge Regression meta-model...")
    meta_model = tune_hyperparameters(Ridge(), param_grids["ridge"], train_meta_features, y_train)

    meta_model.fit(train_meta_features, y_train)
    y_pred_meta = meta_model.predict(valid_meta_features)
    rmse = root_mean_squared_error(y_valid, y_pred_meta)
    print(f"RMSE: {rmse}")

    # Save meta-model
    if save_model:
        meta_model_path = models_dir / "meta_model.pkl"
        joblib.dump(meta_model, meta_model_path)
        # joblib.dump(scaler, models_dir / "scaler.pkl")
        model_paths["meta_model"] = meta_model_path
        print(f"Meta model saved to {meta_model_path}")
    
