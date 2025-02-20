import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error



def get_models():
    """
    Returns a dictionary of base models for the ensemble.
    """
    models = {
        "xgb": XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
            gamma=0, reg_alpha=0, reg_lambda=1, objective='reg:squarederror',
            random_state=42, verbosity=1
        ),
        "lgbm": LGBMRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42
        ),
        "rf": RandomForestRegressor(
            n_estimators=500, max_depth=10, min_samples_split=5,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        )
    }
    return models



def ml_train(X_train, X_valid, y_train, y_valid, save_model, models_dir):

    # model = get_models()
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_valid)
    # rmse = root_mean_squared_error(y_valid, y_pred)


    models = get_models()
    train_meta_features = np.zeros((X_train.shape[0], len(models)))
    valid_meta_features = np.zeros((X_valid.shape[0], len(models)))
    model_paths = {}

    for i, (name, model) in enumerate(models.items()):
        print(f"Training {name} model...")
        model.fit(X_train, y_train)

        # Generate meta-features
        train_meta_features[:, i] = model.predict(X_train)
        valid_meta_features[:, i] = model.predict(X_valid)

        # Save model if required
        if save_model:
            model_path = models_dir / f"{name}_model.pkl"
            joblib.dump(model, model_path)
            model_paths[name] = model_path
            print(f"{name} model saved to {model_path}")


    meta_model = Ridge(alpha=1.0)
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
    
