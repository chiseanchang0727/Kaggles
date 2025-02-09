import pandas as pd
from xgboost import XGBRFRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_percentage_error


class Predictor:
    def __init__(self, X_train, X_valid, y_train, y_valid):
        self.model = self.get_model()
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid

    def xgb_model(self):

        return XGBRFRegressor(
            n_estimators=120,        # Number of trees
            max_depth=5,             # Maximum depth of each tree
            learning_rate=0.01,       # Learning rate (shrinkage factor)
            subsample=0.8,           # Subsample ratio for training instances
            colsample_bynode=0.8,    # Subsample ratio for columns at each tree node
            random_state=42          # For reproducibility
        )
        
    def lgbm_model(self):
        return lgb.LGBMRegressor(
            n_estimators=3770,
            learning_rate=0.05038034487788465,
            max_depth=14,
            reg_alpha=0.20732364284443197,
            reg_lambda=0.004223724135505332,
            min_child_samples=29,
            colsample_bytree=0.6601202363535343,
            subsample=0.5597689123597346,
            objective='regression',  
            metric='mape',  
            n_jobs=-1,
            device='gpu'  # Enable GPU for LightGBM
        )

    def catboost_model(self):
        return CatBoostRegressor(
            n_estimators=1891,
            learning_rate=0.06761514972690001,
            depth=8,
            min_data_in_leaf=54,
            l2_leaf_reg=5.567375613813537,
            bagging_temperature=0.15478395184586632,
            random_strength=0.9462614107298501,
            loss_function='MAPE',
            eval_metric='MAPE',
            random_state=42,
            early_stopping_rounds=50  # Specify the GPU device (use '0' for the first GPU, or '1' for the second GPU, etc.)
        )


    def get_model(self):

        xgb = self.xgb_model()
        lgbm = self.lgbm_model()
        cb = self.catboost_model()
        
        meta_model = LinearRegression()
        
        # stacking
        model = StackingRegressor(
            estimators=[('xgb', xgb), ('cb', cb)],
            final_estimator=meta_model,
            n_jobs=-1
        )

        return model
    

    def train_and_eval(self):
        
        self.model.fit(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_valid)

        mape = mean_absolute_percentage_error(self.y_valid, y_pred)

        return mape

    def predict_on_test(self, df_test: pd.DataFrame):

        y_pred = self.model.predict(df_test)

        return y_pred

