
from pathlib import Path
import pandas as pd
import argparse
import torch
from src.utils import split_data
from src.preprocessing import preprocessing
from src.ml_train import ml_train
from src.dl_train import dl_train
from src.predict import predict

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ml_train", "dl_train", "predict"], required=True, help="Mode: 'train' or 'predict'")
    parser.add_argument('--save_model', action='store_true', required=False, default=False)

    return parser.parse_args()


def get_data(base_path):


    data_path = base_path / "data"
    df_train = pd.read_csv(data_path / 'train.csv')
    # df_train_extra = pd.read_csv('./data/training_extra.csv')
    df_test = pd.read_csv(data_path / 'test.csv')

    df_train = df_train.set_index('id')
    df_test = df_test.set_index('id')

    return df_train, df_test


def main():

    args = get_arguments()

    # Prevent --save_model from being used in predict mode
    if args.mode == "predict" and args.save_model:
        raise ValueError("Error: --save_model is only allowed in train mode. Remove --save_model when using --mode predict.")
    
    base_path = Path(__file__).resolve().parent  # Moves up to the project root dynamically
    models_dir = base_path / "models"
    models_dir.mkdir(exist_ok=True)  
    model_w_dir = base_path / "model_weights"

    df_train, df_test = get_data(base_path)


    if args.mode == 'ml_train':

        df_train_processed = preprocessing(df_train)
        X_train, X_valid, y_train, y_valid = split_data(df_train_processed)

        ml_train(X_train, X_valid, y_train, y_valid, save_model=args.save_model, models_dir=models_dir)

    elif args.mode == 'predict':
        df_test_process = preprocessing(df_test)

        output_dir = base_path / 'submissions'
        model_path = models_dir / "xgb_model.pkl"

        predict(df_test=df_test_process, model_path=model_path, output_dir=output_dir)
        
    elif args.mode == 'dl_train':
        df_train_processed = preprocessing(df_train)
        dl_train(df_train_processed, save_model=args.save_model, model_w_dir=model_w_dir)


 
if __name__ == '__main__':
    main()