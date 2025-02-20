import pandas as pd
import joblib
from datetime import datetime


def load_model(model_path):
    """
    Loads the trained model from a file.
    
    Args:
        model_path: Path to the trained model file.
    
    Returns:
        Loaded model.
    """
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def predict(df_test, model_path, output_dir):
    """
    Loads the trained model and makes predictions on test data.
    
    Args:
        model_path: Path to the trained model file.
        output_path: Path to save the predictions.
    
    Returns:
        Saves the predictions as a CSV file.
    """
    model = load_model(model_path)

    predictions = model.predict(df_test)
    
    submissions = pd.DataFrame({
        'id': df_test.index,
        'Price': predictions
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"submission_{timestamp}.csv"


    
    submissions.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")