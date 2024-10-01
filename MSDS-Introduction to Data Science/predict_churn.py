import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(filepath):
    """
    Loads churn data into a DataFrame from a string filepath.
    """
    df = pd.read_excel('./prepared_churn_data.xlsx', index_col='customerID')
    return df


def make_predictions(df):
    """
    Uses the pycaret best model to make predictions on data in the df dataframe.
    """
    model = load_model('best_churn_model')
    predictions = predict_model(model, data=df)

    # Check the column names
    print(predictions.columns)
    
    # Rename 'prediction_label' to 'Churn_prediction' if it exists
    if 'prediction_label' in predictions.columns:
        predictions.rename(columns={'prediction_label': 'Churn_prediction'}, inplace=True)
        
        # Replace values in the new column
        predictions['Churn_prediction'
].replace({1: 'Churn', 0: 'No Churn'}, inplace=True)
        
        return predictions['Churn_prediction']
    else:
        raise KeyError("The 'prediction_label' column was not found in the predictions DataFrame")


if __name__ == "__main__":
    df = load_data('./new_Churn_data.csv')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)
