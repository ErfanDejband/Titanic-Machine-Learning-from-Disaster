import os
import pandas as pd
from pathlib import Path

def load_data(file_path:str)->pd.DataFrame:
    '''Load data from a CSV file and return a pandas DataFrame.'''
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if "Survived" in df.columns:
            df = df.dropna(subset=["Survived"])
            df = df.reset_index(drop=True)
            return df
        else:
            return df
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return df
    
def save_data(df: pd.DataFrame, output_path: str) -> None:
    '''Save the DataFrame to a CSV file.'''
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

if __name__ == "__main__":
    # Define the path to the data file
    train_data_path = os.path.join('data', 'raw', 'train.csv')
    test_data_path = os.path.join('data', 'raw', 'test.csv')
    
    # Load the data
    train_data = load_data(train_data_path)
    test_data_path = load_data(test_data_path)
    
    # Clean the data
    train_data = clean_data(train_data)
    test_data = clean_data(test_data_path)

    # Display the first few rows of the data
    if not train_data.empty and not test_data_path.empty:
        print(f'Train Data: shape = f{train_data.shape}\n{train_data.head()}')
        print(f'Test Data: shape = f{test_data.shape}\n{test_data_path.head()}')
        # Save the cleaned data
        save_data(train_data, os.path.join('data', 'src_processed', 'cleaned_train.csv'))
        save_data(test_data_path, os.path.join('data', 'src_processed', 'cleaned_test.csv'))
    else:
        print("No data loaded.")