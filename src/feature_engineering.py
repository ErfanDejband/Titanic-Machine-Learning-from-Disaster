import os
import pandas as pd
from data_preparation import load_data, save_data


# create feature engineering class for differnt feature engineering methods
class FeatureEngineering:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.deck_map = {1: 'B', 2: 'D', 3: 'F'}

    def add_FamilySize(self) -> pd.DataFrame:
        '''Add a new feature 'FamilySize' which is the sum of 'SibSp' and 'Parch'.'''
        if 'SibSp' in self.df.columns and 'Parch' in self.df.columns:
            self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
        return self.df
    
    def add_IsAlone(self) -> pd.DataFrame:
        '''Add a new feature 'IsAlone' which is 1 if 'FamilySize' is 1, else 0.'''
        if 'FamilySize' in self.df.columns:
            self.df['IsAlone'] = int(1)
            self.df.loc[self.df['FamilySize'] > 1, 'IsAlone'] = 0
        return self.df

    def add_title(self) -> pd.DataFrame:
        '''Extract title from the 'Name' feature and add it as a new feature 'Title'.'''
        if 'Name' in self.df.columns:
            self.df['Title'] = self.df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
            # Replace rare titles with 'Rare'
            rare_titles = self.df['Title'].value_counts()[self.df['Title'].value_counts() < 10].index
            self.df['Title'] = self.df['Title'].replace(rare_titles, 'Rare')
            self.df['Title'] = self.df['Title'].replace('Mlle', 'Miss')
            self.df['Title'] = self.df['Title'].replace('Ms', 'Miss')
            self.df['Title'] = self.df['Title'].replace('Mme', 'Mrs')
        return self.df
    
    def add_cabin_deck(self) -> pd.DataFrame:
        '''Extract deck information from the 'Cabin' feature and add it as a new feature 'Deck'.'''
        if 'Cabin' in self.df.columns and 'Pclass' in self.df.columns:
            self.df['Cabin_deck'] = self.df['Cabin'].astype(str).str[0]
            # Replace 'n' (from NaN converted to 'nan') with deck inferred from Pclass
            mask = self.df['Cabin_deck'] == 'n'
            if mask.any():
                self.df.loc[mask, 'Cabin_deck'] = self.df.loc[mask, 'Pclass'].map(self.deck_map)
            if self.df['Cabin_deck'].isnull().sum() > 0:
                raise ValueError("Null values found in 'Cabin_deck' after processing.")
            # drop original 'Cabin' feature
            self.df = self.df.drop(columns=['Cabin'])
        return self.df
    
    def drop_feature(self, feature_name: str) -> pd.DataFrame:
        '''Drop a feature from the DataFrame.'''
        if feature_name in self.df.columns:
            self.df = self.df.drop(columns=[feature_name])
        return self.df

# class for fill missing values
class FillMissingValues:
    def __init__(self, df: pd.DataFrame, Name=None):
        self.df = df
        self.Name = Name if Name else "DataFrame"

    def fill_age(self) -> pd.DataFrame:
        '''Fill missing values in the 'Age' feature with the median age based on their title if title available.'''
        if 'Age' in self.df.columns:
            if 'Title' in self.df.columns:
                # Use transform so the returned Series aligns with the original index
                self.df['Age'] = self.df['Age'].fillna(self.df.groupby('Title')['Age'].transform('median'))
            else:
                median_age = self.df['Age'].median()
                self.df['Age'] = self.df['Age'].fillna(median_age)
            if self.df['Age'].isnull().sum() > 0:
                raise ValueError("Null values found in 'Age' after filling.")
        return self.df

    def fill_embarked(self) -> pd.DataFrame:
        '''Fill missing values in the 'Embarked' feature with the mode.'''
        if 'Embarked' in self.df.columns:
            mode_embarked = self.df['Embarked'].mode()[0]
            self.df['Embarked'] = self.df['Embarked'].fillna(mode_embarked)
            if self.df['Embarked'].isnull().sum() > 0:
                raise ValueError("Null values found in 'Embarked' after filling.")
        return self.df

    def fill_fare(self) -> pd.DataFrame:
        '''Fill missing values in the 'Fare' feature with the median fare.'''
        if 'Fare' in self.df.columns:
            median_fare = self.df['Fare'].median()
            self.df['Fare'] = self.df['Fare'].fillna(median_fare)
            if self.df['Fare'].isnull().sum() > 0:
                raise ValueError("Null values found in 'Fare' after filling.")
        return self.df
    def warnings(self) -> None:
        '''Check for any remaining missing values in the DataFrame and raise warnings.'''
        missing_values = self.df.isnull().sum()
        if missing_values.any():
            print(f"**WARNING**: The following columns have missing values for {self.Name}")
            print(missing_values[missing_values > 0])
        else:
            print(f"No missing values found for {self.Name}.")


if __name__ == "__main__":
    # Define the path to the data file
    train_data_path = os.path.join('src','data', 'cleaned_train.csv')
    test_data_path = os.path.join('src','data', 'cleaned_test.csv')
    # Load the data
    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)
    # Feature Engineering
    fe_train = FeatureEngineering(train_data)
    train_data = fe_train.add_FamilySize()
    train_data = fe_train.add_IsAlone()
    train_data = fe_train.add_title()
    train_data = fe_train.add_cabin_deck()
    train_data = fe_train.drop_feature('Ticket')
    train_data = fe_train.drop_feature('Name')
    fe_test = FeatureEngineering(test_data)
    test_data = fe_test.add_FamilySize()
    test_data = fe_test.add_IsAlone()
    test_data = fe_test.add_title()
    test_data = fe_test.add_cabin_deck()
    test_data = fe_test.drop_feature('Ticket')
    test_data = fe_test.drop_feature('Name')
    # Fill Missing Values
    fmv_train = FillMissingValues(train_data, Name="Train Data")
    train_data = fmv_train.fill_age()
    train_data = fmv_train.fill_embarked()
    train_data = fmv_train.fill_fare()
    fmv_train.warnings()
    fmv_test = FillMissingValues(test_data, Name="Test Data")
    test_data = fmv_test.fill_age()
    test_data = fmv_test.fill_embarked()
    test_data = fmv_test.fill_fare()
    fmv_test.warnings()
    # Display the first few rows of the data
    if not train_data.empty and not test_data.empty:
        print(f'Train Data: shape = f{train_data.shape}\n{train_data.head()}')
        print(f'Test Data: shape = f{test_data.shape}\n{test_data.head()}')
        # Save the feature engineered data
        save_data(train_data, os.path.join('src', 'data', 'feature_engineered_train.csv'))
        save_data(test_data, os.path.join('src', 'data', 'feature_engineered_test.csv'))
    else:
        print("No data loaded.")
