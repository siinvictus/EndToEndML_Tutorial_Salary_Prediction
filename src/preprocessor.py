import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


class Preprocessor: 
    """
    The purpose of this class is to do the data prep:
    - rename cols
    - drop nulls
    - split X and y 
    - standartize if needed. 
    """

    def __init__(self):
        pass


    def rename_cols(self, data:pd.DataFrame):
        data_rn = data.rename(columns={'#': 'index', 'Exam Score (0–100)': 'exam_score', 
                                       'Years of Experience': 'years_exp', 'Salary (€)': 'salary'})
        print(f'The renamed columns are: ')
        for i,el in enumerate(data_rn):
            print(f'{i}.{el}')
        return data_rn
    
    def drop_nulls(self, data:pd.DataFrame):
        print(f'Nulls in the dataset: {data.isna().sum()}')
        data_no_null = data.dropna()
        print(f'Nulls droped and the dataset now has {data_no_null.shape[0]} rows.')
        return data_no_null

    def split_X_y(self, data:pd.DataFrame):
        X = data.drop(['salary'], axis = 1)
        y = data['salary'].copy()
        print(f'Your X, indepentent variables are {X.columns} and your dependent variable is {y.name}')
        return X,y
    
    def standartize_X(self, X: pd.DataFrame):
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    
    def min_max_X (self, X:pd.DataFrame):
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)
    


