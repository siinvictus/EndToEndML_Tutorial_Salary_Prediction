import pandas as pd

class DataLoader: 
    """ 
    The purpose of this class is to load the data in a pandas dataframe 
    from wherever they are
    """

    def __init__(self, data_path:str):
        self.data_path = data_path

    
    def load_data(self):
        data = pd.read_excel(self.data_path)
        print(f'The dataset is loaded and it has {data.shape[0]} rows and {data.shape[1]} columns.')
        print('The dataset has the following columns: ')
        for i,el in enumerate(data.columns):
            print(f'{i}.{el}')
        return data



