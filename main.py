from src.dataloader import DataLoader
from src.preprocessor import Preprocessor


def main():

# --------- Load Data + Hellow
    print("Hello from stupid-project!")
    loader = DataLoader('data/salary_data.xlsx')
    df = loader.load_data()

# --------- Data Preprocessing 
    prep = Preprocessor()
    df_rn = prep.rename_cols(df)
    df_nn = prep.drop_nulls(df_rn)





if __name__ == "__main__":
    main()



