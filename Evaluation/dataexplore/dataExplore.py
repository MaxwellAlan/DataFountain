import pandas as pd


def load_data(path_train,path_test):
    train_data = pd.read_csv(path_train)
    test_data = pd.read_csv(path_test)
    return train_data,test_data



