import pandas as pd

def read_data():
    """ this funcitons reads the dataset file and returns a Pandas data frame the contains the data

    Args:
            NONE
    Returns:
        Pandas dataframe: dataframe the contains the data

    """

    df_features = pd.read_csv('../data/xtrain.txt',delimiter='\t', index_col=0,  header=None)
    df_features.replace('?', -99999, inplace=True)



    train_x = df_features.values

    df_labels = pd.read_csv('../data/ytrain.txt',  header=None)
    train_y = df_labels.values

    df_test = pd.read_csv('../data/xtest.txt',delimiter='\t', index_col=0,  header=None)
    df_test.replace('?', -99999, inplace=True)
    test_x = df_test.values




    return train_x, train_y,test_x


