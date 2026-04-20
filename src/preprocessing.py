import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    return df

def split_data(df):
    train_list = []
    test_list = []

    for user in df['user_id'].unique():
        user_data = df[df['user_id'] == user]

        if len(user_data) < 2:
            continue

        train, test = train_test_split(user_data, test_size=0.2, random_state=42)

        train_list.append(train)
        test_list.append(test)

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    return train_df, test_df