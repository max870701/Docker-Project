import pandas as pd
import numpy as np

def data_preprocessing(*args):
    tmp = []
    for df in args:
        data_cleaner(df)
        data_creator(df)
        data_convertor(df)
        tmp.append(df)
    return tmp

def data_cleaner(df):
    # 填充缺失值
    df['Embarked'].fillna('S', inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    # Cabin 缺失值過多，選擇drop；同時查看其他需要 drop 的 columns
    drop_cols = ['Cabin', 'PassengerId', 'Ticket']
    df.drop(columns=drop_cols, inplace=True)

def data_creator(df):
    # 構建家庭成員數量的feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    # 構建是否一個人上船的feature
    df['IsAlone'] = np.where(df['FamilySize'] == 1, 1, 0)
    # 構建 title 的 feature
    df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

def data_convertor(df):
    # 使用頻率編碼
    freq_cols = ['Sex', 'Embarked', 'Title', 'Pclass', 'FamilySize', 'IsAlone']
    for col in freq_cols:
        freq_encoder(df, col)

def freq_encoder(df, col):
    # 計算每個類別的頻率
    freq = df.groupby(col, observed=False).size() / len(df)
    # 將頻率映射到原數據集
    df[col+'_Freq'] = df[col].map(freq)