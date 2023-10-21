#Common Modules for Data Cleaning
import pandas as pd
import numpy as np

#Common Model Algorithms
from sklearn import ensemble

#Common Model Helpers
from sklearn import model_selection
from sklearn import metrics

# Save Model
from joblib import dump

def read_csv(file_name):
    return pd.read_csv(file_name)

def data_preprocessor(*args):
    for df in args:
        data_cleaner(df)
        data_creator(df)
        data_convertor(df)
    
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
    # 構建 FareBin 特徵
    df['FareBin'] = pd.qcut(df['Fare'], 4)
    # 構建 AgeBin 特徵
    df['AgeBin'] = pd.cut(df['Age'].astype(int), 5)

def data_convertor(df):
    # 使用頻率編碼
    freq_cols = ['Sex', 'Embarked', 'Title', 'Pclass', 'AgeBin', 'FareBin', 'FamilySize', 'IsAlone']
    for col in freq_cols:
        freq_encoder(df, col)

def freq_encoder(df, col):
    # 計算每個類別的頻率
    freq = df.groupby(col, observed=False).size() / len(df)
    # 將頻率映射到原數據集
    df[col+'_Freq'] = df[col].map(freq)

def in_sample_training(df):
    # 篩選所需特徵
    X, y = df[x_cols], df[target]
    # 分為訓練集和驗證集合
    X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.2, stratify=y, random_state=422)

    # 定義超參數空間
    # 定義RandomForest的超參數空間
    param_dist = {
        'n_estimators': range(10, 101, 10),
        'max_features': ['sqrt', 'log2'],
        'max_depth': range(2, 21),
        'min_samples_split': range(2, 11),
        'min_samples_leaf': [1, 2, 4, 8, 16],
        'bootstrap': [True, False]
    }

    # 初始化模型
    model = ensemble.RandomForestClassifier(random_state=10)

    # 使用RandomizedSearchCV進行超參數選擇
    random_search = model_selection.RandomizedSearchCV(
                                        estimator=model,
                                        param_distributions=param_dist,
                                        scoring='roc_auc',
                                        n_iter=200,
                                        cv=10,
                                        verbose=1,
                                        random_state=42,
                                        n_jobs=-1
                                        )
    random_search.fit(X_train, y_train)

    # 儲存roc_auc前十名的模型
    top_10_indices = np.argsort(random_search.cv_results_['mean_test_score'])[-10:]
    top_10_models = [random_search.cv_results_['params'][i] for i in top_10_indices]

    # 使用10-Fold交叉驗證並選擇roc_auc標準差最小的模型
    min_std = float("inf")
    best_model_params = None
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=42)

    for params in top_10_models:
        model.set_params(**params)
        scores = model_selection.cross_val_score(model, X, y, cv=kfold, scoring="roc_auc")
        if scores.std() < min_std:
            min_std = scores.std()
            best_model_params = params
            print('='*20)
            print(f"Cross-validation scores:\n{scores}")
            print(f"Average cross-validation score: {scores.mean():.4f}")
            print(f"Cross-validation standard deviation: {scores.std():.4f}")

    # 使用最佳參數擬合模型
    best_model = ensemble.RandomForestClassifier(random_state=10, **best_model_params)
    best_model.fit(X_train, y_train)
    # 輸出最佳模型的相關信息
    best_params = best_model.get_params()
    print(f'Best Params: {best_params}')

    # 使用最佳模型進行預測
    y_pred = best_model.predict(X_val)
    
    # 評估標準
    eval_score = metrics.roc_auc_score(y_val, y_pred)
    print(f'Validation Score: {eval_score:.4f}')

    return best_model  # 也可以返回其他相關信息，如scores等

def out_sample_score(df, df_target, model):
    # 篩選所需特徵
    X_test, y_test = df[x_cols], df_target[target]
    y_pred = model.predict(X_test)
    # 評估標準
    eval_score = metrics.roc_auc_score(y_test, y_pred)
    print(f"Testing Score: {eval_score:.4f}")
    return eval_score 

def train_model(train_data, test_data, test_data_target):
    best_model = in_sample_training(train_data)
    test_scores = out_sample_score(test_data, test_data_target, best_model)
    return best_model


if __name__ == "__main__":
    target = 'Survived'
    x_cols = ['Pclass_Freq', 'Sex_Freq', 'FareBin_Freq', 'Title_Freq', 'FamilySize_Freq', 'IsAlone_Freq', 'Embarked_Freq']

    print("Reading data...")
    train_df, test_df = read_csv('./datasets/train.csv'), read_csv('./datasets/test.csv')
    test_target_df = read_csv('./datasets/gender_submission.csv')

    print("Preprocessing data...")
    data_preprocessor(train_df, test_df)

    print("Training model...")
    best_model = train_model(train_df, test_df, test_target_df)

    print("Saving model...")
    dump(best_model, './models/titanic-model.joblib')

    print("Done")