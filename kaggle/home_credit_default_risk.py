# %% ライブラリのインポート
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import pickle
import os
import gc
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import lightgbm as lgb

warnings.filterwarnings('ignore')

# %% データディレクトリの設定
def get_data_dir():
    """データディレクトリのパスを取得する"""
    # スクリプトのディレクトリを取得
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # データディレクトリのパスを構築
    data_dir = os.path.join(script_dir, 'data/home-credit-default-risk/')
    return data_dir

# データディレクトリのパスを取得
DATA_DIR = get_data_dir()
print(DATA_DIR)

# データディレクトリの存在確認
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"データディレクトリが見つかりません: {DATA_DIR}\n現在の作業ディレクトリ: {os.getcwd()}")

# %% データの読み込み
train_path = os.path.join(DATA_DIR, "application_train.csv")
test_path = os.path.join(DATA_DIR, "application_test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# %% メモリ削減の関数
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# %% データのメモリ削減
# データの読み込み
print('Importing data...')
print('start size: {:5.2f} Mb'.format(train_df.memory_usage().sum() / 1024**2))
print('start size: {:5.2f} Mb'.format(test_df.memory_usage().sum() / 1024**2))
train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)
print(train_df.shape)
print(test_df.shape)

# %% データセットの作成
x_train = train_df.drop(columns=['TARGET', 'SK_ID_CURR'])
y_train = train_df['TARGET']
id_train = train_df['SK_ID_CURR']

# カテゴリ変数をcategoricalに変換
categorical_cols = x_train.select_dtypes(include=['object']).columns
for col in categorical_cols:
    x_train[col] = x_train[col].astype('category')

# %% バリデータション設計
print(f"バリデータション設計 mean: {y_train.mean():.4f}")
print(y_train.value_counts())

# 階層化分割したバリデーションのindexリストの作成
cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(x_train, y_train))

print(f"len(train): {len(cv[0][0])}")
print(f"len(val): {len(cv[0][1])}")

print(f"index(train): {cv[0][0]}")
print(f"index(val): {cv[0][1]}")

# 0foldのindexを取得
n_fold = 0
train_index = cv[n_fold][0]
val_index = cv[n_fold][1]

# 学習データと検証データに分割
x_train_fold = x_train.iloc[train_index]
y_train_fold = y_train.iloc[train_index]
id_train_fold = id_train.iloc[train_index]

x_val_fold = x_train.iloc[val_index]
y_val_fold = y_train.iloc[val_index]
id_val_fold = id_train.iloc[val_index]

print(f"len(train): {len(x_train_fold)}")
print(f"len(val): {len(x_val_fold)}")
print(f"len(id_train): {len(id_train_fold)}")

# %% パラーメータの指定
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'random_state': 123,
    'importance_type': 'gain',
}

# %% モデルの学習
model = lgb.LGBMClassifier(**params)
model.fit(
    X=x_train_fold,
    y=y_train_fold,
    eval_set=[(x_train_fold, y_train_fold), (x_val_fold, y_val_fold)],
    eval_names=['train', 'valid'],
    eval_metric='auc',
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100)
    ]
)

# %% モデルの予測
# 学習データとの予測
y_pred_train = model.predict_proba(x_train_fold)[:, 1]

# 検証データとの予測
y_pred_val = model.predict_proba(x_val_fold)[:, 1]

# 評価値を入れる変数の作成
metrics = []

# 評価値を格納
metrics.append([n_fold, y_pred_train, y_pred_val])

# 結果の表示
print(f"fold: {n_fold}")
print(f"auc: {roc_auc_score(y_train_fold, y_pred_train)}")
print(f"auc: {roc_auc_score(y_val_fold, y_pred_val)}")

# %% モデルの評価
print(f"auc: {roc_auc_score(y_train_fold, y_pred_train)}")
print(f"auc: {roc_auc_score(y_val_fold, y_pred_val)}")

# %% OOFの予測値を格納
train_oof = np.zeros(len(train_df))

# 検証データの推論値に格納
train_oof[val_index] = y_pred_val


# %%

