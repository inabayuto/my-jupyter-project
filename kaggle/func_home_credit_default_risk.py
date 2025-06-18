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
    """メモリ使用量を削減する関数"""
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

# %% データの前処理関数
def preprocess_data(train_df, test_df):
    """データの前処理を行う関数"""
    print('Importing data...')
    print('start size: {:5.2f} Mb'.format(train_df.memory_usage().sum() / 1024**2))
    print('start size: {:5.2f} Mb'.format(test_df.memory_usage().sum() / 1024**2))
    
    # メモリ使用量の削減
    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)
    print(train_df.shape)
    print(test_df.shape)
    
    # データセットの作成
    x_train = train_df.drop(columns=['TARGET', 'SK_ID_CURR'])
    y_train = train_df['TARGET']
    id_train = train_df['SK_ID_CURR']
    
    # カテゴリ変数をcategoricalに変換
    categorical_cols = x_train.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        x_train[col] = x_train[col].astype('category')
    
    return x_train, y_train, id_train, test_df

# %% モデルのパラメータ設定
def get_model_params():
    """モデルのパラメータを取得する関数"""
    return {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'random_state': 123,
        'importance_type': 'gain',
    }

# %% モデルの学習と予測を行う関数
def train_and_predict(x_train_fold, y_train_fold, x_val_fold, y_val_fold, params):
    """モデルの学習と予測を行う関数"""
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
    
    # 予測
    y_pred_train = model.predict_proba(x_train_fold)[:, 1]
    y_pred_val = model.predict_proba(x_val_fold)[:, 1]
    
    return model, y_pred_train, y_pred_val

# %% メイン処理
def main():
    # データディレクトリの設定
    DATA_DIR = get_data_dir()
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"データディレクトリが見つかりません: {DATA_DIR}\n現在の作業ディレクトリ: {os.getcwd()}")
    
    # データの読み込み
    train_path = os.path.join(DATA_DIR, "application_train.csv")
    test_path = os.path.join(DATA_DIR, "application_test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # データの前処理
    x_train, y_train, id_train, test_df = preprocess_data(train_df, test_df)
    
    # 交差検証の設定
    n_splits = 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    
    # モデルのパラメータ取得
    params = get_model_params()

    # 重要度の格納用の変数
    imp_list = []
    
    # 結果格納用の変数
    metrics = []
    train_oof = np.zeros(len(train_df))
    models = []
    
    # 交差検証
    print(f"\nバリデーション設計 mean: {y_train.mean():.4f}")
    print(y_train.value_counts())
    
    for n_fold, (train_index, val_index) in enumerate(cv.split(x_train, y_train)):
        print(f"\nFold {n_fold+1}/{n_splits}")
        
        # データの分割
        x_train_fold = x_train.iloc[train_index]
        y_train_fold = y_train.iloc[train_index]
        id_train_fold = id_train.iloc[train_index]
        
        x_val_fold = x_train.iloc[val_index]
        y_val_fold = y_train.iloc[val_index]
        id_val_fold = id_train.iloc[val_index]
        
        print(f"Training data: {len(x_train_fold)}, Validation data: {len(x_val_fold)}")
        
        # モデルの学習と予測
        model, y_pred_train, y_pred_val = train_and_predict(
            x_train_fold, y_train_fold, x_val_fold, y_val_fold, params
        )
        
        # 結果の保存
        models.append(model)
        metrics.append([n_fold, y_pred_train, y_pred_val])
        train_oof[val_index] = y_pred_val
        
        # スコアの表示
        train_score = roc_auc_score(y_train_fold, y_pred_train)
        val_score = roc_auc_score(y_val_fold, y_pred_val)
        print(f"Fold {n_fold+1} - Train AUC: {train_score:.4f}, Valid AUC: {val_score:.4f}")

        # 各foldの重要度を格納
        imp_fold = pd.DataFrame({
            'feature': x_train_fold.columns,
            'importance': model.feature_importances_,
            'fold': n_fold+1
        })
        imp_list.append(imp_fold)

        # 重要度の表示
        print(imp_fold)
    
    # 全fold分の重要度をまとめる
    imp = pd.concat(imp_list, axis=0)
    
    # 全体のスコアを計算
    overall_score = roc_auc_score(y_train, train_oof)
    print(f"\nOverall OOF AUC: {overall_score:.4f}")

    # 重要度の集計と表示
    imp_mean = imp.groupby('feature')['importance'].mean().reset_index()
    imp_mean = imp_mean.sort_values(by='importance', ascending=False)
    print(imp_mean.head(10))

    return models, metrics, train_oof, imp_mean

# %% スクリプトとして実行された場合の処理
if __name__ == "__main__":
    models, metrics, train_oof, imp_mean = main()

# %% 説明変数の重要度の確認
imp_mean.head(10)
# %%
