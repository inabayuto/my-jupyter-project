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
import lightgbm as lgb
import json
import datetime as dt
from sklearn.metrics import mean_absolute_error

# %% データディレクトリの設定
def get_data_dir():
    """データディレクトリのパスを取得する"""
    # スクリプトのディレクトリを取得
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # データディレクトリのパスを構築
    data_dir = os.path.join(script_dir, 'data/mlb-player-digital-engagement-forecasting/')
    return data_dir

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


# %% JSON文字列を展開する関数
def unpack_json(json_str):
    """JSON文字列を展開する関数"""
    if pd.isna(json_str):
        return None
    try:
        obj = json.loads(json_str)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        elif isinstance(obj, dict):
            return pd.DataFrame([obj])
        else:
            return None
    except Exception as e:
        print(f"unpack_json error: {e}")
        return None


def extract_data(df, col='events', show=False):
    output_list = []
    for i in range(len(df)):
        if show:
            print(f"\r{i+1}/{len(df)}", end="")
        unpacked = unpack_json(df[col].iloc[i])
        if isinstance(unpacked, pd.DataFrame) and not unpacked.empty:
            output_list.append(unpacked)
    if output_list:
        output_df = pd.concat(output_list, axis=0, ignore_index=True)
    else:
        output_df = pd.DataFrame()
    if show: print(f"\noutput_df.shape: {output_df.shape}")
    return output_df


# %% データの前処理関数
def preprocess_data(train_df, player_df):
    """データの前処理を行う関数"""
    print('Importing data...')
    print('start size: {:5.2f} Mb'.format(train_df.memory_usage().sum() / 1024**2))
    print('start size: {:5.2f} Mb'.format(player_df.memory_usage().sum() / 1024**2))
    
    # メモリ使用量の削減
    train_df = reduce_mem_usage(train_df)
    player_df = reduce_mem_usage(player_df)
    train_df = train_df.loc[train_df['date'] >= 20200401, :].reset_index(drop=True)
    print('train_df.shape: ', train_df.shape)
    print('player_df.shape: ', player_df.shape)
    print('player_df["playerId"].nunique(): ', player_df['playerId'].nunique())

    engagement_df = extract_data(train_df, col='nextDayPlayerEngagement', show=True)
    # 結合キーの作成
    engagement_df['date_player_id'] = engagement_df['engagementMetricsDate'].str.replace('-', '') + '_' + engagement_df['playerId'].astype(str)
    # 推論実施日のカラムを作成
    engagement_df['data'] = pd.to_datetime(engagement_df['engagementMetricsDate'], format='%Y-%m-%d') + dt.timedelta(days=1)

    # 推論実施日から「曜日」と「年月」の特徴量を作成
    engagement_df['day_of_week'] = engagement_df['data'].dt.dayofweek
    engagement_df['year_month'] = engagement_df['data'].dt.strftime('%Y-%m')
    engagement_df.head()

    # 評価対象人数の確認
    player_df['playerForTestSetAndFuturePreds'] = np.where(player_df['playerForTestSetAndFuturePreds'] == True, 1, 0)

    # データの結合
    train_df = pd.merge(engagement_df, player_df, on='playerId', how='left')
    # 学習用データセットの作成
    x_train = train_df[['playerId',
                        'day_of_week',
                        'birthCity',
                        'birthStateProvince',
                        'birthCountry',
                        'heightInches',
                        'weight',
                        'primaryPositionCode',
                        'primaryPositionName',
                        'playerForTestSetAndFuturePreds']]

    y_train = train_df[['target1',
                        'target2',
                        'target3',
                        'target4']]
    id_train = train_df[['engagementMetricsDate', 
                        'playerId', 
                        'date_player_id', 
                        'data',
                        'year_month', 
                        'playerForTestSetAndFuturePreds']]

    # ラベルエンコーダーでobject型を数値に変換
    for col in ['birthCity', 'birthStateProvince', 'birthCountry', 'primaryPositionCode', 'primaryPositionName']:
        le = LabelEncoder()
        x_train[col] = le.fit_transform(x_train[col].astype(str))

    print('x_train.shape: ', x_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('id_train.shape: ', id_train.shape)

    return x_train, y_train, id_train

# %% モデルのパラメータ設定
def get_model_params():
    """モデルのパラメータを取得する関数"""
    params = {
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': 'mean_absolute_error',
    'num_leaves': 32,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'min_data_in_leaf': 50,
    'bagging_freq': 5,
    'bagging_fraction': 0.8,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'min_gain_to_split': 0.01,
    'max_depth': 8,
    'num_iterations': 1000,
    'early_stopping_rounds': 100,
    'verbose': -1,
    }
    return params

# %% モデルの学習
def train_and_predict(x_train_fold, y_train_fold, x_val_fold, y_val_fold, params):
    """モデルの学習と予測を行う関数"""
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X=x_train_fold,
        y=y_train_fold,
        eval_set=[(x_train_fold, y_train_fold), (x_val_fold, y_val_fold)],
        eval_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
        ]
    )
    # モデル評価
    # 学習データとの予測
    y_pred_train = model.predict(x_train_fold)

    # 検証データとの予測
    y_pred_val = model.predict(x_val_fold)

    return model, y_pred_train, y_pred_val

# %% 学習データと検証データの期間の設定
list_cv_month = [
[['2020-05', '2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03', '2021-04'], ['2021-05']],
[['2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03', '2021-04', '2021-05'], ['2021-06']],
[['2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06'], ['2021-07']],
]

# %% メイン処理
def main():
    # データディレクトリの設定
    DATA_DIR = get_data_dir()
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"データディレクトリが見つかりません: {DATA_DIR}\n現在の作業ディレクトリ: {os.getcwd()}")
    
    # データの読み込み
    train_path = os.path.join(DATA_DIR, "train_updated.csv")
    player_path = os.path.join(DATA_DIR, "players.csv")
    train_df = pd.read_csv(train_path)
    player_df = pd.read_csv(player_path)
    
    # データの前処理
    x_train, y_train, id_train = preprocess_data(train_df=train_df, player_df=player_df)

    # 学習データと検証データの期間の設定
    cv = []
    for month_train, month_val in list_cv_month:
        cv.append([
            id_train.index[id_train['year_month'].isin(month_train)],
            id_train.index[id_train['year_month'].isin(month_val) & id_train['playerForTestSetAndFuturePreds'] == 1]
        ])    
    # モデルのパラメータ取得
    params = get_model_params()
    
    # 結果格納用の変数
    metrics = []
    models = [] 

    # 交差検証の設定
    list_nfold=[0,1,2]

    for nfold in list_nfold:
        for i, target in enumerate(['target1', 'target2', 'target3', 'target4']):
            train_index = cv[nfold][0]
            val_index = cv[nfold][1]

            # 学習データと検証データに分割
            x_train_fold = x_train.iloc[train_index]
            y_train_fold = y_train.loc[train_index, target]
            id_train_fold = id_train.iloc[train_index]

            x_val_fold = x_train.iloc[val_index]
            y_val_fold = y_train.loc[val_index, target]
            id_val_fold = id_train.iloc[val_index]

             # モデルの学習と予測
            model, y_pred_train, y_pred_val = train_and_predict(
                x_train_fold, y_train_fold, x_val_fold, y_val_fold, params
            )

            # 結果の保存
            models.append(model)
            metrics_train = mean_absolute_error(y_train_fold, y_pred_train)
            metrics_val = mean_absolute_error(y_val_fold, y_pred_val)

            # 推論値を格納
            metrics.append([
                target,
                nfold+1,
                metrics_train,
                metrics_val
            ])

            # 重要度の取得と表示
            imp = pd.DataFrame({
                'feature': x_train_fold.columns,
                'importance': model.feature_importances_,
                'fold': nfold+1
            })
            imp.head()

    # メトリクスデータフレームの作成（ループの外で）
    metrics_df = pd.DataFrame(metrics, columns=['target', 'fold', 'mae_train', 'mae_val'])
    print(f'mcmae: {metrics_df["mae_val"].mean():.4f}')
    return metrics_df  # 明示的にmetrics_dfを返す

# %% スクリプトとして実行された場合の処理
if __name__ == "__main__":
    metrics_df = main()     
     

# %% 結果の表示
display(pd.pivot_table(metrics_df, index='fold', columns='target', values='mae_val',
aggfunc='mean', margins=True))

# %%
print(metrics_df)

# %%
