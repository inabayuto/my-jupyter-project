# MNISTでMLP
# %%
import torch
from sklearn import datasets
import matplotlib.pyplot as plt
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

# %% データロード
dataset = datasets.load_digits()
images = dataset['images']
target = dataset['target']

# %% 学習データと検証データ分割
X_train, X_val, y_train, y_val = train_test_split(images, target, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

# %% 前処理
# ラベルのone-hot encoing
y_train = F.one_hot(torch.tensor(y_train), num_classes=10)
X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 64)

y_val = F.one_hot(torch.tensor(y_val), num_classes=10)
X_val = torch.tensor(X_val, dtype=torch.float32).reshape(-1, 64)

# 画像の標準化
X_train_mean = X_train.mean()
X_train_std = X_train.std()
X_train = (X_train - X_train_mean) / X_train_std
X_val = (X_val - X_train_mean) / X_train_std

# %% スクラッチの実装
# パラメータの初期化
m, n = X_train.shape
nh = 30
class_num = 10
# パラメータの初期化
W1 = torch.randn((nh, n), requires_grad=True) # 出力 x 入力
b1 = torch.zeros((1, nh), requires_grad=True) # 1 x nh

W2 = torch.randn((class_num, nh), requires_grad=True) # 出力 x 入力
b2 = torch.zeros((1, class_num), requires_grad=True) # 1 x nha

print(X_train.shape)
print(W1.T.shape, b1.shape)
print(W2.T.shape, b2.shape)

# %% 線形層
def linear(X, W, b):
    return X@W.T + b

# %% ReLU
def relu(Z):
    return Z.clamp_min(0.)

# %% SoftmaxとCross Entropy関数の定義
def softmax(x):
    e_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
    return e_x / (torch.sum(e_x, dim=-1, keepdim=True) + 1e-10)

def cross_entropy(y_true, y_pred):
    return -torch.sum(y_true * torch.log(y_pred + 1e-10)) / y_true.shape[0]

# %% モデル
def model(X):
    Z1 = linear(X, W1, b1)
    A1 = relu(Z1)
    Z2 = linear(A1, W2, b2)
    A2 = softmax(Z2)
    return A2

y_train_pred = model(X_train)
print(y_train_pred.shape)
print(y_train_pred.sum(dim=1))