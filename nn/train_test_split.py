# %%
import numpy as np
import torch
from torch.nn import functional as F
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# %% ハイパーパラメータ設定と損失ログ用リスト
learning_rate = 0.03
loss_log = []

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

print(X_train.mean(), X_val.mean())
print(X_train.std(), X_val.std())


# %% SoftmaxとCross Entropy関数の定義
def softmax(x):
    e_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
    return e_x / (torch.sum(e_x, dim=-1, keepdim=True) + 1e-10)

def cross_entropy(y_true, y_pred):
    return -torch.sum(y_true * torch.log(y_pred + 1e-10)) / y_true.shape[0]

# %% モデル構築
batch_size = 30
num_batches = np.ceil(len(y_train) / batch_size).astype(int)
loss_log = []
# 3. パラメータの初期化
W = torch.rand((10, 64), requires_grad=True) # 出力 x 入力
b = torch.rand((1, 10), requires_grad=True) # 1 x 出力
# ログ
train_losses = []
val_losses = []
val_accuracies = []
# 5. for文で学習ループ作成
epochs = 30
for epoch in range(epochs):
    shuffled_indices = np.random.permutation(len(y_train))
    running_loss = 0
    for i in range(num_batches):

        # mini batch作成
        start = i * batch_size
        end = start + batch_size
        batch_indices = shuffled_indices[start:end]
        # 入力データXおよび教師ラベルのYを作成
        y_true_ = y_train[batch_indices, :] # データ数 x クラス数
        X = X_train[batch_indices, :] # データ数 x 特徴量数
        # import pdb; pdb.set_trace()

        # Z計算
        Z = X@W.T + b

        # softmaxで予測計算
        y_pred = softmax(Z)

        # 損失計算
        loss = cross_entropy(y_true_, y_pred)
        loss_log.append(loss.item())
        running_loss += loss.item()
        
        # 勾配計算
        loss.backward()

        # パラメータ更新
        with torch.no_grad():
            W -= learning_rate * W.grad
            b -= learning_rate * b.grad

        # 勾配初期化
        W.grad.zero_()
        b.grad.zero_()
    # validation
    with torch.no_grad():
        Z_val = X_val@W.T + b
        y_pred_val = softmax(Z_val)

        val_loss = cross_entropy(y_val, y_pred_val)

        pred_labels_val = torch.argmax(y_pred_val, dim=-1)
        true_labels_val = torch.argmax(y_val, dim=-1)

        correct = torch.sum(pred_labels_val == true_labels_val).item()
        total = y_val.shape[0]
        val_accuracy = correct / total

    train_losses.append(running_loss/num_batches)
    val_losses.append(val_loss.item())
    val_accuracies.append(val_accuracy)


        
    # 損失ログ出力
    print(f'epoch: {epoch+1}: train loss:{running_loss/num_batches:.4f}, val loss: {val_loss.item():.4f}, val accuracy: {val_accuracy:.4f}')


# %% 損失ログ描画
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid()
plt.plot(train_losses)
plt.plot(val_losses)
plt.legend(["train", "val"])
plt.show()
# %%
