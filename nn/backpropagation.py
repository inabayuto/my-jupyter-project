# %% ライブラリのインポート
import torch
from sklearn import datasets
import matplotlib.pyplot as plt
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import numpy as np

# %% ハイパーパラメータ設定と損失ログ用リスト
learning_rate = 0.03
loss_log = []

# %% データロード
dataset = datasets.load_digits()
images = dataset['images']
target = dataset['target']

# %%学習データと検証データ分割
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

# %% Backwardでスクラッチを実装
def linear_backward(A, W, b, Z):
    W.grad = Z.grad.T @ A
    b.grad = torch.sum(Z.grad, dim=0, keepdim=True)
    A.grad = Z.grad @ W

def relu_backward(A, Z):
    return A.grad * (Z > 0).float()

# %% softmaxとcrossentropyを同じ関数にする
def softmax_cross_entropy(x, y_true):
    e_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
    softmax_out =  e_x / (torch.sum(e_x, dim=-1, keepdim=True) + 1e-10)
    loss = -torch.sum(y_true * torch.log(softmax_out + 1e-10)) / y_true.shape[0]
    return loss, softmax_out

# %% モデルの定義
def linear(X, W, b):
    return X@W.T + b
def relu(Z):
    return Z.clamp_min(0.)

# forward_and_backward
def forward_and_backward(X, y):
    # forward
    Z1 = linear(X, W1, b1)
    Z1.retain_grad() # 中間ノードの勾配を保持
    A1 = relu(Z1)
    A1.retain_grad() # 中間ノードの勾配を保持
    Z2 = linear(A1, W2, b2)
    Z2.retain_grad() # 中間ノードの勾配を保持
    loss, A2 = softmax_cross_entropy(Z2, y)

    # backward
    Z2.grad = (A2 - y) / X.shape[0] # バッチサイズで割る
    linear_backward(A1, W2, b2, Z2)
    Z1.grad = relu_backward(A1, Z1)
    linear_backward(X, W1, b1, Z1)

    return loss, Z1, A1, Z2, A2

# %% パラメータの初期化
m, n = X_train.shape
nh = 30
class_num = 10
# パラメータの初期化
# W1 = torch.randn((nh, n), requires_grad=True) # 出力 x 入力
W1 = torch.randn((nh, n)) * torch.sqrt(torch.tensor(2./n))
W1.requires_grad=True # 勾配を計算するために必要
b1 = torch.zeros((1, nh), requires_grad=True) # 1 x 出力

# W2 = torch.randn((class_num, nh), requires_grad=True) # 出力 x 入力
W2 = torch.randn((class_num, nh)) * torch.sqrt(torch.tensor(2./n))
W2.requires_grad=True # 勾配を計算するために必要
b2 = torch.zeros((1, class_num), requires_grad=True) # 1 x 出力

print("Hidden layer weights shape:", W1.shape)
print("Hidden layer biases shape:", b1.shape)
print("Output layer weights shape:", W2.shape)
print("Output layer biases shape:", b2.shape)

# %% モデルの実行
loss, Z1, A1, Z2, A2 = forward_and_backward(X_train, y_train)
print(Z1.grad)
print(A1.grad)
print(Z2.grad)
print(A2.grad)

loss.backward()
print(torch.allclose(W1.grad, W1.grad)) # 勾配が同じかどうか
print(torch.allclose(b1.grad, b1.grad)) # 勾配が同じかどうか
print(torch.allclose(Z1.grad, Z1.grad)) # 勾配が同じかどうか
print(torch.allclose(Z2.grad, Z2.grad)) # 勾配が同じかどうか


# %% モデル構築
batch_size = 30
num_batches = np.ceil(len(y_train) / batch_size).astype(int)
loss_log = []
# ログ
train_losses = []
val_losses = []
val_accuracies = []
# or文で学習ループ作成
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

        Z1 = linear(X, W1, b1)
        A1 = relu(Z1)
        Z2 = linear(A1, W2, b2)
        loss, A2 = softmax_cross_entropy(Z2, y_true_)

        # 損失計算
        loss_log.append(loss.item())
        running_loss += loss.item()
        
        # 勾配計算
        Z2.grad = (A2 - y_true_) / X.shape[0]
        linear_backward(A1, W2, b2, Z2)
        Z1.grad = relu_backward(A1, Z1)
        linear_backward(X, W1, b1, Z1)

        # パラメータ更新
        with torch.no_grad():
            W1 -= learning_rate * W1.grad
            b1 -= learning_rate * b1.grad
            W2 -= learning_rate * W2.grad
            b2 -= learning_rate * b2.grad

        # 勾配初期化
        W1.grad = None
        b1.grad = None
        W2.grad = None
        b2.grad = None
    # validation
    with torch.no_grad():
        Z1_val = linear(X_val, W1, b1)
        A1_val = relu(Z1_val)
        Z2_val = linear(A1_val, W2, b2)
        val_loss, A2_val = softmax_cross_entropy(Z2_val, y_val)

        pred_labels_val = torch.argmax(A2_val, dim=-1)
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



