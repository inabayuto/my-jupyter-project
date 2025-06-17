# %% 必要なライブラリのインポート
from sklearn import datasets
from torch.nn import functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np


# %% ハイパーパラメータ設定と損失ログ用リスト
learning_rate = 0.03
loss_log = []

# %% データ読み込みと前処理
digits = datasets.load_digits()
images = digits["images"]
targets = digits["target"]

print(images.shape)  # (1797, 8, 8)
print(targets.shape) # (1797,)

# ラベルのone-hotエンコーディング
y_true = F.one_hot(torch.tensor(targets), num_classes=10)

# 画像を64次元のベクトルに変換し、float32に変換
images = torch.tensor(images, dtype=torch.float32).reshape(-1, 64)

# 画像の標準化（全体平均・分散で）
images = (images - images.mean()) / images.std()

# %% パラメータ初期化
W = torch.randn((10, 64), requires_grad=True)  # 出力 × 入力
b = torch.zeros((1, 10), requires_grad=True)   # バイアス

print(W.shape)
print(b.shape)

# %% SoftmaxとCross Entropy関数の定義
def softmax(x):
    e_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
    return e_x / (torch.sum(e_x, dim=-1, keepdim=True) + 1e-10)

def cross_entropy(y_true, y_pred):
    return -torch.sum(y_true * torch.log(y_pred + 1e-10)) / y_true.shape[0]

# %% 学習ループ
loss_log = []
batch_size = 30
num_batches = np.ceil(len(targets) / batch_size).astype(int)
# パラメータの初期化
W = torch.rand((10, 64), requires_grad=True) # 出力 x 入力
b = torch.rand((1, 10), requires_grad=True) # 1 x 出力
# for文で学習ループ作成
for epoch in range(5):
    shuffled_indices = np.random.permutation(len(targets))
    running_loss = 0
    for i in range(num_batches):

        # mini batch作成
        start = i * batch_size
        end = start + batch_size
        batch_indices = shuffled_indices[start:end]
        # 入力データXおよび教師ラベルのYを作成
        y_true_ = y_true[batch_indices, :] # データ数 x クラス数
        X = images[batch_indices, :] # データ数 x 特徴量数
        # ブレークポイントを設置
        # import pdb; pdb.set_trace()

        # 7. Z計算
        Z = X@W.T + b

        # 8. softmaxで予測計算
        y_pred = softmax(Z)

        # 9. 損失計算
        loss = cross_entropy(y_true_, y_pred)
        loss_log.append(loss.item())
        running_loss += loss.item()
        
        # 10. 勾配計算
        loss.backward()

        # 11. パラメータ更新
        with torch.no_grad():
            W -= learning_rate * W.grad
            b -= learning_rate * b.grad

        # 12. 勾配初期化
        W.grad.zero_()
        b.grad.zero_()

    # 13. 損失ログ出力
    print(f'epoch: {epoch+1}: LOSS = {running_loss/num_batches:.4f}')

# %% モデル評価（訓練データ上の精度）
X = torch.tensor(images, dtype=torch.float32)
Z = X @ W.T + b
y_pred = softmax(Z)
pred_labels = torch.argmax(y_pred, dim=1)
true_labels = torch.argmax(y_true, dim=1)

correct = torch.sum(pred_labels == true_labels).item()
total = y_true.shape[0]
accuracy = correct / total

print(f'Correct: {correct}')
print(f'Total: {total}')
print(f'Accuracy: {accuracy:.4f}')

# %% 損失の推移を可視化
plt.plot(loss_log)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid()
plt.show()

# %%
shuffled_indices = np.random.permutation(len(targets))
print(shuffled_indices)
batch_size = 30
num_batches = np.ceil(len(targets) / batch_size).astype(int)
print(num_batches)

for i in range(num_batches):
# mini batch作成
    start = i * batch_size
    end = start + batch_size
    batch_indices = shuffled_indices[start:end]
    # print(batch_indices)
print(start)
print(end)
# %%
