# MNISTで多項ロジスティック回帰
# %%
from sklearn import datasets
from torch.nn import functional as F
import torch
import matplotlib.pyplot as plt

learning_rate = 0.03
loss_log = []
# %%
datasets =  datasets.load_digits()
images = datasets["images"]
targets = datasets["target"]

# %%
print(images.shape)
print(targets.shape)

# %%
# 前処理
# ラベルのone-hotエンコーディング
y_true = F.one_hot(torch.tensor(targets), num_classes=10)
images = torch.tensor(images, dtype=torch.float32).reshape(-1, 64) # 64次元に変換

# 画像の標準化
images = (images - images.mean()) / images.std()
# %%
# パラメータの初期化
W = torch.randn((10, 64), requires_grad=True) # 出力 × 入力
b = torch.zeros((1, 10), requires_grad=True) # 1 ×出力

print(W.shape)
print(b.shape)

# %%
# softmaxとcross entropy
def softmax(x):
    e_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
    return e_x / (torch.sum(e_x, dim=-1, keepdim=True) + 1e-10)

def cross_entropy(y_true, y_pred):
    return -torch.sum(y_true * torch.log(y_pred + 1e-10)) / y_true.shape[0]

# %%
# for文で学習ループ作成
for epoch in range(10):
    running_loss = 0
    for i in range(len(targets)):
        # 入力データXおよび教師ラベルのYを作成
        y_true_ = y_true[i].reshape(-1, 10) # データ数xクラス数
        X = images[i].reshape(-1, 64) # データ数 x 特徴量数

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

    # 損失ログ出力
    print(f'epoch: {epoch+1}: {running_loss/len(targets)}')
# %%
# 学習したモデルで全データのaccuracyを計算する（学習に使っているデータに対してのAccuracyであることに注意）
X = torch.tensor(images, dtype=torch.float32)
Z = X@W.T + b
y_pred = softmax(Z)
# accuracy = 正しく分類できた数/全サンプル数
correct = torch.sum(torch.argmax(y_pred, dim=-1) == torch.argmax(y_true, dim=-1))
print(f'correct: {correct.item()}')
print(f'total: {y_true.shape[0]}')
print(f'accuracy: {correct.item() / y_true.shape[0]}')
plt.plot(loss_log)
plt.show()
# %%
