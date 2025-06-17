# %% ライブラリのインポート
import torch
from sklearn import datasets
import matplotlib.pyplot as plt
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import numpy as np
from torch import nn
# %%

# %% nn.Linearの実装
linear = nn.Linear(in_features=64, out_features=30)
list(linear.parameters())
# %%
print(linear.weight.data.shape)
print(linear.bias.data.shape)
# %%
x = torch.randn(5, 64)
z = linear(x)
print(z.shape)

# %% MLPの実装
# クラスと関数を組み合わせたモデル
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x):
        z1 = self.linear1(x)
        a1 = F.relu(z1)
        z2 = self.linear2(a1)
        return z2

mlp = MLP(in_features=64, hidden_features=30, out_features=10)
X = torch.randn(5, 64)
Z = mlp(X)
print(Z.shape)
# %% クラスのみで構成されたモデル
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        z1 = self.linear1(x)
        a1 = self.relu(z1)
        z2 = self.linear2(a1)
        return z2

mlp = MLP(in_features=64, hidden_features=30, out_features=10)
X = torch.randn(5, 64)
Z = mlp(X)
print(Z.shape)
# %% nn.Sequentialの実装
mlp =nn.Sequential(
    nn.Linear(in_features=64, out_features=30),
    nn.ReLU(),
    nn.Linear(in_features=30, out_features=10)
)
Z = mlp(X)
print(Z.shape)

# %%
dataset = datasets.load_digits()
data = dataset['data']
target = dataset['target']
images = dataset['images']

X_train, X_val, y_train, y_val = train_test_split(images, target, test_size=0.2, random_state=42)
X_train_mean = X_train.mean()
X_train_std = X_train.std()

X_train = (X_train - X_train_mean) / X_train_std
X_val = (X_val - X_train_mean) / X_train_std

X_train = torch.tensor(X_train.reshape(-1, 64), dtype=torch.float32)
X_val = torch.tensor(X_val.reshape(-1, 64), dtype=torch.float32)
y_train = torch.tensor(y_train) 
y_val = torch.tensor(y_val) 

# %% 
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
# %%
batch_size = 30
learning_rate = 0.03

# モデルの初期化
mlp = MLP(in_features=64, hidden_features=30, out_features=10)

# ログ
loss_log = []
train_losses = []
val_losses = []
val_accuracies = []
# 学習ループ
batch_size = 30
num_batches = np.ceil(len(y_train) / batch_size).astype(int)
epochs = 30

for epoch in range(epochs):
    shuffled_indices = np.random.permutation(len(y_train))
    running_loss = 0
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_indices = shuffled_indices[start:end]
        y = y_train[batch_indices]
        X = X_train[batch_indices]

        pred = mlp(X)
        loss = F.cross_entropy(pred, y)

        # 損失計算
        loss_log.append(loss.item())
        running_loss += loss.item()

        # 勾配計算
        loss.backward()
        with torch.no_grad():
            for param in mlp.parameters():
                param -= learning_rate * param.grad
        mlp.zero_grad()
        
       # validation
    with torch.no_grad():
        pred_val = mlp(X_val)
        val_loss = F.cross_entropy(pred_val, y_val)
        loss_log.append(val_loss.item())
        running_loss += val_loss.item()

        pred_labels_val = torch.argmax(pred_val, dim=-1)
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
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.grid()
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
plt.show()


# %%
