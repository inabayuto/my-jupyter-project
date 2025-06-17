# %% ライブラリのインポート
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from sklearn import datasets
from sklearn.model_selection import train_test_split


# %% MLPを定義
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        return x

# %% Optimizer
model = MLP(input_size=3072, hidden_size=512, output_size=10)
opt = optim.SGD(model.parameters(), lr=0.001)


# %% Compose
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.CIFAR10(root='./CIFAR10_data', train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR10(root='./CIFAR10_data', train=False, download=True, transform=transform)

# %% DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

# %% データローダーの確認
images, labels = next(iter(train_loader))
print(images.shape)

# %% 学習ループを回す
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(30):
    # エポック毎にデータをシャッフル
    suffled_indices = torch.randperm(len(train_dataset))
    run_loss = 0.0
    run_val_loss = 0.0
    run_val_acc = 0.0

    for train_batch, data in enumerate(train_loader):
        X, y = data

        # forwardとbackwardの計算
        # forward
        opt.zero_grad()
        preds = model(X)
        loss = F.cross_entropy(preds, y)

        # backward
        loss.backward()
        opt.step()

        # ロスの計算
        run_loss += loss.item()

    # validate
    with torch.no_grad():
        for val_batch, data in enumerate(val_loader):
            X_val, y_val = data
            preds_val = model(X_val)
            val_loss = F.cross_entropy(preds_val, y_val)
            run_val_loss += val_loss.item()

            pred_labels_val = torch.argmax(preds_val, dim=-1)
            true_labels_val = y_val  # そのまま
            correct = torch.sum(pred_labels_val == true_labels_val).item()
            total = y_val.shape[0]
            val_accuracy = correct / total
            run_val_acc += val_accuracy

    train_losses.append(run_loss / (train_batch + 1))
    val_losses.append(run_val_loss / (val_batch + 1))
    val_accuracies.append(run_val_acc / (val_batch + 1))
    print(f'epoch: {epoch+1}: train loss: {run_loss / (train_batch + 1):.4f}, val loss: {run_val_loss / (val_batch + 1):.4f}, val accuracy: {run_val_acc / (val_batch + 1):.4f}')

# %%
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.grid()
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
plt.show()
# %% カスタムのDatasetを作成
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]

        if self.transform:
            data = self.transform(data)
        return data, labels

# %% カスタムデータセットの使用例
dataset = datasets.load_digits()
target = dataset['target']
images = dataset['images']
images = images * (255. / 16.) # 0~16の値を0~255に変換
images = images.astype(np.uint8)
X_train, X_val , y_train, y_val = train_test_split(images, target, test_size=0.2, random_state=42)

# %%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_mydatasets = CustomDataset(X_train, y_train, transform=transform)
val_mydatasets = CustomDataset(X_val, y_val, transform=transform)

print(len(train_mydatasets))
print(len(val_mydatasets))

# %%
train_loader = DataLoader(train_mydatasets, batch_size=10, shuffle=True, num_workers=0)
val_loader = DataLoader(val_mydatasets, batch_size=10, shuffle=False, num_workers=0)

# %%
images, labels = next(iter(train_loader))
print(images.shape)

grid = torchvision.utils.make_grid(images)
plt.imshow(grid.permute(1, 2, 0))
print(labels)
