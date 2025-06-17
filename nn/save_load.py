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
        self.dropout = nn.Dropout(0.2)  # 軽めのドロップアウトを追加
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(torch.relu(self.l1(x)))
        x = self.l2(x)
        return x

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

# %% データセットの作成
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_mydatasets = CustomDataset(X_train, y_train, transform=transform)
val_mydatasets = CustomDataset(X_val, y_val, transform=transform)

print(len(train_mydatasets))
print(len(val_mydatasets))

# %% DataLoader
train_loader = DataLoader(train_mydatasets, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_mydatasets, batch_size=32, shuffle=False, num_workers=0)

# %% Optimizer
model = MLP(input_size=64, hidden_size=64, output_size=10)  # hidden_sizeを小さくする
opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # weight_decayを追加し、学習率を下げる

# early stopping
early_stopping_epochs = 5

# %% 学習ループを回す
def train_model(model, train_loader, val_loader, opt, num_epochs, early_stopping_epochs): 
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    no_improvement_epochs = 0 # ベストモデルを更新しなかったエポック数

    for epoch in range(num_epochs):  # num_epochsを使用
        model.train()
        run_loss = 0.0
        run_val_loss = 0.0
        run_val_acc = 0.0

        for train_batch, data in enumerate(train_loader):
            X, y = data

            # forwardとbackwardの計算
            opt.zero_grad()
            preds = model(X)
            loss = F.cross_entropy(preds, y)
            loss.backward()
            opt.step()

            run_loss += loss.item()

        # validate
        model.eval()  # 評価モードに設定
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

        if run_val_loss < best_val_loss:
            best_val_loss = run_val_loss
            no_improvement_epochs = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improvement_epochs += 1
        
        if epoch > early_stopping_epochs and no_improvement_epochs >= early_stopping_epochs:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

 # %% 学習
 train_model(model, train_loader, val_loader, opt, num_epochs=30, early_stopping_epochs=5)

 # %% モデルの読み込み
 model.load_state_dict(torch.load('best_model.pth'))

 # %% モデルの評価
 model.eval()
 with torch.no_grad():

# %%
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.grid()
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
plt.show()   

# %%
