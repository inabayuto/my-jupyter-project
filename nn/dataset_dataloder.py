# DatasetとDataloaderの実装
# %% ライブラリのインポート
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
# %% データセットの読み込み
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
# %%
images, labels = train_dataset[1]
print(images)
plt.imshow(images)
print(len(train_dataset))
# %% Transform
# ToTensor
image_tensor = to_tensor = torchvision.transforms.ToTensor()(images)
type(image_tensor)
plt.imshow(image_tensor.permute(1, 2, 0))
# %% Normalize
normalized_image_tensor = torchvision.transforms.Normalize((0.5,), (0.5,))(image_tensor)
plt.imshow(normalized_image_tensor.permute(1, 2, 0))
print(normalized_image_tensor.min())
print(normalized_image_tensor.max())
# %% Compose
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

plt.imshow(train_dataset[4][0].permute(1, 2, 0))


# %% DataLoader
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=10, num_workers=0)

# %% データローダーの確認
images, labels = next(iter(train_loader))
print(images.shape)

grid = torchvision.utils.make_grid(images)
plt.imshow(grid.permute(1, 2, 0))
print(labels)