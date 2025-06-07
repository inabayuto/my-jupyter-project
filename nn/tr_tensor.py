# Tensorの基礎
# %%
import torch
import numpy as np
torch.manual_seed(42)
# %%
my_list = [1, 2, 3, 4, 5] # リスト
my_tensor = torch.tensor(my_list) # リストをテンソルに変換
print(my_tensor) # テンソルの内容
print(type(my_tensor)) # テンソルの型
print(my_tensor.shape) # テンソルの形状
print(my_tensor.dtype) # テンソルのデータ型

# %%
zeros = torch.zeros((2, 3)) # 2 x 3 のゼロテンソル
print(zeros)
print(zeros.shape)
print(zeros.dtype)

ones = torch.ones((2, 3)) # 2 x 3 の1テンソル
print(ones)
print(ones.shape)
print(ones.dtype)

eye = torch.eye(3) # 3 x 3 の単位テンソル
print(eye)
print(eye.shape)
print(eye.dtype)

randn = torch.randn(2, 3) # 2 x 3 のランダムテンソル
print(randn)
print(randn.shape)
print(randn.dtype)

# Tensorの操作
# %%
tensor = torch.randn((2, 3, 4))
print(tensor.shape)
# 転地
print(torch.permute(tensor, (1, 0, 2)).shape) # (2, 3, 4) -> (3, 2, 4)
print(torch.transpose(tensor, 0, 1).shape) # (2, 3, 4) -> (3, 2, 4)

# 二次元の場合は.T/.t()で転置
# %%
tensor = torch.randn((2, 3)) # 2 x 3 のランダムテンソル
print(tensor.shape) # (2, 3)
print(tensor.T.shape) # (3, 2)
print(tensor.t().shape) # (3, 2)

# reshape
# %%
tensor = torch.randn((2, 3, 4))
print(torch.reshape(tensor, (6, 4)).shape) # (2, 3, 4) -> (6, 4)

# flatten
# %%
tensor = torch.randn((2, 3, 4))
print(tensor.flatten().shape) # (2, 3, 4) -> (24,)

# squeeze
# %%
tensor = torch.tensor([[[1], [2], [3]]])
print(tensor.shape) # (1, 3, 1)
print(torch.squeeze(tensor).shape) # (1, 3, 1) -> (3,)

# unsqueeze
# %%
tensor = torch.tensor([[[1], [2], [3]]])
print(torch.unsqueeze(tensor, dim=0).shape) # (1, 3, 1) -> (1, 1, 3, 1)

# tensorの便利関数 
# %%
tensor = torch.rand((2, 3))
print(tensor.shape) # (2, 3)
# 合計
print(f'合計: {torch.sum(tensor)}')
# 平均値
print(f'平均値: {torch.mean(tensor)}')
# 標準偏差
print(f'標準偏差: {torch.std(tensor)}')
# 最大値
print(f'最大値: {torch.max(tensor)}')
# 最小値
print(f'最小値: {torch.min(tensor)}')
# 最大値のインデックス
print(f'最大値のインデックス: {torch.argmax(tensor)}')
# 最小値のインデックス
print(f'最小値のインデックス: {torch.argmin(tensor)}')

# 行列の演算
# %%
a = torch.rand((3, 3))
b = torch.rand((3, 3))
print(a.shape)
print(b.shape)
# 行列の加算
print(a + b)
# 行列の減算
print(a - b)
# 行列の乗算
print(a * b)
# 行列の除算
print(a / b)
# 行列の積
torch.mm(a, b) == torch.matmul(a, b)
# 行列の逆行列

# Broad casting
# %%
# (3, 3)のスカラーの演算
a = torch.rand((3, 3))
scalar = 2
print(a + scalar)

# (3, 3)と(1, 3)の演算
b = torch.rand((1, 3))
print(a + b)

# (32, 128, 128, 3)と(128, 128, 3)の演算
c = torch.rand((32, 128, 128, 3))
d = torch.rand((128, 128, 3))
print(c + d)

# 
# %%
# テンソルの連結
# %%
a = torch.rand((2, 3))
b = torch.rand((2, 3))
# %%
