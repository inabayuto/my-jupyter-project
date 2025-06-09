# %%
# Autograd 
import torch

# %%
x = torch.tensor(2., requires_grad=True)
y = torch.tensor(3., requires_grad=True)
# %%
# import pdb; pdb.set_trace()
print(x.dtype)
print(y.dtype)

# %%
z = y * torch.log(x) + torch.sin(y)
z.backward()

print(x.grad)
print(y.grad)

# %%
# 中間ノードの勾配計算
x = torch.tensor(2., requires_grad=True)
y = torch.tensor(3., requires_grad=True)
sum_xy = x + y
sum_xy.retain_grad()

z = sum_xy * 2
z.backward()

print(sum_xy.grad)
print(x.grad)
print(y.grad)

# %%
# with torch.no_grad():
x = torch.tensor(2., requires_grad=True)
y = torch.tensor(3., requires_grad=True)

with torch.no_grad():
    z = y * torch.log(x) + torch.sin(y)
# %%
