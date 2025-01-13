import torch
tensor = torch.tensor([[1., 2.], [3., 4.]])
mean_value_dim0 = tensor.mean(dim=0).squeeze()  # 计算每列的平均值
print(mean_value_dim0.shape)  # 输出：torch.Size([2])
print(mean_value_dim0)

mean_value_dim1 = tensor.mean(dim=1).squeeze().mean(dim=0).squeeze()  # 计算每行的平均值
print(mean_value_dim1.shape)  # 输出：torch.Size([2])
print(mean_value_dim1)
