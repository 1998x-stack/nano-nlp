"""LoRA 回归：训练与验证输入互不相同，可独立计算 MSE。"""
import torch

def regression_data(seed=42):
    gen=torch.Generator().manual_seed(seed)
    x=torch.randn(64,8,generator=gen)
    target=torch.stack([x[:,0]*2,x[:,1]*(-3),x[:,2]+x[:,3],x[:,4]],dim=1)
    return (x[:48],target[:48]),(x[48:],target[48:])
