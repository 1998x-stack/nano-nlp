"""LoRA PEFT 回归：train/validation split、冻结检查、MSE 和权重规模。"""
import torch
from torch.nn import functional as F
from loguru import logger
from model import LoRALinear
from data import regression_data

def run(*,steps=30,seed=42,text_file=None):
    torch.manual_seed(seed);torch.set_num_threads(1)
    (x_train,y_train),(x_valid,y_valid)=regression_data(seed)
    layer=LoRALinear(in_dim=8,out_dim=4,rank=2,alpha=4)
    frozen=layer.base.weight.detach().clone()
    optim=torch.optim.AdamW(list(layer.A.parameters())+list(layer.B.parameters()),lr=.07)
    with torch.no_grad():
        baseline=F.mse_loss(layer(x_valid),y_valid).item()
    for step in range(steps):
        loss=F.mse_loss(layer(x_train),y_train)
        optim.zero_grad();loss.backward();optim.step()
        if step in (0,steps-1):logger.info("LoRA iteration={} mse={:.4f}",step+1,loss.item())
    with torch.no_grad():
        pred=layer(x_valid)
        val_loss=F.mse_loss(pred,y_valid).item()
    assert torch.equal(frozen,layer.base.weight.detach()),"基座权重被修改！"
    total=sum(p.numel() for p in layer.parameters())
    trainable=sum(p.numel() for p in layer.parameters() if p.requires_grad)
    rows=[{"target":y.tolist(),"prediction":[round(float(z),4) for z in x]}
          for y,x in zip(y_valid[:5],pred[:5])]
    return {"task":"low_rank_finetuning","train_samples":len(x_train),"valid_samples":len(x_valid),
            "baseline_val_mse":round(baseline,6),"val_mse":round(val_loss,6),
            "trainable_parameters":trainable,"total_parameters":total,
            "trainable_fraction":round(trainable/total,5),"base_unchanged":True},rows
