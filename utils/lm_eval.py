"""字符语言模型的共享评估协议；不掩盖各章的模型 forward。

v2 的三个问题：
1) 使用全量文本建词表，会在训练前看见 held-out 验证字符；
2) range(0,len(val)-BLOCK-1,BLOCK) 丢弃尾部的验证目标；
3) 直接平均各窗口 CE 会让不等长窗口贡献相同权重。

v3 先分割文本，只用训练文本建词表（含固定 <unk>），逐个窗口按
标签 token 数加权汇总。保持每个窗口的上下文严格位于验证段内部。
"""
from __future__ import annotations

UNK = "<unk>"

def encode_train_validation(text: str, *, block_size: int, validation_ratio: float = .2):
    """返回 (chars, train_ids, validation_ids, val_oov_count)。

    未知验证字符统一编码成 0，保留一个可学习的训练输出类别。
    <unk> 是一个特殊输出 token，不是由原始文本字符产生。
    """
    import torch
    if not isinstance(text, str) or not 0 < validation_ratio < 1:
        raise ValueError("文本必须是 str，validation_ratio 必须在 (0,1)")
    if block_size < 1:
        raise ValueError("block_size 必须 >= 1")
    pivot = int(len(text)*(1-validation_ratio))
    train_text, validation_text = text[:pivot], text[pivot:]
    if len(train_text) < block_size + 2 or len(validation_text) < block_size + 2:
        raise ValueError("训练和验证段至少各有 block_size+2 个字符")
    # 仅训练段允许贡献词表：验证中出现的新字符只能落入 <unk>。
    chars = [UNK] + sorted(set(train_text))
    stoi = {token: idx for idx, token in enumerate(chars)}
    train_ids = torch.tensor([stoi[c] for c in train_text], dtype=torch.long)
    valid_ids = torch.tensor([stoi.get(c, 0) for c in validation_text], dtype=torch.long)
    oov_count = sum(c not in stoi for c in validation_text)
    return chars, train_ids, valid_ids, oov_count

def heldout_token_nll(model, validation_ids, *, block_size: int):
    """穷尽验证段的 len(validation_ids)-1 个 next-token 标签。

    按有效标签数量加权，而非简单平均每段 loss。返回平均 NLL、
    实际评测 token 总数以及逐窗口 {"tokens", "nll"} 列表。
    """
    import torch
    if validation_ids.ndim != 1 or len(validation_ids) < 2:
        raise ValueError("validation_ids 必须是至少 2 个 ID 的一维 tensor")
    if block_size < 1:
        raise ValueError("block_size >= 1")
    old_training = model.training
    model.eval()
    total_nll, n_targets, chunks = 0.0, 0, []
    try:
        with torch.no_grad():
            for start in range(0, len(validation_ids)-1, block_size):
                x = validation_ids[start:start+block_size][None, :]
                y = validation_ids[start+1:start+block_size+1][None, :]
                if x.numel() != y.numel():
                    raise AssertionError("next-token input and labels must align")
                _, loss = model(x, y)
                if not torch.isfinite(loss):
                    raise ValueError("验证集 loss 非有限数")
                count = int(y.numel())
                total_nll += float(loss.item()) * count
                n_targets += count
                chunks.append({"tokens": count, "nll": float(loss.item())})
    finally:
        model.train(old_training)
    if n_targets != len(validation_ids)-1:
        raise AssertionError("held-out 验证 token 丢失")
    return total_nll / n_targets, n_targets, chunks
