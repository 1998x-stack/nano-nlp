"""Scaling Laws 的可计算直觉；所有系数都是教学假设。"""
import argparse

def synthetic_loss(n_params, n_tokens):
    """E、A、B、alpha、beta 为自定义玩具参数，非任何模型实测值。"""
    return 1.0 + 0.8 / n_params**0.32 + 0.5 / n_tokens**0.28

def search_budget(budget=4096):
    choices = []
    for n in (2, 4, 8, 16, 32, 64, 128, 256):
        for d in (2, 4, 8, 16, 32, 64, 128, 256):
            if n * d <= budget:
                choices.append((synthetic_loss(n, d), n, d))
    return min(choices), sorted(choices)[:5]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=4096)
    best, top = search_budget(p.parse_args().budget)
    print("RESULT:", {"best_loss": round(best[0], 4),
                      "params_units": best[1], "tokens_units": best[2]})
