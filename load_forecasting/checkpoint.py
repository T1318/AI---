"""跨 PyTorch 版本加载本项目自己生成的可信检查点。"""
import torch


def load_checkpoint(path, map_location=None):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        # PyTorch 旧版本尚不支持 weights_only 参数。
        return torch.load(path, map_location=map_location)
