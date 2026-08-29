import argparse
from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from load_forecasting.data import WAVELET_COLUMNS, build_windows, load_excel
from load_forecasting.model import LoadTransformer


def metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    error = pred - true
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.mean(np.abs(error) / np.maximum(np.abs(true), 1e-6)) * 100),
    }

def model_select(args: argparse.Namespace, feature_names: list[str]) -> nn.Module:
    """模型修改接口，该模型记得改这里，易宸重点检查"""
    if args.model == "LoadTransformer":
        return  LoadTransformer(len(feature_names), args.d_model, args.nhead, args.layers, args.horizon)
    else:
        raise ValueError(f"未知模型: {args.model}")


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("未检测到可用 CUDA GPU")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    print(f"训练设备: {device}")
    frame = load_excel(args.data)
    x, y, feature_names = build_windows(
        frame, args.input_hours, args.horizon, WAVELET_COLUMNS
    )
    split_train = int(len(x) * 0.7)
    split_valid = int(len(x) * 0.85)
    x_train, y_train = x[:split_train], y[:split_train]
    x_valid, y_valid = x[split_train:split_valid], y[split_train:split_valid]
    x_test, y_test = x[split_valid:], y[split_valid:]

    x_mean, x_std = x_train.mean(axis=(0, 1)), x_train.std(axis=(0, 1)) + 1e-6
    y_mean, y_std = y_train.mean(), y_train.std() + 1e-6
    scale = lambda a: (a - x_mean) / x_std
    target_scale = lambda a: (a - y_mean) / y_std

    # 创建数据集
    train_loader = DataLoader(TensorDataset(torch.from_numpy(scale(x_train)), torch.from_numpy(target_scale(y_train))), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(torch.from_numpy(scale(x_valid)), torch.from_numpy(target_scale(y_valid))), batch_size=args.batch_size)

    model = model_select(args, feature_names).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    best = float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss_fn(model(batch_x.to(device)), batch_y.to(device)).backward()
            optimizer.step()
        model.eval()

        with torch.no_grad():
            valid_loss = np.mean([loss_fn(model(a.to(device)), b.to(device)).item() for a, b in valid_loader])
        if valid_loss < best:
            best = valid_loss
            # 间隔100步保存一次模型
            if epoch == 1 or epoch % args.save_every == 0:
                torch.save({"model": model.state_dict(), "feature_names": feature_names, "wavelet_columns": WAVELET_COLUMNS, "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std, "input_hours": args.input_hours, "horizon": args.horizon, "d_model": args.d_model, "nhead": args.nhead, "layers": args.layers}, output)
        if epoch == 1 or epoch % args.log_every == 0:
            print(f"epoch={epoch} valid_mse={valid_loss:.6f}")

    checkpoint = torch.load(output, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(scale(x_test)).to(device)).cpu().numpy() * y_std + y_mean
    print(json.dumps(metrics(pred, y_test), ensure_ascii=False))
    np.savetxt(output.with_suffix(".csv"), np.column_stack([y_test, pred]), delimiter=",", header="真实值_24步,预测值_24步", comments="")


if __name__ == "__main__":
    config = Namespace(
        data="附件3：训练数据集/训练数据项目A历史数据_2025-04-01_2025-10-31.xlsx",
        output="outputs/load_transformer.pt",

        model = "LoadTransformer",
        input_hours=168,
        horizon=24,
        epochs=30,
        batch_size=64,
        d_model=64,
        nhead=4,
        layers=2,
        lr=1e-3,
        save_every=20,
        log_every=5,
        device="cuda",
    )
    
    run(config)
