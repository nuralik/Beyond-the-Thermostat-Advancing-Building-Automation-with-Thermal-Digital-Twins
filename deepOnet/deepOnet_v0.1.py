from __future__ import annotations

"""
ThermoDT Step 1 — single-file baseline
======================================

A single consolidated script for training and evaluating the current DeepONet
baseline for masked sensor reconstruction. Edit the PARAMS block below and run:

    python thermodt_step1_singlefile.py

Choose PARAMS["mode"] = "train", "eval", or "train_eval".

Design notes
------------
- Missing sensors are represented with BOTH a fill value and an explicit
  observed/missing mask appended to the branch input.
- The checkpoint stores the normalization stats, splits, and geometry needed for
  later evaluation, so there is no separate metadata JSON to manage.
- The default setup matches the current Step 1 baseline direction: Lab1,
  5-minute data, temperature only, blocked test windows, blocked validation.
"""

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


PARAMS: Dict[str, Any] = {
    "mode": "train_eval",  # "train", "eval", "train_eval"
    "seed": 42,
    "device": "auto",     # "auto", "cpu", "cuda", "mps"
    "experiment_name": "lab1_temperature_step1_singlefile",
    "data_root": "./data",
    "lab": "lab1",
    "resolution": "5min",
    "time_file": "Time_5min.xlsx",
    "time_column": "tin",
    "scale": 100.0,
    "target_variables": [
        {
            "name": "temperature",
            "file": "Temperature_5min.xlsx",
            "column_prefix": "tem",
        }
    ],
    "sensor_ids": list(range(1, 14)),
    "room_width": 7.81,
    "room_height": 7.82,
    "sensor_coords": {
        1: (7.21, 0.60),  2: (3.88, 5.88),  3: (5.85, 5.88),
        4: (0.60, 0.60),  5: (1.92, 1.94),  6: (3.88, 3.91),
        7: (1.92, 3.91),  8: (5.85, 3.91),  9: (7.21, 7.22),
        10: (1.92, 5.88), 11: (0.60, 7.22), 12: (5.85, 1.94),
        13: (3.88, 1.94),
    },
    "test_windows": [
        ("2022-06-15", 8),
        ("2022-06-20", 14),
        ("2022-07-06", 10),
        ("2022-07-10", 20),
        ("2022-08-05", 6),
        ("2022-08-25", 12),
        ("2022-09-04", 16),
        ("2022-09-12", 22),
    ],
    "test_window_hours": 6,
    "val_mode": "blocked_windows",  # "blocked_windows" or "random"
    "val_fraction": 0.15,
    "val_num_windows": 8,
    "val_window_hours": 6,
    "include_observed_mask": True,
    "mask_fill_value": 0.0,
    "train_extra_mask_min": 1,
    "train_extra_mask_max": 1,
    "eval_masked_total": [1, 3, 6],
    "eval_repeats": 3,
    "hidden": 128,
    "latent_dim": 128,
    "depth": 4,
    "activation": "tanh",  # "tanh", "relu", "gelu"
    "batch_size": 4096,
    "epochs": 80,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "num_workers": 0,
    "output_dir": "./deepOnet/runs/s0data_v0.1",
    "best_checkpoint_name": "thermodt_best.pt",
    "final_checkpoint_name": "thermodt_final.pt",
    "eval_csv_name": "thermodt_eval_metrics.csv",
    "eval_summary_name": "thermodt_eval_summary.json",
    "checkpoint_to_evaluate": "best",  # "best", "final", or full path
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_data_dir(cfg: Dict[str, Any]) -> Path:
    return (Path(cfg["data_root"]) / f"{cfg['lab']}").resolve()


def build_window_mask(time_idx: pd.DatetimeIndex, windows: Sequence[Sequence[Any]], hours: int) -> np.ndarray:
    mask = np.zeros(len(time_idx), dtype=bool)
    for date_str, start_hour in windows:
        start = pd.Timestamp(f"{date_str} {int(start_hour):02d}:00:00")
        end = start + pd.Timedelta(hours=hours)
        mask |= np.asarray((time_idx >= start) & (time_idx < end), dtype=bool)
    return mask


def infer_steps_per_hour(time_idx: pd.DatetimeIndex) -> int:
    if len(time_idx) < 2:
        return 1
    diffs_ns = np.diff(time_idx.view("i8"))
    median_seconds = float(np.median(diffs_ns) / 1e9)
    if median_seconds <= 0:
        return 1
    return max(1, int(round(3600.0 / median_seconds)))


def choose_blocked_validation_windows(
    time_idx: pd.DatetimeIndex,
    occupied_mask: np.ndarray,
    num_windows: int,
    window_hours: int,
    seed: int,
) -> List[Tuple[str, int]]:
    rng = np.random.default_rng(seed)
    steps_per_hour = infer_steps_per_hour(time_idx)
    window_steps = max(1, steps_per_hour * int(window_hours))

    candidate_starts: List[int] = []
    for start_idx in range(0, max(0, len(time_idx) - window_steps + 1)):
        end_idx = start_idx + window_steps
        if occupied_mask[start_idx:end_idx].any():
            continue
        candidate_starts.append(start_idx)
    if not candidate_starts:
        return []

    anchors = np.linspace(0, len(candidate_starts) - 1, num=min(max(1, num_windows * 3), len(candidate_starts)), dtype=int)
    ordered_candidates = [candidate_starts[i] for i in anchors]
    rng.shuffle(ordered_candidates)
    ordered_candidates.extend(candidate_starts)

    selected: List[int] = []
    taken = occupied_mask.copy()
    for start_idx in ordered_candidates:
        end_idx = min(len(time_idx), start_idx + window_steps)
        if taken[start_idx:end_idx].any():
            continue
        selected.append(start_idx)
        taken[start_idx:end_idx] = True
        if len(selected) >= num_windows:
            break

    windows: List[Tuple[str, int]] = []
    for idx in sorted(selected):
        ts = pd.Timestamp(time_idx[idx])
        windows.append((ts.strftime("%Y-%m-%d"), int(ts.hour)))
    return windows


@dataclass
class SplitIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    test_mask: np.ndarray
    val_mask: np.ndarray
    split_meta: Dict[str, Any]


@dataclass
class LoadedData:
    data_norm: np.ndarray
    time_idx: pd.DatetimeIndex
    coords_norm: np.ndarray
    sensor_ids: List[int]
    variable_names: List[str]
    mean: np.ndarray
    std: np.ndarray
    splits: SplitIndices


@dataclass
class Batch:
    branch_input: torch.Tensor
    trunk_coord: torch.Tensor
    target: torch.Tensor


def collate_batches(items: Sequence[Batch]) -> Batch:
    return Batch(
        branch_input=torch.stack([x.branch_input for x in items], dim=0),
        trunk_coord=torch.stack([x.trunk_coord for x in items], dim=0),
        target=torch.stack([x.target for x in items], dim=0),
    )


def split_indices(time_idx: pd.DatetimeIndex, cfg: Dict[str, Any]) -> SplitIndices:
    test_mask = build_window_mask(time_idx, cfg["test_windows"], int(cfg["test_window_hours"]))

    if cfg["val_mode"] == "blocked_windows":
        val_windows = choose_blocked_validation_windows(
            time_idx=time_idx,
            occupied_mask=test_mask,
            num_windows=int(cfg["val_num_windows"]),
            window_hours=int(cfg["val_window_hours"]),
            seed=int(cfg["seed"]),
        )
        val_mask = build_window_mask(time_idx, val_windows, int(cfg["val_window_hours"]))
    elif cfg["val_mode"] == "random":
        rng = np.random.default_rng(int(cfg["seed"]))
        available = np.where(~test_mask)[0]
        n_val = max(1, int(len(available) * float(cfg["val_fraction"])))
        chosen = rng.choice(available, size=n_val, replace=False)
        val_mask = np.zeros(len(time_idx), dtype=bool)
        val_mask[chosen] = True
        val_windows = []
    else:
        raise ValueError(f"Unsupported val_mode: {cfg['val_mode']}")

    val_mask &= ~test_mask
    train_mask = ~(test_mask | val_mask)
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    split_meta = {
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "val_mode": cfg["val_mode"],
        "test_windows": [
            {
                "date": d,
                "start_hour": int(h),
                "start": f"{d} {int(h):02d}:00",
                "end": str(pd.Timestamp(f"{d} {int(h):02d}:00") + pd.Timedelta(hours=int(cfg['test_window_hours']))),
            }
            for d, h in cfg["test_windows"]
        ],
        "val_windows": [
            {
                "date": d,
                "start_hour": int(h),
                "start": f"{d} {int(h):02d}:00",
                "end": str(pd.Timestamp(f"{d} {int(h):02d}:00") + pd.Timedelta(hours=int(cfg['val_window_hours']))),
            }
            for d, h in val_windows
        ],
    }
    return SplitIndices(train_idx, val_idx, test_idx, test_mask, val_mask, split_meta)


def load_data(cfg: Dict[str, Any]) -> LoadedData:
    data_dir = resolve_data_dir(cfg)
    sensor_ids = [int(x) for x in cfg["sensor_ids"]]
    time_df = pd.read_excel(data_dir / cfg["time_file"])
    time_idx = pd.to_datetime(time_df[cfg["time_column"]])

    variable_arrays: List[np.ndarray] = []
    variable_names: List[str] = []
    for var_cfg in cfg["target_variables"]:
        df = pd.read_excel(data_dir / var_cfg["file"])
        cols = [f"{var_cfg['column_prefix']}{sid}" for sid in sensor_ids]
        arr = np.stack([df[col].to_numpy(dtype=float) for col in cols], axis=1) / float(cfg["scale"])
        variable_arrays.append(arr)
        variable_names.append(str(var_cfg["name"]))

    data_raw = np.stack(variable_arrays, axis=-1)
    valid_mask = ~np.isnan(data_raw).any(axis=(1, 2))
    data_raw = data_raw[valid_mask]
    time_idx = pd.DatetimeIndex(np.asarray(time_idx)[valid_mask])

    coords_norm = np.array(
        [
            (
                cfg["sensor_coords"][sid][0] / float(cfg["room_width"]),
                cfg["sensor_coords"][sid][1] / float(cfg["room_height"]),
            )
            for sid in sensor_ids
        ],
        dtype=np.float32,
    )

    splits = split_indices(time_idx, cfg)
    train_data = data_raw[splits.train_idx]
    mean = train_data.mean(axis=(0, 1)).astype(np.float32)
    std = train_data.std(axis=(0, 1)).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    data_norm = (data_raw - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)

    return LoadedData(
        data_norm=data_norm.astype(np.float32),
        time_idx=time_idx,
        coords_norm=coords_norm,
        sensor_ids=sensor_ids,
        variable_names=variable_names,
        mean=mean,
        std=std,
        splits=splits,
    )


class MultiSensorMaskedDataset(Dataset):
    def __init__(self, data_norm: np.ndarray, coords_norm: np.ndarray, mask_fill_value: float, include_observed_mask: bool, extra_mask_min: int, extra_mask_max: int, seed: int):
        self.data_norm = np.asarray(data_norm, dtype=np.float32)
        self.coords_norm = np.asarray(coords_norm, dtype=np.float32)
        self.mask_fill_value = float(mask_fill_value)
        self.include_observed_mask = bool(include_observed_mask)
        self.extra_mask_min = int(extra_mask_min)
        self.extra_mask_max = int(extra_mask_max)
        self.base_seed = int(seed)
        self.n_time, self.n_sensors, self.n_channels = self.data_norm.shape

    def __len__(self) -> int:
        return self.n_time * self.n_sensors

    def __getitem__(self, idx: int) -> Batch:
        t = idx // self.n_sensors
        s = idx % self.n_sensors
        rng = np.random.default_rng(self.base_seed + idx)

        values = self.data_norm[t].copy()
        observed = np.ones((self.n_sensors,), dtype=np.float32)
        if self.extra_mask_max >= self.extra_mask_min:
            extra_count = int(rng.integers(self.extra_mask_min, self.extra_mask_max + 1))
        else:
            extra_count = 0
        extra_count = max(0, min(extra_count, self.n_sensors - 1))

        mask_ids = [s]
        if extra_count > 0:
            others = np.array([i for i in range(self.n_sensors) if i != s], dtype=int)
            extra_ids = rng.choice(others, size=extra_count, replace=False).tolist()
            mask_ids.extend(extra_ids)

        values[mask_ids, :] = self.mask_fill_value
        observed[mask_ids] = 0.0

        branch_parts = [values.reshape(-1)]
        if self.include_observed_mask:
            branch_parts.append(observed)
        branch_input = np.concatenate(branch_parts, axis=0).astype(np.float32)
        target = self.data_norm[t, s, :].astype(np.float32)
        return Batch(torch.from_numpy(branch_input), torch.from_numpy(self.coords_norm[s]), torch.from_numpy(target))


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int, activation: str):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}.get(activation.lower())
        if act_cls is None:
            raise ValueError(f"Unsupported activation: {activation}")
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden), act_cls()]
        for _ in range(depth - 2):
            layers.extend([nn.Linear(hidden, hidden), act_cls()])
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiOutputDeepONet(nn.Module):
    def __init__(self, branch_in_dim: int, coord_dim: int, output_dim: int, hidden: int, latent_dim: int, depth: int, activation: str):
        super().__init__()
        self.output_dim = int(output_dim)
        self.latent_dim = int(latent_dim)
        branch_out_dim = self.output_dim * self.latent_dim
        trunk_out_dim = self.output_dim * self.latent_dim
        self.branch = MLP(branch_in_dim, hidden, branch_out_dim, depth, activation)
        self.trunk = MLP(coord_dim, hidden, trunk_out_dim, depth, activation)
        self.bias = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self, branch_input: torch.Tensor, trunk_coord: torch.Tensor) -> torch.Tensor:
        branch = self.branch(branch_input).view(-1, self.output_dim, self.latent_dim)
        trunk = self.trunk(trunk_coord).view(-1, self.output_dim, self.latent_dim)
        return (branch * trunk).sum(dim=-1) + self.bias


def build_model(cfg: Dict[str, Any], n_sensors: int, n_channels: int) -> MultiOutputDeepONet:
    branch_in_dim = n_sensors * n_channels
    if cfg["include_observed_mask"]:
        branch_in_dim += n_sensors
    return MultiOutputDeepONet(branch_in_dim, 2, n_channels, int(cfg["hidden"]), int(cfg["latent_dim"]), int(cfg["depth"]), str(cfg["activation"]))


def build_dataloaders(cfg: Dict[str, Any], loaded: LoadedData, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    train_ds = MultiSensorMaskedDataset(loaded.data_norm[loaded.splits.train_idx], loaded.coords_norm, float(cfg["mask_fill_value"]), bool(cfg["include_observed_mask"]), int(cfg["train_extra_mask_min"]), int(cfg["train_extra_mask_max"]), int(cfg["seed"]))
    val_ds = MultiSensorMaskedDataset(loaded.data_norm[loaded.splits.val_idx], loaded.coords_norm, float(cfg["mask_fill_value"]), bool(cfg["include_observed_mask"]), 0, 0, int(cfg["seed"]) + 9999)
    pin_memory = str(device) != "cpu"
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=int(cfg["num_workers"]), pin_memory=pin_memory, collate_fn=collate_batches)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=int(cfg["num_workers"]), pin_memory=pin_memory, collate_fn=collate_batches)
    return train_loader, val_loader


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        pred = model(batch.branch_input.to(device), batch.trunk_coord.to(device))
        target = batch.target.to(device)
        loss = criterion(pred, target)
        total_loss += loss.item() * target.shape[0]
        total_count += target.shape[0]
    return float(total_loss / max(1, total_count))


def package_checkpoint_payload(cfg: Dict[str, Any], loaded: LoadedData, model: nn.Module, epoch: int, train_loss: float, val_loss: float, history: Dict[str, List[float]]) -> Dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "history": history,
        "config": copy.deepcopy(cfg),
        "stats": {"mean": loaded.mean.tolist(), "std": loaded.std.tolist(), "scale": [float(cfg["scale"]) for _ in loaded.variable_names]},
        "splits": loaded.splits.split_meta,
        "sensors": {
            "sensor_ids": loaded.sensor_ids,
            "coords_norm": loaded.coords_norm.tolist(),
            "sensor_coords": {int(k): list(v) for k, v in cfg["sensor_coords"].items()},
            "room_width": float(cfg["room_width"]),
            "room_height": float(cfg["room_height"]),
        },
        "variables": loaded.variable_names,
        "time": {"n_timesteps": int(len(loaded.time_idx)), "start": str(loaded.time_idx[0]) if len(loaded.time_idx) else None, "end": str(loaded.time_idx[-1]) if len(loaded.time_idx) else None},
    }


def train(cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    set_seed(int(cfg["seed"]))
    device = get_device(str(cfg["device"]))
    out_dir = Path(cfg["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_data(cfg)
    train_loader, val_loader = build_dataloaders(cfg, loaded, device)
    model = build_model(cfg, len(loaded.sensor_ids), len(loaded.variable_names)).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=int(cfg["epochs"]))
    criterion = nn.MSELoss()
    best_path = out_dir / str(cfg["best_checkpoint_name"])
    final_path = out_dir / str(cfg["final_checkpoint_name"])
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": [], "lr": []}

    print("=" * 76)
    print(f"ThermoDT Step 1 | experiment={cfg['experiment_name']} | device={device}")
    print("=" * 76)
    print(f"Variables         : {loaded.variable_names}")
    print(f"Sensors           : {len(loaded.sensor_ids)}")
    print(f"Timesteps         : {len(loaded.time_idx):,}")
    print(f"Train / Val / Test: {loaded.splits.split_meta['n_train']:,} / {loaded.splits.split_meta['n_val']:,} / {loaded.splits.split_meta['n_test']:,}")
    if loaded.splits.split_meta["val_windows"]:
        print("Validation windows:")
        for w in loaded.splits.split_meta["val_windows"]:
            print(f"  - {w['start']} -> {w['end']}")
    print("Test windows:")
    for w in loaded.splits.split_meta["test_windows"]:
        print(f"  - {w['start']} -> {w['end']}")

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            branch_input = batch.branch_input.to(device)
            trunk_coord = batch.trunk_coord.to(device)
            target = batch.target.to(device)
            optimiser.zero_grad(set_to_none=True)
            pred = model(branch_input, trunk_coord)
            loss = criterion(pred, target)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * target.shape[0]
            total_count += target.shape[0]

        train_loss = float(total_loss / max(1, total_count))
        val_loss = evaluate_loader(model, val_loader, device)
        scheduler.step()
        current_lr = float(scheduler.get_last_lr()[0])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(package_checkpoint_payload(cfg, loaded, model, epoch, train_loss, val_loss, history), best_path)
            marker = " *** best"
        else:
            marker = ""

        print(f"Epoch {epoch:>4}/{int(cfg['epochs'])}  train={train_loss:.6f}  val={val_loss:.6f}  lr={current_lr:.2e}{marker}", flush=True)

    torch.save(package_checkpoint_payload(cfg, loaded, model, int(cfg["epochs"]), history["train_loss"][-1], history["val_loss"][-1], history), final_path)
    print("\nSaved artifacts")
    print(f"  Best checkpoint : {best_path}")
    print(f"  Final checkpoint: {final_path}")
    return best_path, final_path


@torch.no_grad()
def evaluate(cfg: Dict[str, Any], checkpoint_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    set_seed(int(cfg["seed"]))
    device = get_device(str(cfg["device"]))
    loaded = load_data(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg, len(loaded.sensor_ids), len(loaded.variable_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_idx = np.array(checkpoint["splits"]["test_idx"], dtype=int)
    test_data_norm = loaded.data_norm[test_idx]
    mean = np.array(checkpoint["stats"]["mean"], dtype=np.float32)
    std = np.array(checkpoint["stats"]["std"], dtype=np.float32)
    sensor_ids = checkpoint["sensors"]["sensor_ids"]
    coords_norm = np.array(checkpoint["sensors"]["coords_norm"], dtype=np.float32)
    variable_names = checkpoint["variables"]
    n_sensors = len(sensor_ids)

    records: List[Dict[str, Any]] = []
    for masked_total in [int(x) for x in cfg["eval_masked_total"]]:
        extra_masks = max(0, masked_total - 1)
        for repeat in range(int(cfg["eval_repeats"])):
            rng = np.random.default_rng(int(cfg["seed"]) + 1000 * masked_total + repeat)
            preds = np.zeros_like(test_data_norm)
            for t in range(test_data_norm.shape[0]):
                for s in range(n_sensors):
                    values = test_data_norm[t].copy()
                    observed = np.ones((n_sensors,), dtype=np.float32)
                    mask_ids = [s]
                    if extra_masks > 0:
                        others = np.array([i for i in range(n_sensors) if i != s], dtype=int)
                        extra_ids = rng.choice(others, size=extra_masks, replace=False).tolist()
                        mask_ids.extend(extra_ids)
                    values[mask_ids, :] = float(cfg["mask_fill_value"])
                    observed[mask_ids] = 0.0
                    branch_parts = [values.reshape(-1)]
                    if cfg["include_observed_mask"]:
                        branch_parts.append(observed)
                    branch_input = np.concatenate(branch_parts, axis=0).astype(np.float32)
                    pred = model(torch.from_numpy(branch_input).unsqueeze(0).to(device), torch.from_numpy(coords_norm[s]).unsqueeze(0).to(device)).cpu().numpy()[0]
                    preds[t, s, :] = pred

            preds_denorm = preds * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
            actual_denorm = test_data_norm * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
            for s_idx, sensor_id in enumerate(sensor_ids):
                for c_idx, var_name in enumerate(variable_names):
                    y_true = actual_denorm[:, s_idx, c_idx]
                    y_pred = preds_denorm[:, s_idx, c_idx]
                    mse = float(np.mean((y_true - y_pred) ** 2)) if len(y_true) else float("nan")
                    rmse = float(np.sqrt(mse)) if len(y_true) else float("nan")
                    mae = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")
                    ss_res = float(np.sum((y_true - y_pred) ** 2)) if len(y_true) else float("nan")
                    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) if len(y_true) else float("nan")
                    r2 = float(1.0 - ss_res / ss_tot) if len(y_true) and ss_tot > 0 else float("nan")
                    records.append({"masked_total": masked_total, "repeat": repeat, "sensor": sensor_id, "variable": var_name, "rmse": rmse, "mae": mae, "r2": r2})

    metrics_df = pd.DataFrame(records)
    summary = {"checkpoint": str(checkpoint_path), "overall": metrics_df.groupby(["masked_total", "variable"])[["rmse", "mae", "r2"]].mean().reset_index().to_dict(orient="records")}
    out_dir = Path(cfg["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / str(cfg["eval_csv_name"])
    summary_path = out_dir / str(cfg["eval_summary_name"])
    metrics_df.to_csv(csv_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("=" * 76)
    print(f"Evaluation complete | checkpoint={checkpoint_path.name} | device={device}")
    print("=" * 76)
    print(metrics_df.groupby(["masked_total", "variable"])[["rmse", "mae", "r2"]].mean())
    print(f"\nSaved metrics : {csv_path}")
    print(f"Saved summary : {summary_path}")
    return metrics_df, summary


def resolve_checkpoint_for_eval(cfg: Dict[str, Any]) -> Path:
    target = str(cfg["checkpoint_to_evaluate"])
    out_dir = Path(cfg["output_dir"]).resolve()
    if target == "best":
        return out_dir / str(cfg["best_checkpoint_name"])
    if target == "final":
        return out_dir / str(cfg["final_checkpoint_name"])
    return Path(target).resolve()


def main(cfg: Dict[str, Any] | None = None) -> None:
    cfg = copy.deepcopy(PARAMS if cfg is None else cfg)
    mode = str(cfg["mode"]).lower()
    if mode == "train":
        train(cfg)
    elif mode == "eval":
        evaluate(cfg, resolve_checkpoint_for_eval(cfg))
    elif mode == "train_eval":
        best_path, _ = train(cfg)
        evaluate(cfg, best_path)
    else:
        raise ValueError("mode must be one of: 'train', 'eval', 'train_eval'")


if __name__ == "__main__":
    main()
