
from __future__ import annotations

"""
DeepONet v0.3.2 — targeted sparsity characterization
====================================================

Built from v0.2.2 with a new evaluation suite for probing the sparsity boundary.

What is new
-----------
- keeps the multi-output temperature + humidity model
- keeps early stopping from v0.2.2
- adds targeted evaluation protocols for masked_total = 8, 9, 10, 11, 12
- supports:
    * random masks
    * central sensors removed
    * corner / perimeter sensors removed
    * spatially spread sensors removed
    * leave-only-k-active protocols:
        - diagonal corners only (both diagonals)
        - central diagonals only (both diagonals)
        - center one only

Important interpretation
------------------------
For active-only protocols, the target sensor is always masked as well. That
means the *effective* number of masked sensors can exceed the nominal
masked_total when the target itself belongs to the active set.
"""

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

# IMPORTANT on Windows: import torch before numpy/pandas to reduce DLL conflicts.
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from torch.utils.data import DataLoader, Dataset


PARAMS: Dict[str, Any] = {
    "mode": "train_eval",  # "train", "eval", "train_eval"
    "seed": 42,
    "device": "auto",  # "auto", "cpu", "cuda", "mps"
    "experiment_name": "lab1_temp_humidity_v0.3.2",
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
            "units": "°C",
        },
        {
            "name": "humidity",
            "file": "Humidity_5min.xlsx",
            "column_prefix": "hum",
            "units": "%RH",
        },
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
    "train_extra_mask_max": 12,
    "hidden": 128,
    "latent_dim": 128,
    "depth": 4,
    "activation": "tanh",  # "tanh", "relu", "gelu"
    "batch_size": 4096,
    "epochs": 150,
    "lr": 1e-3,
    "early_stopping_enabled": True,
    "early_stopping_patience": 20,
    "early_stopping_min_delta": 1e-4,
    "weight_decay": 0.0,
    "num_workers": 0,
    "output_dir": "./deepOnet/runs/lab1data_v0.3.1",
    "best_checkpoint_name": "thermodt_best.pt",
    "final_checkpoint_name": "thermodt_final.pt",
    "checkpoint_to_evaluate": "best",  # "best", "final", or full path

    # ---- v0.3.1 evaluation protocols ----
    "eval_random_masked_total": [8, 9, 10, 11, 12],
    "eval_structured_masked_total": [8, 9, 10, 11, 12],
    "eval_random_repeats": 3,
    "eval_structured_repeats": 1,

    # Optional explicit priority orders. Set to None to compute from geometry.
    "central_priority_order": None,
    "perimeter_priority_order": None,
    "spread_priority_order": None,

    # Leave-only-k-active protocols.
    "eval_active_only_sets": [
        {
            "protocol_name": "active_corners_diag_bl_tr",
            "display_name": "Corners diagonal BL-TR only",
            "active_sensors": [4, 9],
        },
        {
            "protocol_name": "active_corners_diag_br_tl",
            "display_name": "Corners diagonal BR-TL only",
            "active_sensors": [1, 11],
        },
        {
            "protocol_name": "active_central_diag_tl_br",
            "display_name": "Central diagonal TL-BR only",
            "active_sensors": [10, 12],
        },
        {
            "protocol_name": "active_central_diag_tr_bl",
            "display_name": "Central diagonal TR-BL only",
            "active_sensors": [3, 5],
        },
        {
            "protocol_name": "active_center_one_only",
            "display_name": "Center one only",
            "active_sensors": [6],
        },
    ],

    # Output files.
    "eval_csv_name": "thermodt_eval_metrics_v032.csv",
    "eval_summary_name": "thermodt_eval_summary_v032.json",
    "eval_sensor_summary_name": "thermodt_eval_sensor_summary_v032.csv",
    "eval_protocol_summary_name": "thermodt_eval_protocol_summary_v032.csv",

    # Plotting defaults are off here because v0.3.1 can generate many protocols.
    "plot_timeseries": False,
    "plot_scatter": True,
    "plot_floormap": False,
    "plot_protocol_names": [],  # [] means "all" if plotting is enabled
    "floormap_grid_res": 300,
    "plot_repeat_aggregation": "mean",  # "mean" or "first"
    "scatter_point_size": 3,
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


def resolve_data_dir(cfg: Mapping[str, Any]) -> Path:
    primary = Path(cfg["data_root"]) / f"{cfg['lab']}_{cfg['resolution']}"
    fallback = Path(cfg["data_root"]) / f"{cfg['lab']}"
    return primary.resolve() if primary.exists() else fallback.resolve()


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

    anchors = np.linspace(
        0,
        len(candidate_starts) - 1,
        num=min(max(1, num_windows * 3), len(candidate_starts)),
        dtype=int,
    )
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
    variable_units: Dict[str, str]
    mean: np.ndarray
    std: np.ndarray
    splits: SplitIndices


@dataclass
class Batch:
    branch_input: torch.Tensor
    trunk_coord: torch.Tensor
    target: torch.Tensor


@dataclass
class EvalProtocol:
    protocol_family: str
    protocol_name: str
    display_name: str
    selection_mode: str
    nominal_masked_total: int
    repeat_count: int
    description: str
    priority_order: List[int] | None = None
    active_sensors: List[int] | None = None


@dataclass
class EvalOutputs:
    metrics_df: pd.DataFrame
    summary: Dict[str, Any]
    sensor_summary_df: pd.DataFrame
    protocol_summary_df: pd.DataFrame
    pred_store: Dict[str, np.ndarray]
    actual_denorm: np.ndarray
    time_test: pd.DatetimeIndex
    sensor_ids: List[int]
    variable_names: List[str]
    variable_units: Dict[str, str]
    room_width: float
    room_height: float
    sensor_coords: Dict[int, Tuple[float, float]]
    test_windows: List[Dict[str, Any]]
    output_dir: Path


def collate_batches(items: Sequence[Batch]) -> Batch:
    return Batch(
        branch_input=torch.stack([x.branch_input for x in items], dim=0),
        trunk_coord=torch.stack([x.trunk_coord for x in items], dim=0),
        target=torch.stack([x.target for x in items], dim=0),
    )


def split_indices(time_idx: pd.DatetimeIndex, cfg: Mapping[str, Any]) -> SplitIndices:
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


def load_data(cfg: Mapping[str, Any]) -> LoadedData:
    data_dir = resolve_data_dir(cfg)
    sensor_ids = [int(x) for x in cfg["sensor_ids"]]
    time_df = pd.read_excel(data_dir / cfg["time_file"])
    time_idx = pd.to_datetime(time_df[cfg["time_column"]])

    variable_arrays: List[np.ndarray] = []
    variable_names: List[str] = []
    variable_units: Dict[str, str] = {}
    for var_cfg in cfg["target_variables"]:
        df = pd.read_excel(data_dir / var_cfg["file"])
        cols = [f"{var_cfg['column_prefix']}{sid}" for sid in sensor_ids]
        arr = np.stack([df[col].to_numpy(dtype=float) for col in cols], axis=1) / float(cfg["scale"])
        variable_arrays.append(arr)
        variable_names.append(str(var_cfg["name"]))
        variable_units[str(var_cfg["name"])] = str(var_cfg.get("units", ""))

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
        variable_units=variable_units,
        mean=mean,
        std=std,
        splits=splits,
    )


class MultiSensorMaskedDataset(Dataset):
    def __init__(
        self,
        data_norm: np.ndarray,
        coords_norm: np.ndarray,
        mask_fill_value: float,
        include_observed_mask: bool,
        extra_mask_min: int,
        extra_mask_max: int,
        seed: int,
    ):
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
        return Batch(
            torch.from_numpy(branch_input),
            torch.from_numpy(self.coords_norm[s]),
            torch.from_numpy(target),
        )


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
    def __init__(
        self,
        branch_in_dim: int,
        coord_dim: int,
        output_dim: int,
        hidden: int,
        latent_dim: int,
        depth: int,
        activation: str,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.latent_dim = int(latent_dim)
        out_dim = self.output_dim * self.latent_dim
        self.branch = MLP(branch_in_dim, hidden, out_dim, depth, activation)
        self.trunk = MLP(coord_dim, hidden, out_dim, depth, activation)
        self.bias = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self, branch_input: torch.Tensor, trunk_coord: torch.Tensor) -> torch.Tensor:
        branch = self.branch(branch_input).view(-1, self.output_dim, self.latent_dim)
        trunk = self.trunk(trunk_coord).view(-1, self.output_dim, self.latent_dim)
        return (branch * trunk).sum(dim=-1) + self.bias


def build_model(cfg: Mapping[str, Any], n_sensors: int, n_channels: int) -> MultiOutputDeepONet:
    branch_in_dim = n_sensors * n_channels
    if cfg["include_observed_mask"]:
        branch_in_dim += n_sensors
    return MultiOutputDeepONet(
        branch_in_dim=branch_in_dim,
        coord_dim=2,
        output_dim=n_channels,
        hidden=int(cfg["hidden"]),
        latent_dim=int(cfg["latent_dim"]),
        depth=int(cfg["depth"]),
        activation=str(cfg["activation"]),
    )


def build_dataloaders(cfg: Mapping[str, Any], loaded: LoadedData, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    train_ds = MultiSensorMaskedDataset(
        loaded.data_norm[loaded.splits.train_idx],
        loaded.coords_norm,
        float(cfg["mask_fill_value"]),
        bool(cfg["include_observed_mask"]),
        int(cfg["train_extra_mask_min"]),
        int(cfg["train_extra_mask_max"]),
        int(cfg["seed"]),
    )
    val_ds = MultiSensorMaskedDataset(
        loaded.data_norm[loaded.splits.val_idx],
        loaded.coords_norm,
        float(cfg["mask_fill_value"]),
        bool(cfg["include_observed_mask"]),
        0,
        0,
        int(cfg["seed"]) + 9999,
    )
    pin_memory = str(device) != "cpu"
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=pin_memory,
        collate_fn=collate_batches,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=pin_memory,
        collate_fn=collate_batches,
    )
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


def is_improved(current: float, best: float, min_delta: float) -> bool:
    return current < (best - min_delta)


def package_checkpoint_payload(
    cfg: Mapping[str, Any],
    loaded: LoadedData,
    model: nn.Module,
    epoch: int,
    train_loss: float,
    val_loss: float,
    history: Dict[str, List[float]],
) -> Dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "history": history,
        "config": copy.deepcopy(dict(cfg)),
        "stats": {
            "mean": loaded.mean.tolist(),
            "std": loaded.std.tolist(),
            "scale": [float(cfg["scale"]) for _ in loaded.variable_names],
        },
        "splits": loaded.splits.split_meta,
        "sensors": {
            "sensor_ids": loaded.sensor_ids,
            "coords_norm": loaded.coords_norm.tolist(),
            "sensor_coords": {int(k): list(v) for k, v in cfg["sensor_coords"].items()},
            "room_width": float(cfg["room_width"]),
            "room_height": float(cfg["room_height"]),
        },
        "variables": loaded.variable_names,
        "variable_units": loaded.variable_units,
        "time": {
            "n_timesteps": int(len(loaded.time_idx)),
            "start": str(loaded.time_idx[0]) if len(loaded.time_idx) else None,
            "end": str(loaded.time_idx[-1]) if len(loaded.time_idx) else None,
        },
    }


def train(cfg: MutableMapping[str, Any]) -> Tuple[Path, Path]:
    set_seed(int(cfg["seed"]))
    device = get_device(str(cfg["device"]))
    out_dir = Path(cfg["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_data(cfg)
    train_loader, val_loader = build_dataloaders(cfg, loaded, device)
    model = build_model(cfg, len(loaded.sensor_ids), len(loaded.variable_names)).to(device)

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=int(cfg["epochs"]))
    criterion = nn.MSELoss()
    best_path = out_dir / str(cfg["best_checkpoint_name"])
    final_path = out_dir / str(cfg["final_checkpoint_name"])
    best_val = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    early_stopping_enabled = bool(cfg.get("early_stopping_enabled", False))
    early_stopping_patience = int(cfg.get("early_stopping_patience", 20))
    early_stopping_min_delta = float(cfg.get("early_stopping_min_delta", 0.0))
    history = {"train_loss": [], "val_loss": [], "lr": []}

    print("=" * 92)
    print(f"DeepONet v0.3.2 | experiment={cfg['experiment_name']} | device={device}")
    print("=" * 92)
    print(f"Variables         : {loaded.variable_names}")
    print(f"Sensors           : {len(loaded.sensor_ids)}")
    print(f"Timesteps         : {len(loaded.time_idx):,}")
    print(
        f"Train / Val / Test: {loaded.splits.split_meta['n_train']:,} / "
        f"{loaded.splits.split_meta['n_val']:,} / {loaded.splits.split_meta['n_test']:,}"
    )
    if loaded.splits.split_meta["val_windows"]:
        print("Validation windows:")
        for w in loaded.splits.split_meta["val_windows"]:
            print(f"  - {w['start']} -> {w['end']}")
    print("Test windows:")
    for w in loaded.splits.split_meta["test_windows"]:
        print(f"  - {w['start']} -> {w['end']}")
    if early_stopping_enabled:
        print(
            f"Early stopping    : enabled | patience={early_stopping_patience} | "
            f"min_delta={early_stopping_min_delta:g}"
        )
    else:
        print("Early stopping    : disabled")

    stopped_early = False
    stop_epoch = int(cfg["epochs"])

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

        if is_improved(val_loss, best_val, early_stopping_min_delta):
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                package_checkpoint_payload(cfg, loaded, model, epoch, train_loss, val_loss, history),
                best_path,
            )
            marker = " *** best"
        else:
            epochs_no_improve += 1
            marker = f" | no_improve={epochs_no_improve}/{early_stopping_patience}" if early_stopping_enabled else ""

        print(
            f"Epoch {epoch:>4}/{int(cfg['epochs'])}  train={train_loss:.6f}  "
            f"val={val_loss:.6f}  lr={current_lr:.2e}{marker}",
            flush=True,
        )

        if early_stopping_enabled and epochs_no_improve >= early_stopping_patience:
            stopped_early = True
            stop_epoch = epoch
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(best epoch={best_epoch}, best val={best_val:.6f}).",
                flush=True,
            )
            break

    final_payload = package_checkpoint_payload(
        cfg,
        loaded,
        model,
        int(stop_epoch),
        history["train_loss"][-1],
        history["val_loss"][-1],
        history,
    )
    final_payload["best_epoch"] = int(best_epoch)
    final_payload["best_val_loss"] = float(best_val)
    final_payload["stopped_early"] = bool(stopped_early)
    final_payload["stop_epoch"] = int(stop_epoch)
    final_payload["epochs_no_improve_final"] = int(epochs_no_improve)
    torch.save(final_payload, final_path)
    print("\nSaved artifacts")
    print(f"  Best checkpoint : {best_path}")
    print(f"  Final checkpoint: {final_path}")
    print(f"  Best epoch      : {best_epoch}")
    if stopped_early:
        print(f"  Training stop   : early stop at epoch {stop_epoch}")
    else:
        print(f"  Training stop   : completed {stop_epoch} epochs")
    return best_path, final_path


def _sanitize_name(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _sensor_distance_to_center(sensor_coords: Mapping[int, Sequence[float]], room_width: float, room_height: float) -> Dict[int, float]:
    center = (float(room_width) / 2.0, float(room_height) / 2.0)
    return {
        int(sid): float(math.dist(tuple(sensor_coords[int(sid)]), center))
        for sid in sensor_coords
    }


def _compute_spread_order(sensor_ids: Sequence[int], sensor_coords: Mapping[int, Sequence[float]], room_width: float, room_height: float) -> List[int]:
    distances = _sensor_distance_to_center(sensor_coords, room_width, room_height)
    remaining = [int(s) for s in sensor_ids]
    start = max(remaining, key=lambda sid: (distances[sid], -sid))
    order = [start]
    remaining.remove(start)
    while remaining:
        def score(sid: int) -> Tuple[float, float, int]:
            min_dist = min(
                math.dist(tuple(sensor_coords[int(sid)]), tuple(sensor_coords[int(chosen)]))
                for chosen in order
            )
            return (min_dist, distances[sid], -sid)

        best = max(remaining, key=score)
        order.append(best)
        remaining.remove(best)
    return order


def build_strategy_orders(cfg: Mapping[str, Any], sensor_ids: Sequence[int]) -> Dict[str, List[int]]:
    sensor_coords = {int(k): tuple(v) for k, v in cfg["sensor_coords"].items()}
    distances = _sensor_distance_to_center(sensor_coords, float(cfg["room_width"]), float(cfg["room_height"]))

    central_order_cfg = cfg.get("central_priority_order")
    if central_order_cfg:
        central_order = [int(x) for x in central_order_cfg]
    else:
        central_order = sorted([int(x) for x in sensor_ids], key=lambda sid: (distances[sid], sid))

    perimeter_order_cfg = cfg.get("perimeter_priority_order")
    if perimeter_order_cfg:
        perimeter_order = [int(x) for x in perimeter_order_cfg]
    else:
        perimeter_order = sorted([int(x) for x in sensor_ids], key=lambda sid: (-distances[sid], sid))

    spread_order_cfg = cfg.get("spread_priority_order")
    if spread_order_cfg:
        spread_order = [int(x) for x in spread_order_cfg]
    else:
        spread_order = _compute_spread_order(sensor_ids, sensor_coords, float(cfg["room_width"]), float(cfg["room_height"]))

    return {
        "central_removed": central_order,
        "perimeter_removed": perimeter_order,
        "spread_removed": spread_order,
    }


def build_eval_protocols(cfg: Mapping[str, Any], sensor_ids: Sequence[int]) -> List[EvalProtocol]:
    protocols: List[EvalProtocol] = []
    orders = build_strategy_orders(cfg, sensor_ids)

    for masked_total in [int(x) for x in cfg["eval_random_masked_total"]]:
        protocols.append(
            EvalProtocol(
                protocol_family="random",
                protocol_name=f"random_m{masked_total}",
                display_name=f"Random masks (m={masked_total})",
                selection_mode="random",
                nominal_masked_total=masked_total,
                repeat_count=int(cfg["eval_random_repeats"]),
                description=f"Random masking with total masked sensors = {masked_total}",
            )
        )

    for masked_total in [int(x) for x in cfg["eval_structured_masked_total"]]:
        for family in ("central_removed", "perimeter_removed", "spread_removed"):
            protocols.append(
                EvalProtocol(
                    protocol_family=family,
                    protocol_name=f"{family}_m{masked_total}",
                    display_name=f"{family.replace('_', ' ').title()} (m={masked_total})",
                    selection_mode="ordered_removed",
                    nominal_masked_total=masked_total,
                    repeat_count=int(cfg["eval_structured_repeats"]),
                    description=f"{family.replace('_', ' ')} with total masked sensors = {masked_total}",
                    priority_order=list(orders[family]),
                )
            )

    for entry in cfg["eval_active_only_sets"]:
        active_sensors = [int(x) for x in entry["active_sensors"]]
        protocols.append(
            EvalProtocol(
                protocol_family="active_only",
                protocol_name=str(entry["protocol_name"]),
                display_name=str(entry.get("display_name", entry["protocol_name"])),
                selection_mode="active_only",
                nominal_masked_total=len(sensor_ids) - len(active_sensors),
                repeat_count=1,
                description=f"Only sensors {active_sensors} are observed",
                active_sensors=active_sensors,
            )
        )

    return protocols


def make_mask_from_protocol(
    protocol: EvalProtocol,
    target_sid: int,
    sensor_ids: Sequence[int],
    rng: np.random.Generator,
) -> Tuple[List[int], List[int], int, int, bool]:
    n_sensors = len(sensor_ids)
    target_sid = int(target_sid)
    sensor_id_set = [int(s) for s in sensor_ids]

    if protocol.selection_mode == "random":
        extra_masks = max(0, int(protocol.nominal_masked_total) - 1)
        others = np.array([sid for sid in sensor_id_set if sid != target_sid], dtype=int)
        chosen = rng.choice(others, size=extra_masks, replace=False).tolist() if extra_masks > 0 else []
        removed = [target_sid] + [int(x) for x in chosen]
        active = [sid for sid in sensor_id_set if sid not in removed]
    elif protocol.selection_mode == "ordered_removed":
        extra_masks = max(0, int(protocol.nominal_masked_total) - 1)
        priority = [int(sid) for sid in (protocol.priority_order or []) if int(sid) != target_sid]
        removed = [target_sid] + priority[:extra_masks]
        active = [sid for sid in sensor_id_set if sid not in removed]
    elif protocol.selection_mode == "active_only":
        active_seed = [int(sid) for sid in (protocol.active_sensors or [])]
        removed = [sid for sid in sensor_id_set if sid not in active_seed]
        if target_sid not in removed:
            removed.append(target_sid)
        active = [sid for sid in active_seed if sid != target_sid]
    else:
        raise ValueError(f"Unsupported protocol selection_mode: {protocol.selection_mode}")

    removed = sorted(set(int(s) for s in removed))
    active = sorted(set(int(s) for s in active))
    effective_masked_total = len(removed)
    effective_active_total = len(active)
    target_in_active_seed = bool(protocol.active_sensors and target_sid in protocol.active_sensors)
    return removed, active, effective_masked_total, effective_active_total, target_in_active_seed


@torch.no_grad()
def run_eval_predictions(
    model: nn.Module,
    cfg: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    loaded: LoadedData,
    device: torch.device,
) -> EvalOutputs:
    test_idx = np.array(checkpoint["splits"]["test_idx"], dtype=int)
    test_data_norm = loaded.data_norm[test_idx]
    time_test = loaded.time_idx[test_idx]
    mean = np.array(checkpoint["stats"]["mean"], dtype=np.float32)
    std = np.array(checkpoint["stats"]["std"], dtype=np.float32)
    sensor_ids = [int(x) for x in checkpoint["sensors"]["sensor_ids"]]
    coords_norm = np.array(checkpoint["sensors"]["coords_norm"], dtype=np.float32)
    variable_names = list(checkpoint["variables"])
    variable_units = dict(checkpoint.get("variable_units", {k: "" for k in variable_names}))
    n_sensors = len(sensor_ids)
    n_variables = len(variable_names)
    sensor_coords = {int(k): tuple(v) for k, v in checkpoint["sensors"]["sensor_coords"].items()}
    room_width = float(checkpoint["sensors"]["room_width"])
    room_height = float(checkpoint["sensors"]["room_height"])

    actual_denorm = test_data_norm * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    pred_store: Dict[str, np.ndarray] = {}
    records: List[Dict[str, Any]] = []
    protocol_defs = build_eval_protocols(cfg, sensor_ids)

    for protocol in protocol_defs:
        pred_repeats = np.zeros((protocol.repeat_count, test_data_norm.shape[0], n_sensors, n_variables), dtype=np.float32)

        for repeat in range(protocol.repeat_count):
            rng = np.random.default_rng(int(cfg["seed"]) + 1000 * len(protocol.protocol_name) + 37 * repeat + protocol.nominal_masked_total)
            preds = np.zeros_like(test_data_norm)

            for t in range(test_data_norm.shape[0]):
                for s_idx, sensor_id in enumerate(sensor_ids):
                    values = test_data_norm[t].copy()
                    observed = np.ones((n_sensors,), dtype=np.float32)

                    removed, active, eff_masked_total, eff_active_total, target_in_active_seed = make_mask_from_protocol(
                        protocol=protocol,
                        target_sid=sensor_id,
                        sensor_ids=sensor_ids,
                        rng=rng,
                    )
                    removed_idx = [sensor_ids.index(sid) for sid in removed]
                    values[removed_idx, :] = float(cfg["mask_fill_value"])
                    observed[removed_idx] = 0.0

                    branch_parts = [values.reshape(-1)]
                    if cfg["include_observed_mask"]:
                        branch_parts.append(observed)
                    branch_input = np.concatenate(branch_parts, axis=0).astype(np.float32)
                    pred = model(
                        torch.from_numpy(branch_input).unsqueeze(0).to(device),
                        torch.from_numpy(coords_norm[s_idx]).unsqueeze(0).to(device),
                    ).cpu().numpy()[0]
                    preds[t, s_idx, :] = pred

            preds_denorm = preds * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
            pred_repeats[repeat] = preds_denorm

            for s_idx, sensor_id in enumerate(sensor_ids):
                removed, active, eff_masked_total, eff_active_total, target_in_active_seed = make_mask_from_protocol(
                    protocol=protocol,
                    target_sid=sensor_id,
                    sensor_ids=sensor_ids,
                    rng=np.random.default_rng(int(cfg["seed"]) + 999999 + repeat + s_idx),
                )
                for c_idx, var_name in enumerate(variable_names):
                    y_true = actual_denorm[:, s_idx, c_idx]
                    y_pred = preds_denorm[:, s_idx, c_idx]
                    mse = float(np.mean((y_true - y_pred) ** 2))
                    rmse = float(np.sqrt(mse))
                    mae = float(np.mean(np.abs(y_true - y_pred)))
                    ss_res = float(np.sum((y_true - y_pred) ** 2))
                    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
                    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
                    records.append(
                        {
                            "protocol_family": protocol.protocol_family,
                            "protocol_name": protocol.protocol_name,
                            "display_name": protocol.display_name,
                            "selection_mode": protocol.selection_mode,
                            "masked_total": int(protocol.nominal_masked_total),
                            "effective_masked_total": int(eff_masked_total),
                            "active_total_nominal": int(n_sensors - protocol.nominal_masked_total),
                            "active_total_effective": int(eff_active_total),
                            "target_in_active_seed": bool(target_in_active_seed),
                            "repeat": repeat,
                            "sensor": int(sensor_id),
                            "variable": var_name,
                            "rmse": rmse,
                            "mae": mae,
                            "r2": r2,
                        }
                    )

        pred_store[protocol.protocol_name] = pred_repeats[0] if str(cfg.get("plot_repeat_aggregation", "mean")).lower() == "first" else pred_repeats.mean(axis=0)

    metrics_df = pd.DataFrame(records)
    protocol_summary_df = (
        metrics_df.groupby(["protocol_family", "protocol_name", "display_name", "masked_total", "variable"])[["rmse", "mae", "r2"]]
        .mean()
        .reset_index()
        .sort_values(["masked_total", "protocol_family", "protocol_name", "variable"])
        .reset_index(drop=True)
    )
    sensor_summary_df = (
        metrics_df.groupby(["protocol_family", "protocol_name", "display_name", "masked_total", "sensor", "variable"])[["rmse", "mae", "r2"]]
        .mean()
        .reset_index()
    )
    summary = {
        "checkpoint": str(cfg.get("checkpoint_path_used", "")),
        "protocols": [
            {
                "protocol_family": p.protocol_family,
                "protocol_name": p.protocol_name,
                "display_name": p.display_name,
                "selection_mode": p.selection_mode,
                "masked_total": p.nominal_masked_total,
                "repeat_count": p.repeat_count,
                "description": p.description,
                "priority_order": p.priority_order,
                "active_sensors": p.active_sensors,
            }
            for p in protocol_defs
        ],
        "overall": protocol_summary_df.to_dict(orient="records"),
    }

    out_dir = Path(cfg["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    return EvalOutputs(
        metrics_df=metrics_df,
        summary=summary,
        sensor_summary_df=sensor_summary_df,
        protocol_summary_df=protocol_summary_df,
        pred_store=pred_store,
        actual_denorm=actual_denorm,
        time_test=time_test,
        sensor_ids=sensor_ids,
        variable_names=variable_names,
        variable_units=variable_units,
        room_width=room_width,
        room_height=room_height,
        sensor_coords=sensor_coords,
        test_windows=list(checkpoint["splits"]["test_windows"]),
        output_dir=out_dir,
    )


def hour_to_period(h: int) -> str:
    if h < 6:
        return "Night"
    if h < 8:
        return "Early AM"
    if h < 12:
        return "Morning"
    if h < 14:
        return "Noon"
    if h < 17:
        return "Afternoon"
    if h < 21:
        return "Evening"
    return "Night"


def make_tick_positions(time_test: pd.DatetimeIndex, test_windows: Sequence[Mapping[str, Any]]) -> Tuple[List[int], List[str]]:
    tick_pos: List[int] = []
    tick_labels: List[str] = []
    for w in test_windows:
        start = pd.Timestamp(w["start"])
        matches = np.where(time_test == start)[0]
        if len(matches) == 0:
            continue
        pos = int(matches[0])
        tick_pos.append(pos)
        date_short = start.strftime("%b %d")
        tick_labels.append(f"{pos}\n{date_short}\n{hour_to_period(int(start.hour))}")
    tick_pos.append(len(time_test) - 1)
    tick_labels.append(str(len(time_test) - 1))
    return tick_pos, tick_labels


def _should_plot_protocol(cfg: Mapping[str, Any], protocol_name: str) -> bool:
    selected = list(cfg.get("plot_protocol_names", []))
    return (not selected) or (protocol_name in selected)


def save_timeseries_plots(outputs: EvalOutputs, protocol_name: str, display_name: str, masked_total: int) -> None:
    n_sensors = len(outputs.sensor_ids)
    n_cols = 3
    n_rows = math.ceil(n_sensors / n_cols)
    x_axis = np.arange(len(outputs.time_test))
    tick_pos, tick_labels = make_tick_positions(outputs.time_test, outputs.test_windows)
    pred = outputs.pred_store[protocol_name]

    for c_idx, variable in enumerate(outputs.variable_names):
        units = outputs.variable_units.get(variable, "")
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.8), sharex=True)
        axes = np.atleast_1d(axes).flatten()

        sensor_metrics = outputs.sensor_summary_df[
            (outputs.sensor_summary_df["protocol_name"] == protocol_name) &
            (outputs.sensor_summary_df["variable"] == variable)
        ].set_index("sensor")

        for s_idx, sensor_id in enumerate(outputs.sensor_ids):
            ax = axes[s_idx]
            actual = outputs.actual_denorm[:, s_idx, c_idx]
            predicted = pred[:, s_idx, c_idx]

            ax.plot(x_axis, actual, lw=0.85, alpha=0.85, label="Actual")
            ax.plot(x_axis, predicted, lw=0.85, alpha=0.90, label="Predicted")
            metrics_row = sensor_metrics.loc[sensor_id]
            ax.set_title(
                f"Sensor {sensor_id} | RMSE={metrics_row['rmse']:.3f} {units} | R²={metrics_row['r2']:.3f}",
                fontsize=9,
            )
            ax.set_ylabel(f"{variable.capitalize()} ({units})" if units else variable.capitalize(), fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            for wp in tick_pos[:-1]:
                ax.axvline(wp, color="gray", lw=0.5, ls="--", alpha=0.5, zorder=1)
            if s_idx == 0:
                ax.legend(fontsize=7)

        for ax in axes:
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, fontsize=6.5)
            ax.set_xlabel("Sample index / Date-time", fontsize=7)
        for ax in axes[n_sensors:]:
            ax.set_visible(False)

        fig.suptitle(
            f"DeepONet v0.3.2 — {variable.capitalize()} | {display_name} | masked_total={masked_total}",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        out_path = outputs.output_dir / f"thermodt_eval_timeseries_{_sanitize_name(protocol_name)}_{_sanitize_name(variable)}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved : {out_path}")


def save_scatter_plots(outputs: EvalOutputs, protocol_name: str, display_name: str, masked_total: int, point_size: float) -> None:
    n_sensors = len(outputs.sensor_ids)
    n_cols = 3
    n_rows = math.ceil(n_sensors / n_cols)
    pred = outputs.pred_store[protocol_name]

    for c_idx, variable in enumerate(outputs.variable_names):
        units = outputs.variable_units.get(variable, "")
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = np.atleast_1d(axes).flatten()

        sensor_metrics = outputs.sensor_summary_df[
            (outputs.sensor_summary_df["protocol_name"] == protocol_name) &
            (outputs.sensor_summary_df["variable"] == variable)
        ].set_index("sensor")

        for s_idx, sensor_id in enumerate(outputs.sensor_ids):
            ax = axes[s_idx]
            actual = outputs.actual_denorm[:, s_idx, c_idx]
            predicted = pred[:, s_idx, c_idx]

            ax.scatter(actual, predicted, s=point_size, alpha=0.3, rasterized=True)
            lims = [min(actual.min(), predicted.min()) - 0.5, max(actual.max(), predicted.max()) + 0.5]
            ax.plot(lims, lims, "r--", lw=1.2, label="Ideal")
            if len(actual) >= 2 and np.std(actual) > 0:
                coeffs = np.polyfit(actual, predicted, 1)
                x_line = np.linspace(lims[0], lims[1], 200)
                ax.plot(x_line, np.polyval(coeffs, x_line), "k-", lw=1.2, label=f"Fit: {coeffs[0]:.2f}x+{coeffs[1]:.2f}")
            metrics_row = sensor_metrics.loc[sensor_id]
            ax.set_title(
                f"Sensor {sensor_id} | RMSE={metrics_row['rmse']:.3f} {units} | R²={metrics_row['r2']:.3f}",
                fontsize=9,
            )
            ax.set_xlabel(f"Actual {variable.capitalize()} ({units})" if units else f"Actual {variable.capitalize()}", fontsize=8)
            ax.set_ylabel(f"Predicted {variable.capitalize()} ({units})" if units else f"Predicted {variable.capitalize()}", fontsize=8)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

        for ax in axes[n_sensors:]:
            ax.set_visible(False)

        fig.suptitle(
            f"DeepONet v0.3.2 — {variable.capitalize()} | {display_name} | masked_total={masked_total}",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        out_path = outputs.output_dir / f"thermodt_eval_scatter_{_sanitize_name(protocol_name)}_{_sanitize_name(variable)}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved : {out_path}")


def save_floormap_plots(outputs: EvalOutputs, protocol_name: str, display_name: str, masked_total: int, grid_res: int) -> None:
    gx = np.linspace(0, outputs.room_width, grid_res)
    gy = np.linspace(0, outputs.room_height, grid_res)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
    coords_m = np.array([outputs.sensor_coords[i] for i in outputs.sensor_ids], dtype=float)

    for variable in outputs.variable_names:
        var_metrics = outputs.sensor_summary_df[
            (outputs.sensor_summary_df["protocol_name"] == protocol_name) &
            (outputs.sensor_summary_df["variable"] == variable)
        ].sort_values("sensor")
        if len(var_metrics) != len(outputs.sensor_ids):
            continue

        rmse_vals = var_metrics["rmse"].to_numpy(dtype=float)
        mae_vals = var_metrics["mae"].to_numpy(dtype=float)
        r2_vals = var_metrics["r2"].to_numpy(dtype=float)
        panels = [
            ("RMSE", rmse_vals, "RdYlGn_r"),
            ("MAE", mae_vals, "RdYlGn_r"),
            ("R²", r2_vals, "RdYlGn"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"DeepONet v0.3.2 — {variable.capitalize()} | {display_name} | masked_total={masked_total}",
            fontsize=12,
            fontweight="bold",
        )

        for ax, (label, values, cmap) in zip(axes, panels):
            try:
                rbf = RBFInterpolator(coords_m, values, kernel="thin_plate_spline", smoothing=0)
                Z = rbf(grid_pts).reshape(GX.shape)
                im = ax.imshow(
                    Z,
                    origin="lower",
                    extent=[0, outputs.room_width, 0, outputs.room_height],
                    cmap=cmap,
                    aspect="equal",
                    interpolation="bilinear",
                )
            except Exception:
                im = None
                ax.set_facecolor("white")

            ax.scatter(
                coords_m[:, 0],
                coords_m[:, 1],
                c=values,
                cmap=cmap,
                s=160,
                edgecolors="black",
                linewidths=0.9,
                zorder=5,
            )
            for sid, (x, y), v in zip(outputs.sensor_ids, coords_m, values):
                ax.annotate(
                    f"{sid}\n{v:.3f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=7,
                    fontweight="bold",
                    color="white",
                    zorder=6,
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")],
                )
            if im is not None:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(label, fontsize=9)

            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_edgecolor("black")
            ax.set_title(label, fontsize=11)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_xlim(0, outputs.room_width)
            ax.set_ylim(0, outputs.room_height)
            ax.set_xticks(np.arange(0, outputs.room_width + 1, 2))
            ax.set_yticks(np.arange(0, outputs.room_height + 1, 2))

        plt.tight_layout()
        out_path = outputs.output_dir / f"thermodt_eval_floormap_{_sanitize_name(protocol_name)}_{_sanitize_name(variable)}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved : {out_path}")


@torch.no_grad()
def evaluate(cfg: MutableMapping[str, Any], checkpoint_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    set_seed(int(cfg["seed"]))
    device = get_device(str(cfg["device"]))
    loaded = load_data(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg, len(loaded.sensor_ids), len(loaded.variable_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    cfg = dict(cfg)
    cfg["checkpoint_path_used"] = str(checkpoint_path)
    outputs = run_eval_predictions(model, cfg, checkpoint, loaded, device)

    csv_path = outputs.output_dir / str(cfg["eval_csv_name"])
    summary_path = outputs.output_dir / str(cfg["eval_summary_name"])
    sensor_summary_path = outputs.output_dir / str(cfg["eval_sensor_summary_name"])
    protocol_summary_path = outputs.output_dir / str(cfg["eval_protocol_summary_name"])

    outputs.metrics_df.to_csv(csv_path, index=False)
    outputs.sensor_summary_df.to_csv(sensor_summary_path, index=False)
    outputs.protocol_summary_df.to_csv(protocol_summary_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(outputs.summary, f, indent=2)

    print("=" * 92)
    print(f"Evaluation complete | checkpoint={checkpoint_path.name} | device={device}")
    print("=" * 92)
    print(outputs.protocol_summary_df.set_index(["protocol_name", "masked_total", "variable"])[["rmse", "mae", "r2"]])

    print(f"\nSaved raw metrics      : {csv_path}")
    print(f"Saved sensor summary   : {sensor_summary_path}")
    print(f"Saved protocol summary : {protocol_summary_path}")
    print(f"Saved summary JSON     : {summary_path}")

    protocol_lookup = outputs.protocol_summary_df[["protocol_name", "display_name", "masked_total"]].drop_duplicates()
    for _, row in protocol_lookup.iterrows():
        protocol_name = str(row["protocol_name"])
        if not _should_plot_protocol(cfg, protocol_name):
            continue
        if bool(cfg.get("plot_timeseries", False)):
            save_timeseries_plots(outputs, protocol_name, str(row["display_name"]), int(row["masked_total"]))
        if bool(cfg.get("plot_scatter", False)):
            save_scatter_plots(outputs, protocol_name, str(row["display_name"]), int(row["masked_total"]), float(cfg.get("scatter_point_size", 3)))
        if bool(cfg.get("plot_floormap", False)):
            save_floormap_plots(outputs, protocol_name, str(row["display_name"]), int(row["masked_total"]), int(cfg.get("floormap_grid_res", 300)))

    return outputs.metrics_df, outputs.summary


def resolve_checkpoint_for_eval(cfg: Mapping[str, Any]) -> Path:
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
