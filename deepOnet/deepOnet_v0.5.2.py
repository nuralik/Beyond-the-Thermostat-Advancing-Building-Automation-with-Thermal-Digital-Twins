from __future__ import annotations

"""
DeepONet v0.5.2 — sensor importance and minimal active-set selection
====================================================================

Built from v0.3.2 while preserving the useful core pieces:
- multi-output temperature + humidity reconstruction
- early stopping
- blocked validation / test windows
- explicit observed-mask input
- optional plotting (time-series / scatter / floor-map)
- optional targeted sparsity protocols from v0.3.2

What is new
-----------
- sensor-importance analysis via leave-one-sensor-out active sets
- full-reference protocol (all sensors observed except the target itself)
- greedy backward elimination over active sensor sets
- exhaustive active-set search for small k (default k = 1, 2, 3)
- composite scoring / ranking for subset selection
- dedicated CSV / JSON outputs for importance and subset search

Interpretation reminder
-----------------------
The target sensor is always masked during reconstruction. Therefore,
for any active-set protocol, the *effective* number of active sensors can be one
smaller than the nominal active-set size whenever the target belongs to the
active seed.
"""

import copy
import itertools
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
    "experiment_name": "lab1_temp_humidity_v0.5.2_wide",
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
    "hidden": 256,
    "latent_dim": 256,
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
    "output_dir": "./deepOnet/runs/lab1data_v0.5.2",
    "best_checkpoint_name": "thermodt_best.pt",
    "final_checkpoint_name": "thermodt_final.pt",
    "checkpoint_to_evaluate": "best",  # "best", "final", or full path

    # ---- Step 5 compact comparison suite ----
    "enable_comparison_protocols": True,
    "comparison_random_masked_total": [6, 9, 10, 11],
    "comparison_random_repeats": 3,
    "comparison_structured_masked_total": [11],
    "comparison_structured_repeats": 1,
    "comparison_structured_families": ["central_removed", "perimeter_removed", "spread_removed"],
    "comparison_active_only_sets": [
        {
            "protocol_name": "cmp_active_sensor5_only",
            "display_name": "Comparison active set [5]",
            "active_sensors": [5],
        },
        {
            "protocol_name": "cmp_active_sensor5_8",
            "display_name": "Comparison active set [5, 8]",
            "active_sensors": [5, 8],
        },
        {
            "protocol_name": "cmp_active_sensor2_5_8",
            "display_name": "Comparison active set [2, 5, 8]",
            "active_sensors": [2, 5, 8],
        },
    ],

    # ---- preserved optional targeted sparsity evaluation from v0.3.2 ----
    "enable_targeted_sparsity_protocols": False,
    "eval_random_masked_total": [8, 9, 10, 11, 12],
    "eval_structured_masked_total": [8, 9, 10, 11, 12],
    "eval_random_repeats": 3,
    "eval_structured_repeats": 1,
    "central_priority_order": None,
    "perimeter_priority_order": None,
    "spread_priority_order": None,
    "enable_legacy_active_only_protocols": False,
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

    # ---- new v0.4.1 evaluation controls ----
    "enable_sensor_importance_protocols": False,
    "enable_backward_elimination": False,
    "backward_min_active_count": 1,
    "enable_exhaustive_small_k": False,
    "exhaustive_active_k_values": [1, 2, 3],
    "exhaustive_top_n_per_k": 10,
    "selection_score_metric": "weighted_r2",  # currently supported: weighted_r2
    "selection_variable_weights": {
        "temperature": 1.0,
        "humidity": 1.0,
    },
    "selection_thresholds": {
        "temperature": None,
        "humidity": None,
    },
    "selection_score_direction": "higher_is_better",

    # Output files.
    "eval_csv_name": "thermodt_eval_metrics_v052.csv",
    "eval_summary_name": "thermodt_eval_summary_v052.json",
    "eval_sensor_summary_name": "thermodt_eval_sensor_summary_v052.csv",
    "eval_protocol_summary_name": "thermodt_eval_protocol_summary_v052.csv",
    "sensor_importance_summary_name": "thermodt_sensor_importance_v052.csv",
    "backward_elimination_path_name": "thermodt_backward_elimination_path_v052.csv",
    "backward_elimination_candidates_name": "thermodt_backward_elimination_candidates_v052.csv",
    "exhaustive_subset_search_name": "thermodt_exhaustive_subsets_v052.csv",

    # Plotting defaults remain available but off by default.
    "plot_timeseries": False,
    "plot_scatter": False,
    "plot_floormap": False,
    "plot_protocol_names": [],  # [] means "all" if plotting is enabled
    "floormap_grid_res": 300,
    "plot_repeat_aggregation": "mean",  # "mean" or "first"
    "scatter_point_size": 3,
}


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


@dataclass
class ProtocolEvalResult:
    protocol: EvalProtocol
    metrics_df: pd.DataFrame
    mean_preds_denorm: np.ndarray | None


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Dataset / model / training (preserved from v0.3.2)
# -----------------------------------------------------------------------------
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



def collate_batches(items: Sequence[Batch]) -> Batch:
    return Batch(
        branch_input=torch.stack([x.branch_input for x in items], dim=0),
        trunk_coord=torch.stack([x.trunk_coord for x in items], dim=0),
        target=torch.stack([x.target for x in items], dim=0),
    )



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
    print(f"DeepONet v0.5.2 | experiment={cfg['experiment_name']} | device={device}")
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


# -----------------------------------------------------------------------------
# Protocol helpers / scoring
# -----------------------------------------------------------------------------
def _sanitize_name(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")



def _sensor_distance_to_center(sensor_coords: Mapping[int, Sequence[float]], room_width: float, room_height: float) -> Dict[int, float]:
    center = (float(room_width) / 2.0, float(room_height) / 2.0)
    return {int(sid): float(math.dist(tuple(sensor_coords[int(sid)]), center)) for sid in sensor_coords}



def _compute_spread_order(sensor_ids: Sequence[int], sensor_coords: Mapping[int, Sequence[float]], room_width: float, room_height: float) -> List[int]:
    distances = _sensor_distance_to_center(sensor_coords, room_width, room_height)
    remaining = [int(s) for s in sensor_ids]
    start = max(remaining, key=lambda sid: (distances[sid], -sid))
    order = [start]
    remaining.remove(start)
    while remaining:
        def score(sid: int) -> Tuple[float, float, int]:
            min_dist = min(math.dist(tuple(sensor_coords[int(sid)]), tuple(sensor_coords[int(chosen)])) for chosen in order)
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

    # Compact Step 5 comparison suite.
    if bool(cfg.get("enable_comparison_protocols", False)):
        orders = build_strategy_orders(cfg, sensor_ids)
        for masked_total in [int(x) for x in cfg.get("comparison_random_masked_total", [])]:
            protocols.append(
                EvalProtocol(
                    protocol_family="comparison_random",
                    protocol_name=f"cmp_random_m{masked_total}",
                    display_name=f"Comparison random masks (m={masked_total})",
                    selection_mode="random",
                    nominal_masked_total=masked_total,
                    repeat_count=int(cfg.get("comparison_random_repeats", 1)),
                    description=f"Step 5 comparison: random masking with total masked sensors = {masked_total}.",
                )
            )

        comparison_families = [str(x) for x in cfg.get("comparison_structured_families", ["central_removed", "perimeter_removed", "spread_removed"])]
        for masked_total in [int(x) for x in cfg.get("comparison_structured_masked_total", [])]:
            for family in comparison_families:
                if family not in orders:
                    raise ValueError(f"Unsupported comparison structured family: {family}")
                protocols.append(
                    EvalProtocol(
                        protocol_family=f"comparison_{family}",
                        protocol_name=f"cmp_{family}_m{masked_total}",
                        display_name=f"Comparison {family.replace('_', ' ')} (m={masked_total})",
                        selection_mode="ordered_removed",
                        nominal_masked_total=masked_total,
                        repeat_count=int(cfg.get("comparison_structured_repeats", 1)),
                        description=f"Step 5 comparison: {family.replace('_', ' ')} with total masked sensors = {masked_total}.",
                        priority_order=list(orders[family]),
                    )
                )

        for entry in cfg.get("comparison_active_only_sets", []):
            active_sensors = [int(x) for x in entry["active_sensors"]]
            protocols.append(
                EvalProtocol(
                    protocol_family="comparison_active_only",
                    protocol_name=str(entry["protocol_name"]),
                    display_name=str(entry.get("display_name", entry["protocol_name"])),
                    selection_mode="active_only",
                    nominal_masked_total=len(sensor_ids) - len(active_sensors),
                    repeat_count=1,
                    description=f"Step 5 comparison: only sensors {active_sensors} are observed.",
                    active_sensors=active_sensors,
                )
            )

    # v0.4.1 sensor-importance protocols.
    if bool(cfg.get("enable_sensor_importance_protocols", True)):
        all_sensors = [int(s) for s in sensor_ids]
        protocols.append(
            EvalProtocol(
                protocol_family="sensor_importance",
                protocol_name="full_reference_all_active",
                display_name="Full reference (all sensors active)",
                selection_mode="active_only",
                nominal_masked_total=0,
                repeat_count=1,
                description="All sensors are observed except the target itself.",
                active_sensors=all_sensors,
            )
        )
        for sid in sensor_ids:
            active = [int(s) for s in sensor_ids if int(s) != int(sid)]
            protocols.append(
                EvalProtocol(
                    protocol_family="sensor_importance",
                    protocol_name=f"drop_sensor_{int(sid)}",
                    display_name=f"Drop sensor {int(sid)}",
                    selection_mode="active_only",
                    nominal_masked_total=1,
                    repeat_count=1,
                    description=f"All sensors active except sensor {int(sid)}.",
                    active_sensors=active,
                )
            )

    # Preserve special active-only sets from v0.3.2 behind an explicit switch.
    if bool(cfg.get("enable_legacy_active_only_protocols", False)):
        for entry in cfg.get("eval_active_only_sets", []):
            active_sensors = [int(x) for x in entry["active_sensors"]]
            protocols.append(
                EvalProtocol(
                    protocol_family="active_only",
                    protocol_name=str(entry["protocol_name"]),
                    display_name=str(entry.get("display_name", entry["protocol_name"])),
                    selection_mode="active_only",
                    nominal_masked_total=len(sensor_ids) - len(active_sensors),
                    repeat_count=1,
                    description=f"Only sensors {active_sensors} are observed.",
                    active_sensors=active_sensors,
                )
            )

    # Preserve old targeted sparsity protocols as optional.
    if bool(cfg.get("enable_targeted_sparsity_protocols", False)):
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
                    description=f"Random masking with total masked sensors = {masked_total}.",
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
                        description=f"{family.replace('_', ' ')} with total masked sensors = {masked_total}.",
                        priority_order=list(orders[family]),
                    )
                )

    return protocols


def get_variable_weights(cfg: Mapping[str, Any], variable_names: Sequence[str]) -> Dict[str, float]:
    raw = dict(cfg.get("selection_variable_weights", {}))
    return {str(v): float(raw.get(str(v), 1.0)) for v in variable_names}



def compute_score_from_metrics(metrics_df: pd.DataFrame, variable_names: Sequence[str], cfg: Mapping[str, Any]) -> Tuple[float, Dict[str, float], bool]:
    summary = metrics_df.groupby("variable")[["rmse", "mae", "r2"]].mean()
    weights = get_variable_weights(cfg, variable_names)
    metric_name = str(cfg.get("selection_score_metric", "weighted_r2"))
    if metric_name != "weighted_r2":
        raise ValueError(f"Unsupported selection_score_metric: {metric_name}")
    total_weight = sum(weights.get(v, 1.0) for v in variable_names)
    score = sum(weights.get(v, 1.0) * float(summary.loc[v, "r2"]) for v in variable_names) / max(total_weight, 1e-12)
    thresholds_cfg = dict(cfg.get("selection_thresholds", {}))
    meets_thresholds = True
    for v in variable_names:
        threshold = thresholds_cfg.get(v)
        if threshold is None:
            continue
        if float(summary.loc[v, "r2"]) < float(threshold):
            meets_thresholds = False
            break
    per_var_r2 = {v: float(summary.loc[v, "r2"]) for v in variable_names}
    return float(score), per_var_r2, bool(meets_thresholds)



def make_mask_from_protocol(
    protocol: EvalProtocol,
    target_sid: int,
    sensor_ids: Sequence[int],
    rng: np.random.Generator,
) -> Tuple[List[int], List[int], int, int, bool]:
    sensor_id_set = [int(s) for s in sensor_ids]
    target_sid = int(target_sid)

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


# -----------------------------------------------------------------------------
# Protocol evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_protocol(
    model: nn.Module,
    protocol: EvalProtocol,
    cfg: Mapping[str, Any],
    test_data_norm: np.ndarray,
    coords_norm: np.ndarray,
    sensor_ids: Sequence[int],
    device: torch.device,
) -> np.ndarray:
    """Returns predictions in normalized space with shape (repeat, T, S, V)."""
    n_time, n_sensors, n_variables = test_data_norm.shape
    preds_repeats = np.zeros((protocol.repeat_count, n_time, n_sensors, n_variables), dtype=np.float32)
    sensor_index = {int(sid): idx for idx, sid in enumerate(sensor_ids)}

    for repeat in range(protocol.repeat_count):
        rng = np.random.default_rng(int(cfg["seed"]) + 1000 * len(protocol.protocol_name) + 37 * repeat + protocol.nominal_masked_total)

        if protocol.selection_mode in {"active_only", "ordered_removed"}:
            for s_idx, sensor_id in enumerate(sensor_ids):
                removed, _, _, _, _ = make_mask_from_protocol(protocol, int(sensor_id), sensor_ids, rng)
                removed_idx = [sensor_index[sid] for sid in removed]
                values = test_data_norm.copy()
                observed = np.ones((n_time, n_sensors), dtype=np.float32)
                values[:, removed_idx, :] = float(cfg["mask_fill_value"])
                observed[:, removed_idx] = 0.0
                branch_parts = [values.reshape(n_time, -1)]
                if cfg["include_observed_mask"]:
                    branch_parts.append(observed)
                branch_input = np.concatenate(branch_parts, axis=1).astype(np.float32)
                coord_batch = np.repeat(coords_norm[s_idx:s_idx + 1], n_time, axis=0).astype(np.float32)
                pred = model(
                    torch.from_numpy(branch_input).to(device),
                    torch.from_numpy(coord_batch).to(device),
                ).cpu().numpy()
                preds_repeats[repeat, :, s_idx, :] = pred
        elif protocol.selection_mode == "random":
            for t in range(n_time):
                for s_idx, sensor_id in enumerate(sensor_ids):
                    removed, _, _, _, _ = make_mask_from_protocol(protocol, int(sensor_id), sensor_ids, rng)
                    removed_idx = [sensor_index[sid] for sid in removed]
                    values = test_data_norm[t].copy()
                    observed = np.ones((n_sensors,), dtype=np.float32)
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
                    preds_repeats[repeat, t, s_idx, :] = pred
        else:
            raise ValueError(f"Unsupported protocol selection_mode: {protocol.selection_mode}")

    return preds_repeats


@torch.no_grad()
def evaluate_single_protocol(
    model: nn.Module,
    protocol: EvalProtocol,
    cfg: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    loaded: LoadedData,
    device: torch.device,
    keep_predictions: bool,
) -> ProtocolEvalResult:
    test_idx = np.array(checkpoint["splits"]["test_idx"], dtype=int)
    test_data_norm = loaded.data_norm[test_idx]
    mean = np.array(checkpoint["stats"]["mean"], dtype=np.float32)
    std = np.array(checkpoint["stats"]["std"], dtype=np.float32)
    sensor_ids = [int(x) for x in checkpoint["sensors"]["sensor_ids"]]
    coords_norm = np.array(checkpoint["sensors"]["coords_norm"], dtype=np.float32)
    variable_names = list(checkpoint["variables"])
    actual_denorm = test_data_norm * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)

    pred_repeats = predict_protocol(model, protocol, cfg, test_data_norm, coords_norm, sensor_ids, device)
    pred_repeats_denorm = pred_repeats * std.reshape(1, 1, 1, -1) + mean.reshape(1, 1, 1, -1)

    records: List[Dict[str, Any]] = []
    for repeat in range(protocol.repeat_count):
        rng_meta = np.random.default_rng(int(cfg["seed"]) + 999999 + repeat + len(protocol.protocol_name))
        preds_denorm = pred_repeats_denorm[repeat]
        for s_idx, sensor_id in enumerate(sensor_ids):
            removed, active, eff_masked_total, eff_active_total, target_in_active_seed = make_mask_from_protocol(
                protocol=protocol,
                target_sid=int(sensor_id),
                sensor_ids=sensor_ids,
                rng=rng_meta,
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
                        "active_total_nominal": int(len(sensor_ids) - protocol.nominal_masked_total),
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

    mean_preds_denorm = None
    if keep_predictions:
        mode = str(cfg.get("plot_repeat_aggregation", "mean")).lower()
        mean_preds_denorm = pred_repeats_denorm[0] if mode == "first" else pred_repeats_denorm.mean(axis=0)

    return ProtocolEvalResult(protocol=protocol, metrics_df=pd.DataFrame(records), mean_preds_denorm=mean_preds_denorm)


# -----------------------------------------------------------------------------
# Sensor importance / subset search
# -----------------------------------------------------------------------------
def protocol_from_active_set(protocol_name: str, display_name: str, sensor_ids: Sequence[int], active_sensors: Sequence[int]) -> EvalProtocol:
    active_sensors = sorted({int(s) for s in active_sensors})
    nominal_masked_total = len(sensor_ids) - len(active_sensors)
    return EvalProtocol(
        protocol_family="active_set_search",
        protocol_name=protocol_name,
        display_name=display_name,
        selection_mode="active_only",
        nominal_masked_total=nominal_masked_total,
        repeat_count=1,
        description=f"Fixed active set: {active_sensors}",
        active_sensors=active_sensors,
    )


@torch.no_grad()
def evaluate_fixed_active_set_summary(
    model: nn.Module,
    cfg: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    loaded: LoadedData,
    device: torch.device,
    active_sensors: Sequence[int],
    protocol_name: str,
    display_name: str,
    keep_predictions: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, Dict[str, float], bool, np.ndarray | None]:
    protocol = protocol_from_active_set(protocol_name, display_name, loaded.sensor_ids, active_sensors)
    result = evaluate_single_protocol(model, protocol, cfg, checkpoint, loaded, device, keep_predictions=keep_predictions)
    sensor_summary_df = (
        result.metrics_df.groupby(["protocol_family", "protocol_name", "display_name", "masked_total", "sensor", "variable"])[["rmse", "mae", "r2"]]
        .mean()
        .reset_index()
    )
    score, per_var_r2, meets_thresholds = compute_score_from_metrics(result.metrics_df, loaded.variable_names, cfg)
    return result.metrics_df, sensor_summary_df, score, per_var_r2, meets_thresholds, result.mean_preds_denorm


@torch.no_grad()
def run_backward_elimination(
    model: nn.Module,
    cfg: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    loaded: LoadedData,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[EvalProtocol]]:
    if not bool(cfg.get("enable_backward_elimination", True)):
        return pd.DataFrame(), pd.DataFrame(), []

    active_set = sorted(int(s) for s in loaded.sensor_ids)
    min_active = max(1, int(cfg.get("backward_min_active_count", 1)))
    path_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    chosen_protocols: List[EvalProtocol] = []
    step = 0

    while len(active_set) > min_active:
        best_candidate: Dict[str, Any] | None = None
        for remove_sid in active_set:
            candidate_active = [sid for sid in active_set if sid != remove_sid]
            metrics_df, _, score, per_var_r2, meets_thresholds, _ = evaluate_fixed_active_set_summary(
                model=model,
                cfg=cfg,
                checkpoint=checkpoint,
                loaded=loaded,
                device=device,
                active_sensors=candidate_active,
                protocol_name=f"backward_candidate_remove_{remove_sid}_to_k{len(candidate_active)}",
                display_name=f"Backward candidate remove {remove_sid} to k={len(candidate_active)}",
                keep_predictions=False,
            )
            row = {
                "step": step + 1,
                "removed_sensor_candidate": int(remove_sid),
                "active_count_after": int(len(candidate_active)),
                "active_sensors_after": json.dumps(candidate_active),
                "score": float(score),
                "meets_thresholds": bool(meets_thresholds),
            }
            for v in loaded.variable_names:
                row[f"r2_{v}"] = float(per_var_r2[v])
            candidate_rows.append(row)
            if (best_candidate is None) or (row["score"] > best_candidate["score"]):
                best_candidate = row

        assert best_candidate is not None
        step += 1
        removed_sensor = int(best_candidate["removed_sensor_candidate"])
        active_set = json.loads(best_candidate["active_sensors_after"])
        path_row = {
            "step": step,
            "removed_sensor": removed_sensor,
            "active_count": int(len(active_set)),
            "active_sensors": json.dumps(active_set),
            "score": float(best_candidate["score"]),
            "meets_thresholds": bool(best_candidate["meets_thresholds"]),
        }
        for v in loaded.variable_names:
            path_row[f"r2_{v}"] = float(best_candidate[f"r2_{v}"])
        path_rows.append(path_row)
        chosen_protocols.append(
            protocol_from_active_set(
                protocol_name=f"backward_best_k{len(active_set)}",
                display_name=f"Backward elimination best set (k={len(active_set)})",
                sensor_ids=loaded.sensor_ids,
                active_sensors=active_set,
            )
        )

    return pd.DataFrame(path_rows), pd.DataFrame(candidate_rows), chosen_protocols


@torch.no_grad()
def run_exhaustive_active_search(
    model: nn.Module,
    cfg: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    loaded: LoadedData,
    device: torch.device,
) -> Tuple[pd.DataFrame, List[EvalProtocol]]:
    if not bool(cfg.get("enable_exhaustive_small_k", True)):
        return pd.DataFrame(), []

    rows: List[Dict[str, Any]] = []
    top_protocols: List[EvalProtocol] = []
    k_values = [int(k) for k in cfg.get("exhaustive_active_k_values", [])]
    top_n = int(cfg.get("exhaustive_top_n_per_k", 10))

    for k in k_values:
        for subset in itertools.combinations(loaded.sensor_ids, k):
            active_sensors = [int(s) for s in subset]
            _, _, score, per_var_r2, meets_thresholds, _ = evaluate_fixed_active_set_summary(
                model=model,
                cfg=cfg,
                checkpoint=checkpoint,
                loaded=loaded,
                device=device,
                active_sensors=active_sensors,
                protocol_name=f"exhaustive_k{k}_{'_'.join(map(str, active_sensors))}",
                display_name=f"Exhaustive active set k={k}: {active_sensors}",
                keep_predictions=False,
            )
            row = {
                "active_count": int(k),
                "active_sensors": json.dumps(active_sensors),
                "score": float(score),
                "meets_thresholds": bool(meets_thresholds),
            }
            for v in loaded.variable_names:
                row[f"r2_{v}"] = float(per_var_r2[v])
            rows.append(row)

    exhaustive_df = pd.DataFrame(rows)
    if exhaustive_df.empty:
        return exhaustive_df, []

    exhaustive_df = exhaustive_df.sort_values(["active_count", "score"], ascending=[True, False]).reset_index(drop=True)
    for k in sorted(exhaustive_df["active_count"].unique()):
        top_rows = exhaustive_df[exhaustive_df["active_count"] == k].head(top_n)
        for rank, (_, row) in enumerate(top_rows.iterrows(), start=1):
            active_sensors = json.loads(row["active_sensors"])
            top_protocols.append(
                protocol_from_active_set(
                    protocol_name=f"exhaustive_top{k}_rank{rank}",
                    display_name=f"Exhaustive top rank {rank} (k={k})",
                    sensor_ids=loaded.sensor_ids,
                    active_sensors=active_sensors,
                )
            )

    return exhaustive_df, top_protocols



def build_sensor_importance_summary(protocol_summary_df: pd.DataFrame, variable_names: Sequence[str], cfg: Mapping[str, Any]) -> pd.DataFrame:
    if protocol_summary_df.empty:
        return pd.DataFrame()
    ref = protocol_summary_df[protocol_summary_df["protocol_name"] == "full_reference_all_active"]
    if ref.empty:
        return pd.DataFrame()

    ref_by_var = ref.set_index("variable")
    weights = get_variable_weights(cfg, variable_names)
    ref_score = sum(weights[v] * float(ref_by_var.loc[v, "r2"]) for v in variable_names) / max(sum(weights.values()), 1e-12)

    rows: List[Dict[str, Any]] = []
    for protocol_name in sorted(p for p in protocol_summary_df["protocol_name"].unique() if p.startswith("drop_sensor_")):
        sub = protocol_summary_df[protocol_summary_df["protocol_name"] == protocol_name].set_index("variable")
        removed_sensor = int(protocol_name.split("_")[-1])
        score = sum(weights[v] * float(sub.loc[v, "r2"]) for v in variable_names) / max(sum(weights.values()), 1e-12)
        row = {
            "removed_sensor": removed_sensor,
            "score": float(score),
            "delta_score_vs_reference": float(score - ref_score),
        }
        for v in variable_names:
            row[f"rmse_{v}"] = float(sub.loc[v, "rmse"])
            row[f"mae_{v}"] = float(sub.loc[v, "mae"])
            row[f"r2_{v}"] = float(sub.loc[v, "r2"])
            row[f"delta_r2_{v}_vs_reference"] = float(sub.loc[v, "r2"] - ref_by_var.loc[v, "r2"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["score", "removed_sensor"], ascending=[False, True]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Plotting (preserved)
# -----------------------------------------------------------------------------
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
    if protocol_name not in outputs.pred_store:
        return
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
            ax.set_title(f"Sensor {sensor_id} | RMSE={metrics_row['rmse']:.3f} {units} | R²={metrics_row['r2']:.3f}", fontsize=9)
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
            f"DeepONet v0.5.2 — {variable.capitalize()} | {display_name} | masked_total={masked_total}",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        out_path = outputs.output_dir / f"thermodt_eval_timeseries_{_sanitize_name(protocol_name)}_{_sanitize_name(variable)}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved : {out_path}")



def save_scatter_plots(outputs: EvalOutputs, protocol_name: str, display_name: str, masked_total: int, point_size: float) -> None:
    if protocol_name not in outputs.pred_store:
        return
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
            ax.set_title(f"Sensor {sensor_id} | RMSE={metrics_row['rmse']:.3f} {units} | R²={metrics_row['r2']:.3f}", fontsize=9)
            ax.set_xlabel(f"Actual {variable.capitalize()} ({units})" if units else f"Actual {variable.capitalize()}", fontsize=8)
            ax.set_ylabel(f"Predicted {variable.capitalize()} ({units})" if units else f"Predicted {variable.capitalize()}", fontsize=8)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

        for ax in axes[n_sensors:]:
            ax.set_visible(False)

        fig.suptitle(
            f"DeepONet v0.5.2 — {variable.capitalize()} | {display_name} | masked_total={masked_total}",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        out_path = outputs.output_dir / f"thermodt_eval_scatter_{_sanitize_name(protocol_name)}_{_sanitize_name(variable)}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved : {out_path}")



def save_floormap_plots(outputs: EvalOutputs, protocol_name: str, display_name: str, masked_total: int, grid_res: int) -> None:
    if protocol_name not in outputs.pred_store:
        return
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
        panels = [("RMSE", rmse_vals, "RdYlGn_r"), ("MAE", mae_vals, "RdYlGn_r"), ("R²", r2_vals, "RdYlGn")]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"DeepONet v0.5.2 — {variable.capitalize()} | {display_name} | masked_total={masked_total}",
            fontsize=12,
            fontweight="bold",
        )

        for ax, (label, values, cmap) in zip(axes, panels):
            try:
                rbf = RBFInterpolator(coords_m, values, kernel="thin_plate_spline", smoothing=0)
                Z = rbf(grid_pts).reshape(GX.shape)
                im = ax.imshow(Z, origin="lower", extent=[0, outputs.room_width, 0, outputs.room_height], cmap=cmap, aspect="equal", interpolation="bilinear")
            except Exception:
                im = None
                ax.set_facecolor("white")
            ax.scatter(coords_m[:, 0], coords_m[:, 1], c=values, cmap=cmap, s=160, edgecolors="black", linewidths=0.9, zorder=5)
            for sid, (x, y), v in zip(outputs.sensor_ids, coords_m, values):
                ax.annotate(f"{sid}\n{v:.3f}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7, fontweight="bold", color="white", zorder=6, path_effects=[pe.withStroke(linewidth=2, foreground="black")])
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


# -----------------------------------------------------------------------------
# Main evaluation orchestrator
# -----------------------------------------------------------------------------
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
    out_dir = Path(cfg["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate the protocol families that should become part of the standard summaries.
    protocol_defs = build_eval_protocols(cfg, loaded.sensor_ids)
    keep_for_plot = {p.protocol_name for p in protocol_defs if _should_plot_protocol(cfg, p.protocol_name)}
    protocol_results: List[ProtocolEvalResult] = []
    pred_store: Dict[str, np.ndarray] = {}

    for protocol in protocol_defs:
        keep_predictions = bool(cfg.get("plot_timeseries", False) or cfg.get("plot_scatter", False) or cfg.get("plot_floormap", False)) and (protocol.protocol_name in keep_for_plot)
        result = evaluate_single_protocol(model, protocol, cfg, checkpoint, loaded, device, keep_predictions=keep_predictions)
        protocol_results.append(result)
        if keep_predictions and result.mean_preds_denorm is not None:
            pred_store[protocol.protocol_name] = result.mean_preds_denorm

    metrics_df = pd.concat([r.metrics_df for r in protocol_results], ignore_index=True) if protocol_results else pd.DataFrame()
    if not metrics_df.empty:
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
    else:
        protocol_summary_df = pd.DataFrame()
        sensor_summary_df = pd.DataFrame()

    # Sensor importance summary.
    sensor_importance_df = build_sensor_importance_summary(protocol_summary_df, loaded.variable_names, cfg)

    # Greedy backward elimination.
    backward_path_df, backward_candidates_df, backward_protocols = run_backward_elimination(model, cfg, checkpoint, loaded, device)

    # Exhaustive small-k search.
    exhaustive_df, exhaustive_top_protocols = run_exhaustive_active_search(model, cfg, checkpoint, loaded, device)

    # Evaluate selected active sets from backward elimination and exhaustive search so they can be plotted / compared.
    search_protocols = backward_protocols + exhaustive_top_protocols
    if search_protocols:
        search_results: List[ProtocolEvalResult] = []
        for protocol in search_protocols:
            keep_predictions = bool(cfg.get("plot_timeseries", False) or cfg.get("plot_scatter", False) or cfg.get("plot_floormap", False)) and _should_plot_protocol(cfg, protocol.protocol_name)
            result = evaluate_single_protocol(model, protocol, cfg, checkpoint, loaded, device, keep_predictions=keep_predictions)
            search_results.append(result)
            if keep_predictions and result.mean_preds_denorm is not None:
                pred_store[protocol.protocol_name] = result.mean_preds_denorm
        search_metrics_df = pd.concat([r.metrics_df for r in search_results], ignore_index=True)
        metrics_df = pd.concat([metrics_df, search_metrics_df], ignore_index=True) if not metrics_df.empty else search_metrics_df
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

    # Build outputs structure.
    test_idx = np.array(checkpoint["splits"]["test_idx"], dtype=int)
    test_data_norm = loaded.data_norm[test_idx]
    mean = np.array(checkpoint["stats"]["mean"], dtype=np.float32)
    std = np.array(checkpoint["stats"]["std"], dtype=np.float32)
    actual_denorm = test_data_norm * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)

    outputs = EvalOutputs(
        metrics_df=metrics_df,
        summary={},
        sensor_summary_df=sensor_summary_df,
        protocol_summary_df=protocol_summary_df,
        pred_store=pred_store,
        actual_denorm=actual_denorm,
        time_test=loaded.time_idx[test_idx],
        sensor_ids=[int(x) for x in checkpoint["sensors"]["sensor_ids"]],
        variable_names=list(checkpoint["variables"]),
        variable_units=dict(checkpoint.get("variable_units", {})),
        room_width=float(checkpoint["sensors"]["room_width"]),
        room_height=float(checkpoint["sensors"]["room_height"]),
        sensor_coords={int(k): tuple(v) for k, v in checkpoint["sensors"]["sensor_coords"].items()},
        test_windows=list(checkpoint["splits"]["test_windows"]),
        output_dir=out_dir,
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "overall": protocol_summary_df.to_dict(orient="records") if not protocol_summary_df.empty else [],
        "sensor_importance_top": sensor_importance_df.head(10).to_dict(orient="records") if not sensor_importance_df.empty else [],
        "backward_path": backward_path_df.to_dict(orient="records") if not backward_path_df.empty else [],
        "exhaustive_top_by_k": {},
    }
    if not exhaustive_df.empty:
        for k in sorted(exhaustive_df["active_count"].unique()):
            summary["exhaustive_top_by_k"][str(int(k))] = exhaustive_df[exhaustive_df["active_count"] == k].head(int(cfg.get("exhaustive_top_n_per_k", 10))).to_dict(orient="records")
    outputs.summary = summary

    # Save core outputs.
    csv_path = out_dir / str(cfg["eval_csv_name"])
    summary_path = out_dir / str(cfg["eval_summary_name"])
    sensor_summary_path = out_dir / str(cfg["eval_sensor_summary_name"])
    protocol_summary_path = out_dir / str(cfg["eval_protocol_summary_name"])
    importance_path = out_dir / str(cfg["sensor_importance_summary_name"])
    backward_path_file = out_dir / str(cfg["backward_elimination_path_name"])
    backward_candidates_file = out_dir / str(cfg["backward_elimination_candidates_name"])
    exhaustive_path = out_dir / str(cfg["exhaustive_subset_search_name"])

    outputs.metrics_df.to_csv(csv_path, index=False)
    outputs.sensor_summary_df.to_csv(sensor_summary_path, index=False)
    outputs.protocol_summary_df.to_csv(protocol_summary_path, index=False)
    if not sensor_importance_df.empty:
        sensor_importance_df.to_csv(importance_path, index=False)
    if not backward_path_df.empty:
        backward_path_df.to_csv(backward_path_file, index=False)
    if not backward_candidates_df.empty:
        backward_candidates_df.to_csv(backward_candidates_file, index=False)
    if not exhaustive_df.empty:
        exhaustive_df.to_csv(exhaustive_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 92)
    print(f"Evaluation complete | checkpoint={checkpoint_path.name} | device={device}")
    print("=" * 92)
    if not protocol_summary_df.empty:
        print(protocol_summary_df.set_index(["protocol_name", "masked_total", "variable"])[["rmse", "mae", "r2"]])
    if not sensor_importance_df.empty:
        print("\nTop sensor-importance rows:")
        print(sensor_importance_df.head(10).to_string(index=False))
    if not backward_path_df.empty:
        print("\nBackward elimination path:")
        print(backward_path_df.to_string(index=False))
    if not exhaustive_df.empty:
        print("\nBest exhaustive subsets by k:")
        print(exhaustive_df.groupby("active_count").head(5).to_string(index=False))

    print(f"\nSaved raw metrics           : {csv_path}")
    print(f"Saved sensor summary        : {sensor_summary_path}")
    print(f"Saved protocol summary      : {protocol_summary_path}")
    if not sensor_importance_df.empty:
        print(f"Saved sensor importance     : {importance_path}")
    if not backward_path_df.empty:
        print(f"Saved backward path         : {backward_path_file}")
        print(f"Saved backward candidates   : {backward_candidates_file}")
    if not exhaustive_df.empty:
        print(f"Saved exhaustive subsets    : {exhaustive_path}")
    print(f"Saved summary JSON          : {summary_path}")

    # Optional plotting.
    protocol_lookup = outputs.protocol_summary_df[["protocol_name", "display_name", "masked_total"]].drop_duplicates() if not outputs.protocol_summary_df.empty else pd.DataFrame()
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
