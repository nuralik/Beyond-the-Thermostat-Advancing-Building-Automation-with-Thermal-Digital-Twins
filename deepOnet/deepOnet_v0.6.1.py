from __future__ import annotations

"""
DeepONet v0.6.1 — Step 6 Phase 1 (Lab1 -> Lab2 transfer)
==========================================================

Built from the v0.5.3 deep MLP DeepONet, with:
- blocked 80/10/10 train/val/test split for Lab1
- blocked Lab2 transfer blocked:
    * 10% held-out test
    * 16% adaptation train
    * 4% adaptation val
    * remaining rows unused
- Phase 1 transfer modes:
    * zero-shot on Lab2
    * full fine-tuning on Lab2 with a small LR
- Lab2 evaluation sensor regimes:
    * all 13 sensors active (target always masked)
    * fixed 5-sensor active set [11, 4, 6, 9, 1]
    * fixed 3-sensor active set [2, 5, 10]

Important note
--------------
This script expects Lab2 geometry to be provided in PARAMS["labs"]["lab2"]
below. If those coordinates are still placeholders, fill them before running.
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

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


PARAMS: Dict[str, Any] = {
    "mode": "train_and_phase1",  # train_lab1, phase1_transfer, train_and_phase1
    "seed": 42,
    "device": "auto",  # auto, cpu, cuda, mps
    "experiment_name": "lab1_to_lab2_transfer_v0.6.1",
    "data_root": "./data",
    "resolution": "5min",
    "time_file": "Time_5min.xlsx",
    "time_column": "tin",
    "scale": 100.0,
    "target_variables": [
        {"name": "temperature", "file": "Temperature_5min.xlsx", "column_prefix": "tem", "units": "°C"},
        {"name": "humidity", "file": "Humidity_5min.xlsx", "column_prefix": "hum", "units": "%RH"},
    ],
    "sensor_ids": list(range(1, 14)),
    "labs": {
        "lab1": {
            "room_width": 7.81,
            "room_height": 7.82,
            "sensor_coords": {
                1: (7.21, 0.60),  2: (3.88, 5.88),  3: (5.85, 5.88),
                4: (0.60, 0.60),  5: (1.92, 1.94),  6: (3.88, 3.91),
                7: (1.92, 3.91),  8: (5.85, 3.91),  9: (7.21, 7.22),
                10: (1.92, 5.88), 11: (0.60, 7.22), 12: (5.85, 1.94),
                13: (3.88, 1.94),
            },
        },
        "lab2": {
            "room_width": 6.60,
            "room_height": 12.60,
            "sensor_coords": {
                1: (5.70, 7.54), 2: (5.70, 10.17), 3: (3.30, 11.60),
                4: (3.30, 9.12), 5: (3.30, 6.34), 6: (3.30, 3.86),
                7: (3.30, 1.38), 8: (5.70, 4.91), 9: (5.70, 2.43),
                10: (0.90, 2.43), 11: (0.90, 10.17), 12: (0.90, 4.91),
                13: (0.90, 7.54),
            },
        },
    },

    # Lab1 blocked split (Option B): approximately 80/10/10 using fixed-length windows.
    "lab1_split": {
        "block_hours": 6,
        "test_fraction": 0.10,
        "val_fraction": 0.10,
        "seed_offset": 100,
    },

    # Lab2 Phase-1 blocked split.
    # Fractions are w.r.t. the full Lab2 timeline and are selected as blocked windows.
    "lab2_split": {
        "block_hours": 6,
        "test_fraction": 0.10,
        "adapt_train_fraction": 0.16,
        "adapt_val_fraction": 0.04,
        "seed_offset": 200,
    },

    # Reconstruction / masking.
    "include_observed_mask": True,
    "mask_fill_value": 0.0,
    "train_extra_mask_min": 1,
    "train_extra_mask_max": 12,

    # v0.5.3 architecture.
    "hidden": 128,
    "latent_dim": 128,
    "depth": 6,
    "activation": "tanh",  # tanh, relu, gelu

    # Training.
    "batch_size": 4096,
    "epochs": 150,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "num_workers": 0,
    "early_stopping_enabled": True,
    "early_stopping_patience": 20,
    "early_stopping_min_delta": 1e-4,

    # Phase-1 transfer modes.
    "run_lab1_test_eval": True,
    "run_lab2_zero_shot": True,
    "run_lab2_full_finetune": True,
    "full_finetune_lr": 2e-4,
    "full_finetune_epochs": 100,
    "full_finetune_patience": 15,
    "full_finetune_min_delta": 1e-4,

    # Lab2 sensor regimes for Phase 1.
    "lab2_phase1_protocols": [
        {
            "protocol_name": "lab2_all13",
            "display_name": "Lab2 all 13 active",
            "active_sensors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        },
        {
            "protocol_name": "lab2_active_5",
            "display_name": "Lab2 active set [11, 4, 6, 9, 1]",
            "active_sensors": [11, 4, 6, 9, 1],
        },
        {
            "protocol_name": "lab2_active_3",
            "display_name": "Lab2 active set [2, 5, 10]",
            "active_sensors": [2, 5, 10],
        },
    ],

    # Output.
    "output_dir": "./deepOnet/runs/lab1_to_lab2_v0.6.1",
    "best_checkpoint_name": "thermodt_lab1_best.pt",
    "final_checkpoint_name": "thermodt_lab1_final.pt",
    "full_ft_best_checkpoint_name": "thermodt_lab2_fullft_best.pt",
    "full_ft_final_checkpoint_name": "thermodt_lab2_fullft_final.pt",
    "lab1_eval_metrics_name": "thermodt_lab1_test_metrics_v061.csv",
    "phase1_metrics_name": "thermodt_phase1_lab2_metrics_v061.csv",
    "phase1_sensor_summary_name": "thermodt_phase1_lab2_sensor_summary_v061.csv",
    "phase1_protocol_summary_name": "thermodt_phase1_lab2_protocol_summary_v061.csv",
    "phase1_summary_name": "thermodt_phase1_lab2_summary_v061.json",
}


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@dataclass
class RawLabData:
    lab_name: str
    data_raw: np.ndarray  # (time, sensor, variable)
    time_idx: pd.DatetimeIndex
    coords_norm: np.ndarray
    sensor_ids: List[int]
    variable_names: List[str]
    variable_units: Dict[str, str]
    room_width: float
    room_height: float
    sensor_coords: Dict[int, Tuple[float, float]]


@dataclass
class NormalizedLabData:
    lab_name: str
    data_norm: np.ndarray
    time_idx: pd.DatetimeIndex
    coords_norm: np.ndarray
    sensor_ids: List[int]
    variable_names: List[str]
    variable_units: Dict[str, str]
    mean: np.ndarray
    std: np.ndarray
    room_width: float
    room_height: float
    sensor_coords: Dict[int, Tuple[float, float]]


@dataclass
class SplitMasks:
    masks: Dict[str, np.ndarray]
    meta: Dict[str, Any]


@dataclass
class Batch:
    branch_input: torch.Tensor
    trunk_coord: torch.Tensor
    target: torch.Tensor


@dataclass
class Phase1Protocol:
    protocol_name: str
    display_name: str
    active_sensors: List[int]


# -----------------------------------------------------------------------------
# Utilities
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



def resolve_data_dir(data_root: str, lab_name: str, resolution: str) -> Path:
    primary = Path(data_root) / f"{lab_name}_{resolution}"
    fallback = Path(data_root) / lab_name
    return primary.resolve() if primary.exists() else fallback.resolve()



def infer_steps_per_hour(time_idx: pd.DatetimeIndex) -> int:
    if len(time_idx) < 2:
        return 1
    diffs_ns = np.diff(time_idx.view("i8"))
    median_seconds = float(np.median(diffs_ns) / 1e9)
    if median_seconds <= 0:
        return 1
    return max(1, int(round(3600.0 / median_seconds)))



def _candidate_block_starts(mask_available: np.ndarray, window_steps: int) -> List[int]:
    starts: List[int] = []
    n = len(mask_available)
    for start in range(0, max(0, n - window_steps + 1)):
        end = start + window_steps
        if mask_available[start:end].all():
            starts.append(start)
    return starts



def choose_block_starts_by_fraction(
    time_idx: pd.DatetimeIndex,
    occupied_mask: np.ndarray,
    target_fraction: float,
    block_hours: int,
    seed: int,
) -> List[int]:
    steps_per_hour = infer_steps_per_hour(time_idx)
    window_steps = max(1, steps_per_hour * int(block_hours))
    target_steps = max(1, int(round(len(time_idx) * float(target_fraction))))
    target_windows = max(1, int(round(target_steps / window_steps)))

    available_mask = ~occupied_mask
    candidate_starts = _candidate_block_starts(available_mask, window_steps)
    if not candidate_starts:
        return []
    target_windows = min(target_windows, len(candidate_starts))

    rng = np.random.default_rng(seed)
    anchors = np.linspace(
        0,
        len(candidate_starts) - 1,
        num=min(max(target_windows * 4, 1), len(candidate_starts)),
        dtype=int,
    )
    ordered = [candidate_starts[i] for i in anchors]
    ordered.extend(candidate_starts)

    # Deduplicate preserving order, then shuffle a copy lightly for diversity.
    seen = set()
    unique_ordered: List[int] = []
    for x in ordered:
        if x not in seen:
            unique_ordered.append(x)
            seen.add(x)
    if len(unique_ordered) > 1:
        front = unique_ordered[: max(target_windows * 2, 1)]
        back = unique_ordered[max(target_windows * 2, 1):]
        rng.shuffle(front)
        unique_ordered = front + back

    selected: List[int] = []
    taken = occupied_mask.copy()
    for start in unique_ordered:
        end = start + window_steps
        if taken[start:end].any():
            continue
        selected.append(start)
        taken[start:end] = True
        if len(selected) >= target_windows:
            break
    return sorted(selected)



def starts_to_mask(n: int, starts: Sequence[int], window_steps: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for start in starts:
        end = min(n, start + window_steps)
        mask[start:end] = True
    return mask



def starts_to_meta(time_idx: pd.DatetimeIndex, starts: Sequence[int], window_steps: int) -> List[Dict[str, Any]]:
    meta: List[Dict[str, Any]] = []
    for start in starts:
        end = min(len(time_idx), start + window_steps)
        ts0 = pd.Timestamp(time_idx[start])
        ts1 = pd.Timestamp(time_idx[end - 1]) if end > start else ts0
        meta.append({
            "start_idx": int(start),
            "end_idx_exclusive": int(end),
            "start": str(ts0),
            "end": str(ts1),
            "start_hour": int(ts0.hour),
        })
    return meta



def build_lab1_blocked_split(time_idx: pd.DatetimeIndex, cfg: Mapping[str, Any], seed: int) -> SplitMasks:
    split_cfg = cfg["lab1_split"]
    block_hours = int(split_cfg["block_hours"])
    steps_per_hour = infer_steps_per_hour(time_idx)
    window_steps = max(1, steps_per_hour * block_hours)

    occupied = np.zeros(len(time_idx), dtype=bool)
    test_starts = choose_block_starts_by_fraction(time_idx, occupied, float(split_cfg["test_fraction"]), block_hours, seed + 1)
    test_mask = starts_to_mask(len(time_idx), test_starts, window_steps)
    occupied |= test_mask

    val_starts = choose_block_starts_by_fraction(time_idx, occupied, float(split_cfg["val_fraction"]), block_hours, seed + 2)
    val_mask = starts_to_mask(len(time_idx), val_starts, window_steps)
    occupied |= val_mask

    train_mask = ~(test_mask | val_mask)
    meta = {
        "split_type": "blocked_fraction",
        "block_hours": block_hours,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "train_fraction_realized": float(train_mask.mean()),
        "val_fraction_realized": float(val_mask.mean()),
        "test_fraction_realized": float(test_mask.mean()),
        "val_windows": starts_to_meta(time_idx, val_starts, window_steps),
        "test_windows": starts_to_meta(time_idx, test_starts, window_steps),
    }
    return SplitMasks(masks={"train": train_mask, "val": val_mask, "test": test_mask}, meta=meta)



def build_lab2_phase1_split(time_idx: pd.DatetimeIndex, cfg: Mapping[str, Any], seed: int) -> SplitMasks:
    split_cfg = cfg["lab2_split"]
    block_hours = int(split_cfg["block_hours"])
    steps_per_hour = infer_steps_per_hour(time_idx)
    window_steps = max(1, steps_per_hour * block_hours)

    occupied = np.zeros(len(time_idx), dtype=bool)
    test_starts = choose_block_starts_by_fraction(time_idx, occupied, float(split_cfg["test_fraction"]), block_hours, seed + 11)
    test_mask = starts_to_mask(len(time_idx), test_starts, window_steps)
    occupied |= test_mask

    adapt_val_starts = choose_block_starts_by_fraction(time_idx, occupied, float(split_cfg["adapt_val_fraction"]), block_hours, seed + 12)
    adapt_val_mask = starts_to_mask(len(time_idx), adapt_val_starts, window_steps)
    occupied |= adapt_val_mask

    adapt_train_starts = choose_block_starts_by_fraction(time_idx, occupied, float(split_cfg["adapt_train_fraction"]), block_hours, seed + 13)
    adapt_train_mask = starts_to_mask(len(time_idx), adapt_train_starts, window_steps)
    occupied |= adapt_train_mask

    unused_mask = ~(test_mask | adapt_val_mask | adapt_train_mask)
    meta = {
        "split_type": "blocked_fraction_phase1",
        "block_hours": block_hours,
        "n_adapt_train": int(adapt_train_mask.sum()),
        "n_adapt_val": int(adapt_val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_unused": int(unused_mask.sum()),
        "adapt_train_fraction_realized": float(adapt_train_mask.mean()),
        "adapt_val_fraction_realized": float(adapt_val_mask.mean()),
        "test_fraction_realized": float(test_mask.mean()),
        "adapt_train_windows": starts_to_meta(time_idx, adapt_train_starts, window_steps),
        "adapt_val_windows": starts_to_meta(time_idx, adapt_val_starts, window_steps),
        "test_windows": starts_to_meta(time_idx, test_starts, window_steps),
    }
    return SplitMasks(
        masks={
            "adapt_train": adapt_train_mask,
            "adapt_val": adapt_val_mask,
            "test": test_mask,
            "unused": unused_mask,
        },
        meta=meta,
    )


# -----------------------------------------------------------------------------
# Data loading / normalization
# -----------------------------------------------------------------------------
def validate_lab_geometry(lab_name: str, lab_cfg: Mapping[str, Any], sensor_ids: Sequence[int]) -> None:
    if lab_cfg.get("room_width") in (None, 0) or lab_cfg.get("room_height") in (None, 0):
        raise ValueError(f"{lab_name}: room_width / room_height must be set.")
    coords = lab_cfg.get("sensor_coords", {})
    missing = [sid for sid in sensor_ids if int(sid) not in {int(k) for k in coords.keys()}]
    if missing:
        raise ValueError(f"{lab_name}: missing sensor coordinates for sensor ids: {missing}")



def load_raw_lab_data(cfg: Mapping[str, Any], lab_name: str) -> RawLabData:
    sensor_ids = [int(x) for x in cfg["sensor_ids"]]
    lab_cfg = cfg["labs"][lab_name]
    validate_lab_geometry(lab_name, lab_cfg, sensor_ids)

    data_dir = resolve_data_dir(str(cfg["data_root"]), lab_name, str(cfg["resolution"]))
    time_df = pd.read_excel(data_dir / str(cfg["time_file"]))
    time_idx = pd.to_datetime(time_df[str(cfg["time_column"])])

    variable_arrays: List[np.ndarray] = []
    variable_names: List[str] = []
    variable_units: Dict[str, str] = {}
    for var_cfg in cfg["target_variables"]:
        df = pd.read_excel(data_dir / str(var_cfg["file"]))
        cols = [f"{var_cfg['column_prefix']}{sid}" for sid in sensor_ids]
        arr = np.stack([df[col].to_numpy(dtype=float) for col in cols], axis=1) / float(cfg["scale"])
        variable_arrays.append(arr)
        variable_names.append(str(var_cfg["name"]))
        variable_units[str(var_cfg["name"])] = str(var_cfg.get("units", ""))

    data_raw = np.stack(variable_arrays, axis=-1)  # (time, sensor, variable)
    valid_mask = ~np.isnan(data_raw).any(axis=(1, 2))
    data_raw = data_raw[valid_mask]
    time_idx = pd.DatetimeIndex(np.asarray(time_idx)[valid_mask])

    sensor_coords = {int(k): tuple(v) for k, v in lab_cfg["sensor_coords"].items()}
    room_width = float(lab_cfg["room_width"])
    room_height = float(lab_cfg["room_height"])
    coords_norm = np.array(
        [(sensor_coords[sid][0] / room_width, sensor_coords[sid][1] / room_height) for sid in sensor_ids],
        dtype=np.float32,
    )

    return RawLabData(
        lab_name=lab_name,
        data_raw=data_raw.astype(np.float32),
        time_idx=time_idx,
        coords_norm=coords_norm,
        sensor_ids=sensor_ids,
        variable_names=variable_names,
        variable_units=variable_units,
        room_width=room_width,
        room_height=room_height,
        sensor_coords=sensor_coords,
    )



def normalize_raw_lab_data(raw: RawLabData, mean: np.ndarray, std: np.ndarray) -> NormalizedLabData:
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    data_norm = (raw.data_raw - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    return NormalizedLabData(
        lab_name=raw.lab_name,
        data_norm=data_norm.astype(np.float32),
        time_idx=raw.time_idx,
        coords_norm=raw.coords_norm,
        sensor_ids=raw.sensor_ids,
        variable_names=raw.variable_names,
        variable_units=raw.variable_units,
        mean=mean,
        std=std,
        room_width=raw.room_width,
        room_height=raw.room_height,
        sensor_coords=raw.sensor_coords,
    )


# -----------------------------------------------------------------------------
# Dataset / model
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
        return Batch(torch.from_numpy(branch_input), torch.from_numpy(self.coords_norm[s]), torch.from_numpy(target))



def collate_batches(items: Sequence[Batch]) -> Batch:
    return Batch(
        branch_input=torch.stack([x.branch_input for x in items], dim=0),
        trunk_coord=torch.stack([x.trunk_coord for x in items], dim=0),
        target=torch.stack([x.target for x in items], dim=0),
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
    def __init__(self, branch_in_dim: int, coord_dim: int, output_dim: int, hidden: int, latent_dim: int, depth: int, activation: str):
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



def make_loader(data_norm: np.ndarray, coords_norm: np.ndarray, cfg: Mapping[str, Any], seed: int, extra_min: int, extra_max: int, device: torch.device, shuffle: bool) -> DataLoader:
    ds = MultiSensorMaskedDataset(
        data_norm=data_norm,
        coords_norm=coords_norm,
        mask_fill_value=float(cfg["mask_fill_value"]),
        include_observed_mask=bool(cfg["include_observed_mask"]),
        extra_mask_min=int(extra_min),
        extra_mask_max=int(extra_max),
        seed=int(seed),
    )
    return DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["num_workers"]),
        pin_memory=(str(device) != "cpu"),
        collate_fn=collate_batches,
    )


# -----------------------------------------------------------------------------
# Training / checkpointing
# -----------------------------------------------------------------------------
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
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
    split_meta: Mapping[str, Any],
    lab_name: str,
    normalized: NormalizedLabData,
    model: nn.Module,
    epoch: int,
    train_loss: float,
    val_loss: float,
    history: Dict[str, List[float]],
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "history": history,
        "config": copy.deepcopy(dict(cfg)),
        "lab_name": lab_name,
        "stats": {"mean": normalized.mean.tolist(), "std": normalized.std.tolist()},
        "splits": copy.deepcopy(dict(split_meta)),
        "sensors": {
            "sensor_ids": normalized.sensor_ids,
            "coords_norm": normalized.coords_norm.tolist(),
            "sensor_coords": {int(k): list(v) for k, v in normalized.sensor_coords.items()},
            "room_width": float(normalized.room_width),
            "room_height": float(normalized.room_height),
        },
        "variables": normalized.variable_names,
        "variable_units": normalized.variable_units,
        "time": {
            "n_timesteps": int(len(normalized.time_idx)),
            "start": str(normalized.time_idx[0]) if len(normalized.time_idx) else None,
            "end": str(normalized.time_idx[-1]) if len(normalized.time_idx) else None,
        },
    }
    if extra:
        payload.update(copy.deepcopy(dict(extra)))
    return payload



def train_model(
    cfg: Mapping[str, Any],
    normalized: NormalizedLabData,
    split_masks: SplitMasks,
    train_key: str,
    val_key: str,
    device: torch.device,
    out_dir: Path,
    best_name: str,
    final_name: str,
    lr: float,
    epochs: int,
    patience: int,
    min_delta: float,
    seed: int,
    init_state_dict: Mapping[str, Any] | None = None,
    tag: str = "train",
) -> Tuple[Path, Path, Dict[str, Any]]:
    train_idx = np.where(split_masks.masks[train_key])[0]
    val_idx = np.where(split_masks.masks[val_key])[0]
    train_loader = make_loader(normalized.data_norm[train_idx], normalized.coords_norm, cfg, seed, int(cfg["train_extra_mask_min"]), int(cfg["train_extra_mask_max"]), device, shuffle=True)
    val_loader = make_loader(normalized.data_norm[val_idx], normalized.coords_norm, cfg, seed + 9999, 0, 0, device, shuffle=False)

    model = build_model(cfg, len(normalized.sensor_ids), len(normalized.variable_names)).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs))
    criterion = nn.MSELoss()

    best_path = out_dir / best_name
    final_path = out_dir / final_name
    best_val = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}
    stopped_early = False
    stop_epoch = int(epochs)

    print("=" * 92)
    print(f"DeepONet v0.6.1 | {tag} | lab={normalized.lab_name} | device={device}")
    print("=" * 92)
    print(f"Variables         : {normalized.variable_names}")
    print(f"Sensors           : {len(normalized.sensor_ids)}")
    print(f"Timesteps         : {len(normalized.time_idx):,}")
    print(f"Train rows        : {len(train_idx):,}")
    print(f"Val rows          : {len(val_idx):,}")
    print(f"LR / epochs       : {lr:g} / {epochs}")
    print(f"Early stopping    : enabled | patience={patience} | min_delta={min_delta:g}")

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            branch_input = batch.branch_input.to(device)
            trunk_coord = batch.trunk_coord.to(device)
            target = batch.target.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(branch_input, trunk_coord)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * target.shape[0]
            total_count += target.shape[0]

        train_loss = float(total_loss / max(1, total_count))
        val_loss = evaluate_loader(model, val_loader, device)
        scheduler.step()
        current_lr = float(scheduler.get_last_lr()[0])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        if is_improved(val_loss, best_val, min_delta):
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                package_checkpoint_payload(
                    cfg, split_masks.meta, normalized.lab_name, normalized, model, epoch, train_loss, val_loss, history,
                    extra={
                        "tag": tag,
                        "best_epoch": int(best_epoch),
                        "best_val_loss": float(best_val),
                        "stopped_early": False,
                        "stop_epoch": int(epoch),
                        "epochs_no_improve_final": int(epochs_no_improve),
                    },
                ),
                best_path,
            )
            marker = " *** best"
        else:
            epochs_no_improve += 1
            marker = f" | no_improve={epochs_no_improve}/{patience}"

        print(
            f"Epoch {epoch:>4}/{int(epochs)}  train={train_loss:.6f}  val={val_loss:.6f}  lr={current_lr:.2e}{marker}",
            flush=True,
        )

        if epochs_no_improve >= patience:
            stopped_early = True
            stop_epoch = epoch
            print(
                f"Early stopping triggered at epoch {epoch} (best epoch={best_epoch}, best val={best_val:.6f}).",
                flush=True,
            )
            break

    final_payload = package_checkpoint_payload(
        cfg, split_masks.meta, normalized.lab_name, normalized, model, stop_epoch, history["train_loss"][-1], history["val_loss"][-1], history,
        extra={
            "tag": tag,
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "stopped_early": bool(stopped_early),
            "stop_epoch": int(stop_epoch),
            "epochs_no_improve_final": int(epochs_no_improve),
        },
    )
    torch.save(final_payload, final_path)

    print("\nSaved artifacts")
    print(f"  Best checkpoint : {best_path}")
    print(f"  Final checkpoint: {final_path}")
    print(f"  Best epoch      : {best_epoch}")
    if stopped_early:
        print(f"  Training stop   : early stop at epoch {stop_epoch}")
    else:
        print(f"  Training stop   : completed {stop_epoch} epochs")

    return best_path, final_path, {"best_epoch": best_epoch, "best_val_loss": best_val, "stop_epoch": stop_epoch, "stopped_early": stopped_early}


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def build_phase1_protocols(cfg: Mapping[str, Any], sensor_ids: Sequence[int]) -> List[Phase1Protocol]:
    sensor_set = set(int(s) for s in sensor_ids)
    protocols: List[Phase1Protocol] = []
    for entry in cfg["lab2_phase1_protocols"]:
        active = [int(x) for x in entry["active_sensors"]]
        bad = [x for x in active if x not in sensor_set]
        if bad:
            raise ValueError(f"Protocol {entry['protocol_name']} references unknown Lab2 sensor ids: {bad}")
        protocols.append(Phase1Protocol(str(entry["protocol_name"]), str(entry["display_name"]), active))
    return protocols


@torch.no_grad()
def evaluate_active_protocols(
    model: nn.Module,
    normalized: NormalizedLabData,
    eval_mask: np.ndarray,
    protocols: Sequence[Phase1Protocol],
    device: torch.device,
    transfer_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model.eval()
    eval_idx = np.where(eval_mask)[0]
    test_data_norm = normalized.data_norm[eval_idx]
    actual_denorm = test_data_norm * normalized.std.reshape(1, 1, -1) + normalized.mean.reshape(1, 1, -1)
    coords_norm = normalized.coords_norm
    sensor_ids = normalized.sensor_ids

    records: List[Dict[str, Any]] = []
    for protocol in protocols:
        active_seed = [int(x) for x in protocol.active_sensors]
        preds_denorm = np.zeros_like(actual_denorm)

        for t in range(test_data_norm.shape[0]):
            for s_idx, sensor_id in enumerate(sensor_ids):
                values = test_data_norm[t].copy()
                observed = np.ones((len(sensor_ids),), dtype=np.float32)
                removed = [sid for sid in sensor_ids if sid not in active_seed]
                if sensor_id not in removed:
                    removed.append(sensor_id)
                removed_idx = [sensor_ids.index(sid) for sid in sorted(set(removed))]
                values[removed_idx, :] = float(PARAMS["mask_fill_value"])
                observed[removed_idx] = 0.0

                branch_parts = [values.reshape(-1)]
                if PARAMS["include_observed_mask"]:
                    branch_parts.append(observed)
                branch_input = np.concatenate(branch_parts, axis=0).astype(np.float32)
                pred = model(
                    torch.from_numpy(branch_input).unsqueeze(0).to(device),
                    torch.from_numpy(coords_norm[s_idx]).unsqueeze(0).to(device),
                ).cpu().numpy()[0]
                preds_denorm[t, s_idx, :] = pred * normalized.std + normalized.mean

        for s_idx, sensor_id in enumerate(sensor_ids):
            for c_idx, var_name in enumerate(normalized.variable_names):
                y_true = actual_denorm[:, s_idx, c_idx]
                y_pred = preds_denorm[:, s_idx, c_idx]
                mse = float(np.mean((y_true - y_pred) ** 2))
                rmse = float(np.sqrt(mse))
                mae = float(np.mean(np.abs(y_true - y_pred)))
                ss_res = float(np.sum((y_true - y_pred) ** 2))
                ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
                records.append({
                    "transfer_mode": transfer_mode,
                    "protocol_name": protocol.protocol_name,
                    "display_name": protocol.display_name,
                    "masked_total": int(len(sensor_ids) - len(protocol.active_sensors)),
                    "active_total_nominal": int(len(protocol.active_sensors)),
                    "sensor": int(sensor_id),
                    "variable": var_name,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                })

    metrics_df = pd.DataFrame(records)
    sensor_summary_df = (
        metrics_df.groupby(["transfer_mode", "protocol_name", "display_name", "masked_total", "sensor", "variable"])[["rmse", "mae", "r2"]]
        .mean().reset_index()
    )
    protocol_summary_df = (
        metrics_df.groupby(["transfer_mode", "protocol_name", "display_name", "masked_total", "variable"])[["rmse", "mae", "r2"]]
        .mean().reset_index()
    )
    return metrics_df, sensor_summary_df, protocol_summary_df


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def main(cfg: Dict[str, Any] | None = None) -> None:
    cfg = copy.deepcopy(PARAMS if cfg is None else cfg)
    set_seed(int(cfg["seed"]))
    device = get_device(str(cfg["device"]))
    out_dir = Path(cfg["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = str(cfg["mode"]).lower()

    # ---- Lab1 load / split / normalization ----
    raw_lab1 = load_raw_lab_data(cfg, "lab1")
    lab1_split = build_lab1_blocked_split(raw_lab1.time_idx, cfg, int(cfg["seed"]) + int(cfg["lab1_split"]["seed_offset"]))
    lab1_train_idx = np.where(lab1_split.masks["train"])[0]
    lab1_mean = raw_lab1.data_raw[lab1_train_idx].mean(axis=(0, 1)).astype(np.float32)
    lab1_std = raw_lab1.data_raw[lab1_train_idx].std(axis=(0, 1)).astype(np.float32)
    lab1_std = np.where(lab1_std < 1e-8, 1.0, lab1_std).astype(np.float32)
    lab1_norm = normalize_raw_lab_data(raw_lab1, lab1_mean, lab1_std)

    best_lab1_path = out_dir / str(cfg["best_checkpoint_name"])
    final_lab1_path = out_dir / str(cfg["final_checkpoint_name"])

    if mode in ("train_lab1", "train_and_phase1"):
        train_model(
            cfg=cfg,
            normalized=lab1_norm,
            split_masks=lab1_split,
            train_key="train",
            val_key="val",
            device=device,
            out_dir=out_dir,
            best_name=str(cfg["best_checkpoint_name"]),
            final_name=str(cfg["final_checkpoint_name"]),
            lr=float(cfg["lr"]),
            epochs=int(cfg["epochs"]),
            patience=int(cfg["early_stopping_patience"]),
            min_delta=float(cfg["early_stopping_min_delta"]),
            seed=int(cfg["seed"]),
            init_state_dict=None,
            tag="lab1_pretrain",
        )

    if mode == "train_lab1":
        return

    if not best_lab1_path.exists():
        raise FileNotFoundError(f"Lab1 checkpoint not found: {best_lab1_path}")

    lab1_ckpt = torch.load(best_lab1_path, map_location=device)
    lab1_model = build_model(cfg, len(lab1_norm.sensor_ids), len(lab1_norm.variable_names)).to(device)
    lab1_model.load_state_dict(lab1_ckpt["model_state"])

    # Optional Lab1 test eval for reference.
    summary_json: Dict[str, Any] = {
        "lab1_split": lab1_split.meta,
        "lab2_split": None,
        "lab1_reference": None,
        "phase1_lab2": {},
    }
    if bool(cfg.get("run_lab1_test_eval", True)):
        ref_protocols = build_phase1_protocols(cfg, lab1_norm.sensor_ids)
        lab1_metrics, lab1_sensor_summary, lab1_protocol_summary = evaluate_active_protocols(
            lab1_model, lab1_norm, lab1_split.masks["test"], ref_protocols, device, transfer_mode="lab1_reference"
        )
        lab1_metrics.to_csv(out_dir / str(cfg["lab1_eval_metrics_name"]), index=False)
        summary_json["lab1_reference"] = lab1_protocol_summary.to_dict(orient="records")
        print("=" * 92)
        print("Lab1 reference evaluation complete")
        print("=" * 92)
        print(lab1_protocol_summary.set_index(["protocol_name", "masked_total", "variable"])[["rmse", "mae", "r2"]])

    # ---- Lab2 load / normalize with Lab1 stats / split ----
    raw_lab2 = load_raw_lab_data(cfg, "lab2")
    lab2_norm = normalize_raw_lab_data(raw_lab2, lab1_mean, lab1_std)
    lab2_split = build_lab2_phase1_split(raw_lab2.time_idx, cfg, int(cfg["seed"]) + int(cfg["lab2_split"]["seed_offset"]))
    summary_json["lab2_split"] = lab2_split.meta
    phase1_protocols = build_phase1_protocols(cfg, lab2_norm.sensor_ids)

    all_metrics: List[pd.DataFrame] = []
    all_sensor_summary: List[pd.DataFrame] = []
    all_protocol_summary: List[pd.DataFrame] = []

    # ---- Zero-shot ----
    if bool(cfg.get("run_lab2_zero_shot", True)):
        zero_metrics, zero_sensor_summary, zero_protocol_summary = evaluate_active_protocols(
            lab1_model, lab2_norm, lab2_split.masks["test"], phase1_protocols, device, transfer_mode="zero_shot"
        )
        all_metrics.append(zero_metrics)
        all_sensor_summary.append(zero_sensor_summary)
        all_protocol_summary.append(zero_protocol_summary)
        summary_json["phase1_lab2"]["zero_shot"] = zero_protocol_summary.to_dict(orient="records")
        print("=" * 92)
        print("Lab2 zero-shot evaluation complete")
        print("=" * 92)
        print(zero_protocol_summary.set_index(["protocol_name", "masked_total", "variable"])[["rmse", "mae", "r2"]])

    # ---- Full fine-tune ----
    if bool(cfg.get("run_lab2_full_finetune", True)):
        ft_best, ft_final, ft_meta = train_model(
            cfg=cfg,
            normalized=lab2_norm,
            split_masks=lab2_split,
            train_key="adapt_train",
            val_key="adapt_val",
            device=device,
            out_dir=out_dir,
            best_name=str(cfg["full_ft_best_checkpoint_name"]),
            final_name=str(cfg["full_ft_final_checkpoint_name"]),
            lr=float(cfg["full_finetune_lr"]),
            epochs=int(cfg["full_finetune_epochs"]),
            patience=int(cfg["full_finetune_patience"]),
            min_delta=float(cfg["full_finetune_min_delta"]),
            seed=int(cfg["seed"]) + 5000,
            init_state_dict=lab1_ckpt["model_state"],
            tag="lab2_full_finetune",
        )
        ft_ckpt = torch.load(ft_best, map_location=device)
        ft_model = build_model(cfg, len(lab2_norm.sensor_ids), len(lab2_norm.variable_names)).to(device)
        ft_model.load_state_dict(ft_ckpt["model_state"])
        ft_metrics, ft_sensor_summary, ft_protocol_summary = evaluate_active_protocols(
            ft_model, lab2_norm, lab2_split.masks["test"], phase1_protocols, device, transfer_mode="full_finetune"
        )
        all_metrics.append(ft_metrics)
        all_sensor_summary.append(ft_sensor_summary)
        all_protocol_summary.append(ft_protocol_summary)
        summary_json["phase1_lab2"]["full_finetune"] = {
            "training_meta": ft_meta,
            "results": ft_protocol_summary.to_dict(orient="records"),
        }
        print("=" * 92)
        print("Lab2 full fine-tune evaluation complete")
        print("=" * 92)
        print(ft_protocol_summary.set_index(["protocol_name", "masked_total", "variable"])[["rmse", "mae", "r2"]])

    # ---- Save ----
    if all_metrics:
        metrics_df = pd.concat(all_metrics, ignore_index=True)
        sensor_summary_df = pd.concat(all_sensor_summary, ignore_index=True)
        protocol_summary_df = pd.concat(all_protocol_summary, ignore_index=True)
        metrics_path = out_dir / str(cfg["phase1_metrics_name"])
        sensor_summary_path = out_dir / str(cfg["phase1_sensor_summary_name"])
        protocol_summary_path = out_dir / str(cfg["phase1_protocol_summary_name"])
        summary_path = out_dir / str(cfg["phase1_summary_name"])
        metrics_df.to_csv(metrics_path, index=False)
        sensor_summary_df.to_csv(sensor_summary_path, index=False)
        protocol_summary_df.to_csv(protocol_summary_path, index=False)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_json, f, indent=2)
        print("\nSaved Phase-1 transfer artifacts")
        print(f"  Raw metrics      : {metrics_path}")
        print(f"  Sensor summary   : {sensor_summary_path}")
        print(f"  Protocol summary : {protocol_summary_path}")
        print(f"  Summary JSON     : {summary_path}")


if __name__ == "__main__":
    main()
