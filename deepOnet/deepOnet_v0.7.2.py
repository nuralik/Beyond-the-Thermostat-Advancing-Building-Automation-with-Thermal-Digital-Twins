from __future__ import annotations

"""
DeepONet v0.7.2
================

Unified benchmark pipeline for the ThermoDT project.

What this script does
---------------------
- Trains the final v0.5.3-style DeepONet on Lab 1 from scratch.
- Transfers the Lab 1 checkpoint to Lab 2 via:
    * zero-shot
    * head-only adaptation
    * partial fine-tuning
    * full fine-tuning
- Runs a generated sparse-subset search on Lab 2.
- Selects the top-N subsets per k from a designated source mode.
- Re-evaluates every enabled transfer mode on that same selected benchmark,
  so the resulting CSV tables are directly comparable across scenarios.
- Exports report-ready figures and summary tables.
- Exports example temperature and humidity maps using an all-13 IDW reference field, a sparse-sensor IDW baseline, and direct dense-grid DeepONet queries.

Important scope note
--------------------
This version does NOT compute thermal comfort. It only generates temperature
and humidity field maps. For model panels, the DeepONet is queried directly on
a dense coordinate grid rather than interpolated from sensor-point predictions.
"""

import copy
import itertools
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


PARAMS: Dict[str, Any] = {
    "seed": 20,
    "device": "auto",
    "experiment_name": "lab1_to_lab2_transfer_v0.7.2",
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
                1: (7.21, 0.60), 2: (3.88, 5.88), 3: (5.85, 5.88),
                4: (0.60, 0.60), 5: (1.92, 1.94), 6: (3.88, 3.91),
                7: (1.92, 3.91), 8: (5.85, 3.91), 9: (7.21, 7.22),
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

    "lab1_split": {"block_hours": 6, "test_fraction": 0.10, "val_fraction": 0.10, "seed_offset": 100},
    "lab2_split": {"block_hours": 6, "test_fraction": 0.10, "adapt_train_fraction": 0.16, "adapt_val_fraction": 0.04, "seed_offset": 200},

    "include_observed_mask": True,
    "mask_fill_value": 0.0,
    "train_extra_mask_min": 1,
    "train_extra_mask_max": 12,

    "hidden": 128,
    "latent_dim": 128,
    "depth": 6,
    "activation": "tanh",

    "train_lab1_from_scratch": True,
    "reuse_existing_checkpoints": True,
    "batch_size": 4096,
    "epochs": 150,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "num_workers": 0,
    "early_stopping_patience": 20,
    "early_stopping_min_delta": 1e-4,

    "run_lab1_reference": True,
    "run_lab2_zero_shot": True,
    "run_lab2_head_only": True,
    "run_lab2_partial_finetune": True,
    "run_lab2_full_finetune": True,

    "head_only_lr": 2e-4,
    "head_only_epochs": 100,
    "head_only_patience": 20,
    "head_only_min_delta": 1e-4,

    "partial_finetune_lr": 2e-4,
    "partial_finetune_epochs": 100,
    "partial_finetune_patience": 20,
    "partial_finetune_min_delta": 1e-4,
    "partial_unfreeze_linear_layers": 2,

    "full_finetune_lr": 2e-4,
    "full_finetune_epochs": 100,
    "full_finetune_patience": 15,
    "full_finetune_min_delta": 1e-4,

    "run_manual_lab2_protocol_eval": False,
    "lab2_manual_protocols": [
        {"protocol_name": "lab2_all13", "display_name": "Lab2 all 13 active", "active_sensors": list(range(1, 14))},
        {"protocol_name": "lab2_active_11_4_6_9_1", "display_name": "Lab2 active set [11, 4, 6, 9, 1]", "active_sensors": [11, 4, 6, 9, 1]},
        {"protocol_name": "lab2_active_2_5_10", "display_name": "Lab2 active set [2, 5, 10]", "active_sensors": [2, 5, 10]},
    ],

    "run_subset_search": True,
    "subset_search_source_mode": "head_only",
    "subset_search_enable_k1": True,
    "subset_search_enable_k2": True,
    "subset_search_enable_k3": True,
    "subset_search_enable_k5": True,
    "subset_search_active_counts": [5, 3, 2, 1],
    "subset_search_generation_mode": {5: "sampled", 3: "exhaustive", 2: "exhaustive", 1: "exhaustive"},
    "subset_search_sample_limit": {5: 20, 3: 150, 2: None, 1: None},
    "subset_search_sample_seed_offset": 700,

    "run_unified_benchmark": True,
    "top_n_per_k": 5,
    "selected_active_counts": [5, 3, 2, 1],
    "include_lab1_reference_in_plots": True,
    "include_partial_finetune_in_plots": True,

    "export_subset_distribution_analysis": True,
    "export_subset_r2_boxplots": True,
    "export_subset_gap_table": True,

    "export_field_maps": True,
    "field_map_idw_power": 2.0,
    "field_map_grid_short_side": 160,
    "field_map_grid_long_side": 220,
    "field_map_grid_square": 180,
    "field_map_active_count": 3,
    "field_map_rank_within_k": 1,
    "field_map_eval_index": "middle",
    "field_map_include_sparse_idw_baseline": True,

    "output_dir": "./deepOnet/runs/lab1_to_lab2_v0.7.2",
    "best_checkpoint_name": "thermodt_lab1_best.pt",
    "final_checkpoint_name": "thermodt_lab1_final.pt",
    "head_best_checkpoint_name": "thermodt_lab2_head_best.pt",
    "head_final_checkpoint_name": "thermodt_lab2_head_final.pt",
    "partial_best_checkpoint_name": "thermodt_lab2_partial_best.pt",
    "partial_final_checkpoint_name": "thermodt_lab2_partial_final.pt",
    "full_ft_best_checkpoint_name": "thermodt_lab2_fullft_best.pt",
    "full_ft_final_checkpoint_name": "thermodt_lab2_fullft_final.pt",

    "lab1_eval_metrics_name": "thermodt_lab1_reference_metrics_v072.csv",
    "manual_lab2_metrics_name": "thermodt_lab2_manual_metrics_v072.csv",
    "manual_lab2_sensor_summary_name": "thermodt_lab2_manual_sensor_summary_v072.csv",
    "manual_lab2_protocol_summary_name": "thermodt_lab2_manual_protocol_summary_v072.csv",
    "subset_search_metrics_name": "thermodt_subset_search_metrics_v072.csv",
    "subset_search_sensor_summary_name": "thermodt_subset_search_sensor_summary_v072.csv",
    "subset_search_protocol_summary_name": "thermodt_subset_search_protocol_summary_v072.csv",
    "subset_search_best_subsets_name": "thermodt_subset_search_best_subsets_v072.csv",
    "subset_distribution_summary_name": "thermodt_subset_distribution_summary_v072.csv",
    "subset_gap_summary_name": "thermodt_subset_gap_summary_v072.csv",
    "subset_r2_boxplot_temperature_name": "subset_r2_distribution_temperature_v072.png",
    "subset_r2_boxplot_humidity_name": "subset_r2_distribution_humidity_v072.png",
    "selected_protocols_name": "thermodt_selected_protocols_v072.csv",
    "unified_metrics_name": "thermodt_unified_metrics_v072.csv",
    "unified_sensor_summary_name": "thermodt_unified_sensor_summary_v072.csv",
    "unified_protocol_summary_name": "thermodt_unified_protocol_summary_v072.csv",
    "benchmark_regime_summary_name": "thermodt_benchmark_regime_summary_v072.csv",
    "benchmark_regime_pivot_r2_name": "thermodt_benchmark_regime_pivot_r2_v072.csv",
    "training_runs_summary_name": "thermodt_training_runs_summary_v072.csv",
    "summary_json_name": "thermodt_summary_v072.json",
    "summary_md_name": "thermodt_summary_v072.md",
    "overview_figure_name": "transfer_overview_topk_v072.png",
    "sensor_map_temperature_name": "transfer_sensor_map_temperature_v072.png",
    "sensor_map_humidity_name": "transfer_sensor_map_humidity_v072.png",
    "subset_compare_k5_name": "transfer_subset_compare_k5_v072.png",
    "subset_compare_k3_name": "transfer_subset_compare_k3_v072.png",
    "subset_compare_k2_name": "transfer_subset_compare_k2_v072.png",
    "subset_compare_k1_name": "transfer_subset_compare_k1_v072.png",
    "field_map_temperature_name": "field_map_temperature_example_v072.png",
    "field_map_humidity_name": "field_map_humidity_example_v072.png",
}



@dataclass
class RawLabData:
    lab_name: str
    data_raw: np.ndarray
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


@dataclass(frozen=True)
class TransferProtocol:
    protocol_name: str
    display_name: str
    active_sensors: Tuple[int, ...]


@dataclass(frozen=True)
class SelectedProtocol:
    protocol_name: str
    display_name: str
    active_sensors: Tuple[int, ...]
    active_count: int
    rank_within_k: int | None
    source_transfer_mode: str | None
    source_score: float | None
    source_r2_temperature: float | None
    source_r2_humidity: float | None


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


def choose_block_starts_by_fraction(time_idx: pd.DatetimeIndex, occupied_mask: np.ndarray, target_fraction: float, block_hours: int, seed: int) -> List[int]:
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
    anchors = np.linspace(0, len(candidate_starts) - 1, num=min(max(target_windows * 4, 1), len(candidate_starts)), dtype=int)
    ordered = [candidate_starts[i] for i in anchors]
    ordered.extend(candidate_starts)
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
        mask[start:min(n, start + window_steps)] = True
    return mask


def starts_to_meta(time_idx: pd.DatetimeIndex, starts: Sequence[int], window_steps: int) -> List[Dict[str, Any]]:
    meta: List[Dict[str, Any]] = []
    for start in starts:
        end = min(len(time_idx), start + window_steps)
        ts0 = pd.Timestamp(time_idx[start])
        ts1 = pd.Timestamp(time_idx[end - 1]) if end > start else ts0
        meta.append({"start_idx": int(start), "end_idx_exclusive": int(end), "start": str(ts0), "end": str(ts1), "start_hour": int(ts0.hour)})
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
    return SplitMasks({"train": train_mask, "val": val_mask, "test": test_mask}, meta)


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
    return SplitMasks({"adapt_train": adapt_train_mask, "adapt_val": adapt_val_mask, "test": test_mask, "unused": unused_mask}, meta)


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

    data_raw = np.stack(variable_arrays, axis=-1)
    valid_mask = ~np.isnan(data_raw).any(axis=(1, 2))
    data_raw = data_raw[valid_mask]
    time_idx = pd.DatetimeIndex(np.asarray(time_idx)[valid_mask])
    sensor_coords = {int(k): tuple(v) for k, v in lab_cfg["sensor_coords"].items()}
    room_width = float(lab_cfg["room_width"])
    room_height = float(lab_cfg["room_height"])
    coords_norm = np.array([(sensor_coords[sid][0] / room_width, sensor_coords[sid][1] / room_height) for sid in sensor_ids], dtype=np.float32)
    return RawLabData(lab_name, data_raw.astype(np.float32), time_idx, coords_norm, sensor_ids, variable_names, variable_units, room_width, room_height, sensor_coords)


def normalize_raw_lab_data(raw: RawLabData, mean: np.ndarray, std: np.ndarray) -> NormalizedLabData:
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
    std = np.where(np.abs(std) < 1e-8, 1.0, std).astype(np.float32)
    data_norm = (raw.data_raw - mean) / std
    return NormalizedLabData(raw.lab_name, data_norm.astype(np.float32), raw.time_idx, raw.coords_norm.astype(np.float32), list(raw.sensor_ids), list(raw.variable_names), dict(raw.variable_units), mean.reshape(-1).astype(np.float32), std.reshape(-1).astype(np.float32), raw.room_width, raw.room_height, dict(raw.sensor_coords))


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
        extra_count = int(rng.integers(self.extra_mask_min, self.extra_mask_max + 1)) if self.extra_mask_max >= self.extra_mask_min else 0
        extra_count = max(0, min(extra_count, self.n_sensors - 1))
        mask_ids = [s]
        if extra_count > 0:
            others = np.array([i for i in range(self.n_sensors) if i != s], dtype=int)
            mask_ids.extend(rng.choice(others, size=extra_count, replace=False).tolist())
        values[mask_ids, :] = self.mask_fill_value
        observed[mask_ids] = 0.0
        branch_parts = [values.reshape(-1)]
        if self.include_observed_mask:
            branch_parts.append(observed)
        branch_input = np.concatenate(branch_parts, axis=0).astype(np.float32)
        target = self.data_norm[t, s, :].astype(np.float32)
        return Batch(torch.from_numpy(branch_input), torch.from_numpy(self.coords_norm[s]), torch.from_numpy(target))


def collate_batches(items: Sequence[Batch]) -> Batch:
    return Batch(torch.stack([x.branch_input for x in items], dim=0), torch.stack([x.trunk_coord for x in items], dim=0), torch.stack([x.target for x in items], dim=0))


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
    branch_in_dim = n_sensors * n_channels + (n_sensors if cfg["include_observed_mask"] else 0)
    return MultiOutputDeepONet(branch_in_dim, 2, n_channels, int(cfg["hidden"]), int(cfg["latent_dim"]), int(cfg["depth"]), str(cfg["activation"]))


def make_loader(data_norm: np.ndarray, coords_norm: np.ndarray, cfg: Mapping[str, Any], seed: int, extra_min: int, extra_max: int, device: torch.device, shuffle: bool) -> DataLoader:
    ds = MultiSensorMaskedDataset(data_norm, coords_norm, float(cfg["mask_fill_value"]), bool(cfg["include_observed_mask"]), int(extra_min), int(extra_max), int(seed))
    return DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=shuffle, num_workers=int(cfg["num_workers"]), pin_memory=(str(device) != "cpu"), collate_fn=collate_batches)


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


def package_checkpoint_payload(cfg: Mapping[str, Any], split_meta: Mapping[str, Any], lab_name: str, normalized: NormalizedLabData, model: nn.Module, epoch: int, train_loss: float, val_loss: float, history: Dict[str, List[float]], extra: Mapping[str, Any] | None = None) -> Dict[str, Any]:
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
        "time": {"n_timesteps": int(len(normalized.time_idx)), "start": str(normalized.time_idx[0]) if len(normalized.time_idx) else None, "end": str(normalized.time_idx[-1]) if len(normalized.time_idx) else None},
    }
    if extra:
        payload.update(copy.deepcopy(dict(extra)))
    return payload


def configure_trainable_params(model: nn.Module, transfer_mode: str | None, partial_unfreeze_linear_layers: int = 2) -> List[str]:
    for p in model.parameters():
        p.requires_grad = True
    if transfer_mode is None or transfer_mode == "full":
        return [name for name, p in model.named_parameters() if p.requires_grad]
    for p in model.parameters():
        p.requires_grad = False
    def _last_linear_layers(net: nn.Sequential, n: int) -> List[nn.Linear]:
        linear_layers = [m for m in net if isinstance(m, nn.Linear)]
        return linear_layers[-max(1, n):]
    if transfer_mode == "head_only":
        for layer in _last_linear_layers(model.branch.net, 1):
            for p in layer.parameters():
                p.requires_grad = True
        for layer in _last_linear_layers(model.trunk.net, 1):
            for p in layer.parameters():
                p.requires_grad = True
        model.bias.requires_grad = True
    elif transfer_mode == "partial":
        n = max(1, int(partial_unfreeze_linear_layers))
        for layer in _last_linear_layers(model.branch.net, n):
            for p in layer.parameters():
                p.requires_grad = True
        for layer in _last_linear_layers(model.trunk.net, n):
            for p in layer.parameters():
                p.requires_grad = True
        model.bias.requires_grad = True
    else:
        raise ValueError(f"Unsupported transfer_mode for trainable params: {transfer_mode}")
    return [name for name, p in model.named_parameters() if p.requires_grad]


def train_model(cfg: Mapping[str, Any], normalized: NormalizedLabData, split_masks: SplitMasks, train_key: str, val_key: str, device: torch.device, out_dir: Path, best_name: str, final_name: str, lr: float, epochs: int, patience: int, min_delta: float, seed: int, init_state_dict: Mapping[str, Any] | None = None, tag: str = "train", transfer_train_mode: str | None = None) -> Tuple[Path, Path, Dict[str, Any]]:
    train_idx = np.where(split_masks.masks[train_key])[0]
    val_idx = np.where(split_masks.masks[val_key])[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(f"Training split is empty for {tag}: train_key={train_key} -> {len(train_idx)} rows, val_key={val_key} -> {len(val_idx)} rows. Adjust the blocked split fractions or block size.")
    train_loader = make_loader(normalized.data_norm[train_idx], normalized.coords_norm, cfg, seed, int(cfg["train_extra_mask_min"]), int(cfg["train_extra_mask_max"]), device, shuffle=True)
    val_loader = make_loader(normalized.data_norm[val_idx], normalized.coords_norm, cfg, seed + 9999, 0, 0, device, shuffle=False)
    model = build_model(cfg, len(normalized.sensor_ids), len(normalized.variable_names)).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
    trainable_names = configure_trainable_params(model, transfer_train_mode, int(cfg.get("partial_unfreeze_linear_layers", 2)))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=float(lr), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))
    criterion = nn.MSELoss()
    best_path = out_dir / best_name
    final_path = out_dir / final_name
    best_val = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}
    stopped_early = False
    stop_epoch = int(epochs)

    print("=" * 96)
    print(f"DeepONet v0.7.2 | {tag} | lab={normalized.lab_name} | device={device}")
    print("=" * 96)
    print(f"Variables         : {normalized.variable_names}")
    print(f"Sensors           : {len(normalized.sensor_ids)}")
    print(f"Timesteps         : {len(normalized.time_idx):,}")
    print(f"Train rows        : {len(train_idx):,}")
    print(f"Val rows          : {len(val_idx):,}")
    print(f"LR / epochs       : {lr:g} / {epochs}")
    print(f"Early stopping    : patience={patience} | min_delta={min_delta:g}")
    print(f"Transfer mode     : {transfer_train_mode or 'full'}")
    print(f"Trainable params  : {len(trainable_names)} tensors")

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch.branch_input.to(device), batch.trunk_coord.to(device))
            target = batch.target.to(device)
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
            torch.save(package_checkpoint_payload(cfg, split_masks.meta, normalized.lab_name, normalized, model, epoch, train_loss, val_loss, history, extra={"tag": tag, "transfer_train_mode": transfer_train_mode or "full", "trainable_param_names": trainable_names, "best_epoch": int(best_epoch), "best_val_loss": float(best_val), "stopped_early": False, "stop_epoch": int(epoch), "epochs_no_improve_final": int(epochs_no_improve)}), best_path)
            marker = " *** best"
        else:
            epochs_no_improve += 1
            marker = f" | no_improve={epochs_no_improve}/{patience}"
        print(f"Epoch {epoch:>4}/{int(epochs)}  train={train_loss:.6f}  val={val_loss:.6f}  lr={current_lr:.2e}{marker}", flush=True)
        if epochs_no_improve >= patience:
            stopped_early = True
            stop_epoch = epoch
            print(f"Early stopping triggered at epoch {epoch} (best epoch={best_epoch}, best val={best_val:.6f}).", flush=True)
            break

    final_payload = package_checkpoint_payload(cfg, split_masks.meta, normalized.lab_name, normalized, model=model, epoch=stop_epoch, train_loss=history["train_loss"][-1], val_loss=history["val_loss"][-1], history=history, extra={"tag": tag, "transfer_train_mode": transfer_train_mode or "full", "trainable_param_names": trainable_names, "best_epoch": int(best_epoch), "best_val_loss": float(best_val), "stopped_early": bool(stopped_early), "stop_epoch": int(stop_epoch), "epochs_no_improve_final": int(epochs_no_improve)})
    torch.save(final_payload, final_path)
    meta = {"run_name": tag, "tag": tag, "lab_name": normalized.lab_name, "transfer_train_mode": transfer_train_mode or "full", "best_epoch": int(best_epoch), "best_val_loss": float(best_val), "stop_epoch": int(stop_epoch), "stopped_early": bool(stopped_early), "n_train_rows": int(len(train_idx)), "n_val_rows": int(len(val_idx)), "n_trainable_tensors": int(len(trainable_names)), "trainable_param_names": trainable_names, "best_checkpoint": str(best_path), "final_checkpoint": str(final_path)}
    print("\nSaved artifacts")
    print(f"  Best checkpoint : {best_path}")
    print(f"  Final checkpoint: {final_path}")
    print(f"  Best epoch      : {best_epoch}")
    print(f"  Training stop   : {'early stop at epoch ' + str(stop_epoch) if stopped_early else 'completed ' + str(stop_epoch) + ' epochs'}")
    return best_path, final_path, meta


def _protocol_name_from_active(active: Sequence[int]) -> str:
    return "lab2_active_" + "_".join(str(int(x)) for x in active)


def _display_name_from_active(active: Sequence[int]) -> str:
    return "Lab2 all 13 active" if len(active) == 13 else "Lab2 active set " + str(list(int(x) for x in active))


def _pretty_subset_label(active: Sequence[int]) -> str:
    return "{" + ", ".join(str(int(x)) for x in active) + "}"


def _parse_active_sensors(protocol_name: str) -> Tuple[int, ...]:
    if protocol_name == "lab2_all13":
        return tuple(range(1, 14))
    m = re.fullmatch(r"lab2_active_(\d+(?:_\d+)*)", str(protocol_name))
    if not m:
        raise ValueError(f"Unsupported protocol_name format: {protocol_name}")
    return tuple(int(x) for x in m.group(1).split("_"))


def build_manual_protocols(cfg: Mapping[str, Any], sensor_ids: Sequence[int]) -> List[TransferProtocol]:
    sensor_set = set(int(s) for s in sensor_ids)
    protocols: List[TransferProtocol] = []
    for entry in cfg["lab2_manual_protocols"]:
        active = tuple(int(x) for x in entry["active_sensors"])
        bad = [x for x in active if x not in sensor_set]
        if bad:
            raise ValueError(f"Manual protocol {entry['protocol_name']} references unknown sensor ids: {bad}")
        protocols.append(TransferProtocol(str(entry["protocol_name"]), str(entry["display_name"]), active))
    return protocols


def build_generated_protocols(cfg: Mapping[str, Any], sensor_ids: Sequence[int]) -> List[TransferProtocol]:
    sensor_ids = [int(s) for s in sensor_ids]
    sensor_set = set(sensor_ids)
    rng = np.random.default_rng(int(cfg["seed"]) + int(cfg.get("subset_search_sample_seed_offset", 700)))
    protocols: List[TransferProtocol] = []
    for k in get_enabled_active_counts(cfg):
        if k <= 0 or k > len(sensor_ids):
            raise ValueError(f"Invalid active-count k={k}")
        combos = [tuple(int(x) for x in c) for c in itertools.combinations(sensor_ids, k)]
        mode = str(cfg.get("subset_search_generation_mode", {}).get(k, "exhaustive")).lower()
        limit = cfg.get("subset_search_sample_limit", {}).get(k, None)
        if mode == "sampled" and limit is not None and int(limit) < len(combos):
            picked_idx = np.sort(rng.choice(len(combos), size=int(limit), replace=False))
            combos = [combos[int(i)] for i in picked_idx]
        elif mode not in ("sampled", "exhaustive"):
            raise ValueError(f"Unsupported generation mode for k={k}: {mode}")
        for active in combos:
            bad = [x for x in active if x not in sensor_set]
            if bad:
                raise ValueError(f"Generated protocol references unknown sensor ids: {bad}")
            protocols.append(TransferProtocol(_protocol_name_from_active(active), _display_name_from_active(active), active))
    return protocols




def get_enabled_active_counts(cfg: Mapping[str, Any]) -> List[int]:
    ordered_counts = [5, 3, 2, 1]
    bool_map = {
        5: bool(cfg.get("subset_search_enable_k5", False)),
        3: bool(cfg.get("subset_search_enable_k3", False)),
        2: bool(cfg.get("subset_search_enable_k2", False)),
        1: bool(cfg.get("subset_search_enable_k1", False)),
    }
    enabled = [k for k in ordered_counts if bool_map.get(k, False)]
    if enabled:
        return enabled
    fallback = [int(x) for x in cfg.get("subset_search_active_counts", [])]
    return sorted(set(fallback), reverse=True)


def _enabled_subset_compare_targets(cfg: Mapping[str, Any]) -> List[int]:
    active_counts = [int(x) for x in cfg.get("selected_active_counts", [])]
    return [k for k in [5, 3, 2, 1] if k in active_counts]


def build_selected_protocols(best_subsets_df: pd.DataFrame, top_n_per_k: int, ks: Sequence[int], source_mode: str) -> List[SelectedProtocol]:
    source = best_subsets_df[best_subsets_df["transfer_mode"] == source_mode].copy()
    if source.empty:
        raise ValueError(f"No {source_mode} rows found in best-subsets table.")
    protocols: List[SelectedProtocol] = [SelectedProtocol("lab2_all13", "Lab2 all 13 active", tuple(range(1, 14)), 13, None, source_mode, None, None, None)]
    for k in [int(x) for x in ks]:
        sub = source[source["active_count"] == k].sort_values(["score", "r2_temperature", "r2_humidity"], ascending=False).head(int(top_n_per_k))
        if sub.empty:
            raise ValueError(f"No {source_mode} subsets found for k={k}")
        for rank, row in enumerate(sub.itertuples(index=False), start=1):
            active = _parse_active_sensors(row.protocol_name)
            protocols.append(SelectedProtocol(str(row.protocol_name), _display_name_from_active(active), active, int(k), rank, source_mode, float(row.score), float(row.r2_temperature), float(row.r2_humidity)))
    return protocols


def selected_protocols_to_df(protocols: Sequence[SelectedProtocol]) -> pd.DataFrame:
    return pd.DataFrame([{
        "protocol_name": p.protocol_name,
        "display_name": p.display_name,
        "active_sensors": list(p.active_sensors),
        "active_count": p.active_count,
        "masked_total": 13 - p.active_count,
        "rank_within_k": p.rank_within_k,
        "source_transfer_mode": p.source_transfer_mode,
        "source_score": p.source_score,
        "source_r2_temperature": p.source_r2_temperature,
        "source_r2_humidity": p.source_r2_humidity,
    } for p in protocols])


def attach_protocol_metadata(df: pd.DataFrame, selected_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or selected_df.empty:
        return df.copy()
    meta_cols = [c for c in ["protocol_name", "active_count", "rank_within_k", "source_transfer_mode", "source_score", "source_r2_temperature", "source_r2_humidity"] if c in selected_df.columns]
    out = df.merge(selected_df[meta_cols], on="protocol_name", how="left")
    if "active_count" in out.columns:
        out["active_total_nominal"] = out["active_count"].fillna(out["active_total_nominal"])
    return out


def _build_branch_input_for_protocol(normalized_row: np.ndarray, sensor_ids: Sequence[int], target_sensor_id: int, active_sensors: Sequence[int], include_observed_mask: bool, mask_fill_value: float) -> np.ndarray:
    values = np.asarray(normalized_row, dtype=np.float32).copy()
    observed = np.ones((len(sensor_ids),), dtype=np.float32)
    removed = [sid for sid in sensor_ids if sid not in active_sensors]
    if target_sensor_id not in removed:
        removed.append(target_sensor_id)
    removed_idx = [sensor_ids.index(sid) for sid in sorted(set(removed))]
    values[removed_idx, :] = float(mask_fill_value)
    observed[removed_idx] = 0.0
    parts = [values.reshape(-1)]
    if include_observed_mask:
        parts.append(observed)
    return np.concatenate(parts, axis=0).astype(np.float32)




def _build_branch_input_for_field_query(normalized_row: np.ndarray, sensor_ids: Sequence[int], active_sensors: Sequence[int], include_observed_mask: bool, mask_fill_value: float) -> np.ndarray:
    values = np.asarray(normalized_row, dtype=np.float32).copy()
    observed = np.ones((len(sensor_ids),), dtype=np.float32)
    removed_idx = [sensor_ids.index(sid) for sid in sensor_ids if sid not in active_sensors]
    if removed_idx:
        values[removed_idx, :] = float(mask_fill_value)
        observed[removed_idx] = 0.0
    parts = [values.reshape(-1)]
    if include_observed_mask:
        parts.append(observed)
    return np.concatenate(parts, axis=0).astype(np.float32)


@torch.no_grad()
def evaluate_active_protocols(model: nn.Module, normalized: NormalizedLabData, eval_mask: np.ndarray, protocols: Sequence[TransferProtocol], device: torch.device, transfer_mode: str, cfg: Mapping[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model.eval()
    eval_idx = np.where(eval_mask)[0]
    eval_data_norm = normalized.data_norm[eval_idx]
    actual_denorm = eval_data_norm * normalized.std.reshape(1, 1, -1) + normalized.mean.reshape(1, 1, -1)
    coords_norm = normalized.coords_norm
    sensor_ids = normalized.sensor_ids
    records: List[Dict[str, Any]] = []
    for protocol in protocols:
        active_sensors = [int(x) for x in protocol.active_sensors]
        preds_denorm = np.zeros_like(actual_denorm)
        for t in range(eval_data_norm.shape[0]):
            for s_idx, sensor_id in enumerate(sensor_ids):
                branch_input = _build_branch_input_for_protocol(eval_data_norm[t], sensor_ids, sensor_id, active_sensors, bool(cfg["include_observed_mask"]), float(cfg["mask_fill_value"]))
                pred = model(torch.from_numpy(branch_input).unsqueeze(0).to(device), torch.from_numpy(coords_norm[s_idx]).unsqueeze(0).to(device)).cpu().numpy()[0]
                preds_denorm[t, s_idx, :] = pred * normalized.std + normalized.mean
        active_count = int(len(active_sensors))
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
                    "lab_name": normalized.lab_name,
                    "transfer_mode": transfer_mode,
                    "protocol_name": protocol.protocol_name,
                    "display_name": protocol.display_name,
                    "active_total_nominal": active_count,
                    "masked_total": int(len(sensor_ids) - active_count),
                    "sensor": int(sensor_id),
                    "sensor_is_active": int(sensor_id in active_sensors),
                    "variable": var_name,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                })
    metrics_df = pd.DataFrame(records)
    sensor_summary_df = metrics_df.groupby(["lab_name", "transfer_mode", "protocol_name", "display_name", "active_total_nominal", "masked_total", "sensor", "sensor_is_active", "variable"], as_index=False)[["rmse", "mae", "r2"]].mean()
    protocol_summary_df = metrics_df.groupby(["lab_name", "transfer_mode", "protocol_name", "display_name", "active_total_nominal", "masked_total", "variable"], as_index=False)[["rmse", "mae", "r2"]].mean()
    return metrics_df, sensor_summary_df, protocol_summary_df


@torch.no_grad()
def predict_sensor_grid_for_time(model: nn.Module, normalized: NormalizedLabData, time_global_index: int, active_sensors: Sequence[int], device: torch.device, cfg: Mapping[str, Any]) -> np.ndarray:
    model.eval()
    row = normalized.data_norm[int(time_global_index)]
    preds = np.zeros((len(normalized.sensor_ids), len(normalized.variable_names)), dtype=np.float32)
    for s_idx, sensor_id in enumerate(normalized.sensor_ids):
        branch_input = _build_branch_input_for_protocol(row, normalized.sensor_ids, sensor_id, active_sensors, bool(cfg["include_observed_mask"]), float(cfg["mask_fill_value"]))
        pred = model(torch.from_numpy(branch_input).unsqueeze(0).to(device), torch.from_numpy(normalized.coords_norm[s_idx]).unsqueeze(0).to(device)).cpu().numpy()[0]
        preds[s_idx] = pred * normalized.std + normalized.mean
    return preds




@torch.no_grad()
def predict_dense_field_for_time(
    model: nn.Module,
    normalized: NormalizedLabData,
    time_global_index: int,
    active_sensors: Sequence[int],
    device: torch.device,
    cfg: Mapping[str, Any],
    grid_nx: int,
    grid_ny: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    room_w = float(normalized.room_width)
    room_h = float(normalized.room_height)
    xs = np.linspace(0.0, room_w, int(grid_nx), dtype=np.float32)
    ys = np.linspace(0.0, room_h, int(grid_ny), dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    row = normalized.data_norm[int(time_global_index)]
    branch_input = _build_branch_input_for_field_query(
        row,
        normalized.sensor_ids,
        active_sensors,
        bool(cfg["include_observed_mask"]),
        float(cfg["mask_fill_value"]),
    )
    branch_batch = np.repeat(branch_input[None, :], xx.size, axis=0)
    coords_norm = np.stack([(xx.reshape(-1) / room_w), (yy.reshape(-1) / room_h)], axis=1).astype(np.float32)
    pred = model(
        torch.from_numpy(branch_batch).to(device),
        torch.from_numpy(coords_norm).to(device),
    ).cpu().numpy()
    pred = pred.reshape(int(grid_ny), int(grid_nx), len(normalized.variable_names))
    pred = pred * normalized.std.reshape(1, 1, -1) + normalized.mean.reshape(1, 1, -1)
    return xx, yy, pred.astype(np.float32)


def _mode_label(mode: str) -> str:
    return {
        "lab1_reference": "LAB1 reference",
        "ground_truth": "All-13 IDW reference",
        "reference_idw_all13": "All-13 IDW reference",
        "sparse_idw_baseline": "Sparse IDW baseline",
        "zero_shot": "LAB2 zero-shot",
        "head_only": "LAB2 head-only",
        "partial_finetune": "LAB2 partial fine-tune",
        "full_finetune": "LAB2 full fine-tune",
    }.get(mode, mode)


def _mode_color(mode: str) -> str:
    return {
        "lab1_reference": "#1f77b4",
        "zero_shot": "#ff7f0e",
        "head_only": "#9467bd",
        "partial_finetune": "#d62728",
        "full_finetune": "#2ca02c",
    }.get(mode, "#7f7f7f")


def get_plot_mode_order(cfg: Mapping[str, Any], available_modes: Iterable[str]) -> List[str]:
    available = set(available_modes)
    order: List[str] = []
    if bool(cfg.get("include_lab1_reference_in_plots", True)) and "lab1_reference" in available:
        order.append("lab1_reference")
    for mode in ["zero_shot", "head_only", "partial_finetune", "full_finetune"]:
        if mode == "partial_finetune" and not bool(cfg.get("include_partial_finetune_in_plots", True)):
            continue
        if mode in available:
            order.append(mode)
    return order


def build_regime_summary(protocol_summary: pd.DataFrame, selected_df: pd.DataFrame, cfg: Mapping[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if protocol_summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    regimes = [(13, "all13")] + [(int(k), f"top{int(cfg['top_n_per_k'])}_k{int(k)}") for k in cfg["selected_active_counts"]]
    for variable in ["temperature", "humidity"]:
        for mode in sorted(protocol_summary["transfer_mode"].unique()):
            sub = protocol_summary[(protocol_summary["transfer_mode"] == mode) & (protocol_summary["variable"] == variable)]
            for active_count, regime in regimes:
                g = sub[sub["active_total_nominal"] == active_count]
                if g.empty:
                    continue
                rows.append({"transfer_mode": mode, "variable": variable, "regime": regime, "active_count": int(active_count), "n_protocols": int(g["protocol_name"].nunique()), "mean_r2": float(g["r2"].mean()), "mean_rmse": float(g["rmse"].mean()), "mean_mae": float(g["mae"].mean())})
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df, pd.DataFrame()
    pivot = long_df.pivot_table(index="transfer_mode", columns=["variable", "regime"], values="mean_r2").reset_index()
    pivot.columns = ["transfer_mode" if c == "transfer_mode" else f"{c[0]}__{c[1]}" for c in pivot.columns]
    return long_df, pivot




def build_subset_distribution_tables(protocol_summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if protocol_summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    gap_rows: List[Dict[str, Any]] = []
    metrics = ["r2", "rmse", "mae"]
    for (transfer_mode, active_count, variable), g in protocol_summary.groupby(["transfer_mode", "active_total_nominal", "variable"], sort=True):
        for metric in metrics:
            vals = g[metric].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            q25 = float(np.quantile(vals, 0.25))
            median = float(np.quantile(vals, 0.50))
            q75 = float(np.quantile(vals, 0.75))
            mean = float(np.mean(vals))
            min_v = float(np.min(vals))
            max_v = float(np.max(vals))
            std = float(np.std(vals))
            iqr = float(q75 - q25)
            rows.append({
                "transfer_mode": str(transfer_mode),
                "active_count": int(active_count),
                "variable": str(variable),
                "metric": metric,
                "n_protocols": int(vals.size),
                "min": min_v,
                "q25": q25,
                "median": median,
                "mean": mean,
                "q75": q75,
                "max": max_v,
                "std": std,
                "iqr": iqr,
            })
            if metric == "r2":
                best_value = max_v
                worst_value = min_v
                better_direction = "higher_is_better"
                absolute_gap = float(best_value - worst_value)
            else:
                best_value = min_v
                worst_value = max_v
                better_direction = "lower_is_better"
                absolute_gap = float(worst_value - best_value)
            gap_rows.append({
                "transfer_mode": str(transfer_mode),
                "active_count": int(active_count),
                "variable": str(variable),
                "metric": metric,
                "n_protocols": int(vals.size),
                "best_value": best_value,
                "worst_value": worst_value,
                "absolute_gap": absolute_gap,
                "better_direction": better_direction,
            })
    summary_df = pd.DataFrame(rows).sort_values(["transfer_mode", "active_count", "variable", "metric"]).reset_index(drop=True)
    gap_df = pd.DataFrame(gap_rows).sort_values(["transfer_mode", "active_count", "variable", "metric"]).reset_index(drop=True)
    return summary_df, gap_df


def _plot_subset_r2_boxplot(protocol_summary: pd.DataFrame, variable: str, outfile: Path, cfg: Mapping[str, Any]) -> None:
    sub = protocol_summary[protocol_summary["variable"] == variable].copy()
    if sub.empty:
        return
    source_mode = str(cfg["subset_search_source_mode"])
    sub = sub[sub["transfer_mode"] == source_mode].copy()
    ks = [int(k) for k in get_enabled_active_counts(cfg)]
    data = []
    labels = []
    for k in ks:
        vals = sub[sub["active_total_nominal"] == k]["r2"].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        data.append(vals)
        labels.append(f"k={k}")
    if not data:
        return
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#dbeafe")
        patch.set_edgecolor("#1f2937")
        patch.set_linewidth(1.2)
    for item in bp["whiskers"] + bp["caps"] + bp["medians"]:
        item.set_color("#1f2937")
        item.set_linewidth(1.2)
    for mean_item in bp["means"]:
        mean_item.set_marker("D")
        mean_item.set_markerfacecolor("#dc2626")
        mean_item.set_markeredgecolor("#7f1d1d")
        mean_item.set_markersize(6)
    ax.set_title(f"{variable.capitalize()} subset-search R² distribution by active sensor count", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Active sensor count")
    ax.set_ylabel("Protocol-level R²")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(min(0.0, float(sub["r2"].min()) - 0.02), min(1.02, float(sub["r2"].max()) + 0.02))
    source_mode_label = _mode_label(source_mode)
    ax.text(0.99, 0.02, f"Subset search source mode: {source_mode_label}", transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="#374151")
    fig.tight_layout()
    fig.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _resolve_field_map_grid_shape(room_w: float, room_h: float, cfg: Mapping[str, Any]) -> Tuple[int, int]:
    short_side = int(cfg.get("field_map_grid_short_side", 160))
    long_side = int(cfg.get("field_map_grid_long_side", 220))
    square = int(cfg.get("field_map_grid_square", 180))

    if room_w <= 0 or room_h <= 0:
        return short_side, long_side
    if abs(room_w - room_h) / max(room_w, room_h) < 0.05:
        return square, square
    if room_w >= room_h:
        return long_side, short_side
    return short_side, long_side


def _idw_surface(coords_by_sensor: Mapping[int, Tuple[float, float]], values_by_sensor: Mapping[int, float], room_w: float, room_h: float, grid_nx: int = 220, grid_ny: int = 220, power: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sensor_ids = sorted(values_by_sensor.keys())
    pts = np.array([coords_by_sensor[s] for s in sensor_ids], dtype=float)
    vals = np.array([values_by_sensor[s] for s in sensor_ids], dtype=float)
    xs = np.linspace(0.0, float(room_w), int(grid_nx))
    ys = np.linspace(0.0, float(room_h), int(grid_ny))
    xx, yy = np.meshgrid(xs, ys)
    dx = xx[..., None] - pts[:, 0][None, None, :]
    dy = yy[..., None] - pts[:, 1][None, None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    exact = dist < 1e-12
    weights = 1.0 / np.maximum(dist, 1e-6) ** power
    z = (weights * vals[None, None, :]).sum(axis=2) / weights.sum(axis=2)
    if exact.any():
        hit_rows, hit_cols, hit_sensor_idx = np.where(exact)
        z[hit_rows, hit_cols] = vals[hit_sensor_idx]
    return xx, yy, z


def _plot_overview_bars(protocol_summary: pd.DataFrame, outfile: Path, cfg: Mapping[str, Any]) -> None:
    ks = [13] + [int(k) for k in cfg["selected_active_counts"]]
    labels = {13: "All 13 active"}
    for k in cfg["selected_active_counts"]:
        labels[int(k)] = f"Top {int(cfg['top_n_per_k'])} mean (k={int(k)})"
    mode_order = get_plot_mode_order(cfg, protocol_summary["transfer_mode"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.8), sharey=False)
    width = 0.80 / max(1, len(mode_order))
    x = np.arange(len(ks), dtype=float)
    for ax, variable in zip(axes, ["temperature", "humidity"]):
        sub = protocol_summary[protocol_summary["variable"] == variable].copy()
        for i, mode in enumerate(mode_order):
            vals = []
            for k in ks:
                g = sub[(sub["transfer_mode"] == mode) & (sub["active_total_nominal"] == k)]
                vals.append(float(g["r2"].mean()) if not g.empty else float("nan"))
            offs = (i - (len(mode_order) - 1) / 2.0) * width
            bars = ax.bar(x + offs, vals, width=width, label=_mode_label(mode), color=_mode_color(mode), alpha=0.95)
            for bar, val in zip(bars, vals):
                if math.isfinite(val):
                    ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.004, f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[k] for k in ks], fontsize=10)
        ax.set_ylabel("R²")
        ax.set_title(f"{variable.capitalize()} transfer summary", fontsize=15, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        vals_present = sub["r2"].dropna().to_numpy(dtype=float)
        ymin = max(0.0, float(vals_present.min()) - 0.05) if len(vals_present) else 0.0
        ymax = min(1.01, float(vals_present.max()) + 0.04) if len(vals_present) else 1.0
        ax.set_ylim(ymin, ymax)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", bbox_to_anchor=(0.5, 0.01), ncol=max(1, len(mode_order)), frameon=False, fontsize=11)
    fig.suptitle("Unified comparable LAB1/LAB2 transfer benchmark", fontsize=20, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.18, wspace=0.14)
    fig.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_sensor_map_panel_grid(sensor_df: pd.DataFrame, variable: str, outfile: Path, cfg: Mapping[str, Any]) -> None:
    mode_order = get_plot_mode_order(cfg, sensor_df["transfer_mode"].unique())
    frames = []
    for mode in mode_order:
        sub = sensor_df[(sensor_df["transfer_mode"] == mode) & (sensor_df["protocol_name"] == "lab2_all13") & (sensor_df["variable"] == variable)].copy()
        if not sub.empty:
            frames.append(sub)
    if not frames:
        return
    vmin = min(float(df["r2"].min()) for df in frames)
    vmax = max(float(df["r2"].max()) for df in frames)
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-3
    ncols = len(mode_order)
    fig = plt.figure(figsize=(4.6 * ncols + 0.8, 6.6), constrained_layout=False)
    gs = fig.add_gridspec(1, ncols + 1, width_ratios=[1] * ncols + [0.055], wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
    cax = fig.add_subplot(gs[0, ncols])
    cmap = plt.get_cmap("RdYlGn")
    im = None
    for ax, mode in zip(axes, mode_order):
        sub = sensor_df[(sensor_df["transfer_mode"] == mode) & (sensor_df["protocol_name"] == "lab2_all13") & (sensor_df["variable"] == variable)].copy()
        lab_name = "lab1" if mode == "lab1_reference" else "lab2"
        room_w = float(cfg["labs"][lab_name]["room_width"])
        room_h = float(cfg["labs"][lab_name]["room_height"])
        coords = {int(k): tuple(v) for k, v in cfg["labs"][lab_name]["sensor_coords"].items()}
        ax.set_title(f"{_mode_label(mode)}\n(all 13 active)", fontsize=13.5, fontweight="bold", pad=10)
        ax.set_xlim(0, room_w)
        ax.set_ylim(0, room_h)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.15)
        values_by_sensor = {int(row.sensor): float(row.r2) for row in sub.itertuples(index=False)}
        xx, yy, zz = _idw_surface(coords, values_by_sensor, room_w, room_h, 220, 220, 2.0)
        im = ax.imshow(zz, origin="lower", extent=(0, room_w, 0, room_h), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="bilinear", zorder=0)
        for sid, (x, y) in coords.items():
            if sid not in values_by_sensor:
                continue
            val = values_by_sensor[sid]
            ax.scatter([x], [y], s=220, facecolor="none", edgecolor="black", linewidth=1.1, zorder=3)
            ax.text(x + 0.02 * room_w, y + 0.02 * room_h, f"{sid}\n{val:.3f}", fontsize=9.5, color="black", zorder=4, bbox=dict(boxstyle="round,pad=0.18", facecolor=(1, 1, 1, 0.25), edgecolor="none"))
    if im is not None:
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Sensor R²", fontsize=13)
        cb.ax.tick_params(labelsize=10)
    fig.suptitle(f"Per-sensor transfer map under the all-13 regime ({variable.capitalize()})", fontsize=20, fontweight="bold", y=0.97)
    fig.subplots_adjust(left=0.05, right=0.965, bottom=0.10, top=0.87, wspace=0.18)
    fig.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_subset_compare(protocol_summary: pd.DataFrame, active_count: int, outfile: Path, cfg: Mapping[str, Any]) -> None:
    sub = protocol_summary[(protocol_summary["active_total_nominal"] == int(active_count)) & (protocol_summary["protocol_name"] != "lab2_all13")].copy()
    if sub.empty:
        return
    source_mode = str(cfg["subset_search_source_mode"])
    order = sub[sub["transfer_mode"] == source_mode].groupby("protocol_name")["r2"].mean().sort_values(ascending=False).index.tolist()
    labels = {pn: _pretty_subset_label(_parse_active_sensors(pn)) for pn in order}
    mode_order = get_plot_mode_order(cfg, sub["transfer_mode"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 9.0), sharex=True)
    width = 0.80 / max(1, len(mode_order))
    x = np.arange(len(order), dtype=float)
    for ax, variable in zip(axes, ["temperature", "humidity"]):
        var_df = sub[sub["variable"] == variable].copy()
        for i, mode in enumerate(mode_order):
            vals = []
            for pn in order:
                g = var_df[(var_df["transfer_mode"] == mode) & (var_df["protocol_name"] == pn)]
                vals.append(float(g["r2"].mean()) if not g.empty else float("nan"))
            offs = (i - (len(mode_order) - 1) / 2.0) * width
            bars = ax.bar(x + offs, vals, width=width, label=_mode_label(mode), color=_mode_color(mode), alpha=0.95)
            for bar, val in zip(bars, vals):
                if math.isfinite(val):
                    ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.004, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("R²")
        ax.set_title(f"{variable.capitalize()} — selected subsets with k={active_count}", fontsize=14.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        vals_present = var_df["r2"].dropna().to_numpy(dtype=float)
        ymin = max(0.0, float(vals_present.min()) - 0.05) if len(vals_present) else 0.0
        ymax = min(1.01, float(vals_present.max()) + 0.04) if len(vals_present) else 1.0
        ax.set_ylim(ymin, ymax)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([labels[pn] for pn in order], fontsize=10)
    axes[-1].set_xlabel("Selected sparse subset")
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", bbox_to_anchor=(0.5, 0.01), ncol=max(1, len(mode_order)), frameon=False, fontsize=11)
    fig.suptitle(f"Unified subset comparison for k={active_count}", fontsize=20, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.17, hspace=0.30)
    fig.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _resolve_eval_index(eval_mask: np.ndarray, selector: str | int) -> Tuple[int, int]:
    eval_idx = np.where(eval_mask)[0]
    if len(eval_idx) == 0:
        raise ValueError("Evaluation mask has no rows.")
    if isinstance(selector, int):
        local_idx = int(max(0, min(selector, len(eval_idx) - 1)))
    else:
        selector = str(selector).lower()
        local_idx = 0 if selector == "first" else len(eval_idx) - 1 if selector == "last" else len(eval_idx) // 2
    return int(eval_idx[local_idx]), int(local_idx)


def _plot_example_field_map(normalized: NormalizedLabData, eval_mask: np.ndarray, selected_df: pd.DataFrame, models_by_mode: Mapping[str, nn.Module], device: torch.device, cfg: Mapping[str, Any], variable: str, outfile: Path) -> Dict[str, Any]:
    target_rows = selected_df[
        (selected_df["active_count"] == int(cfg["field_map_active_count"]))
        & (selected_df["rank_within_k"] == int(cfg["field_map_rank_within_k"]))
    ]
    if target_rows.empty:
        fallback = selected_df[selected_df["rank_within_k"].notna()].copy()
        fallback = fallback.sort_values(["active_count", "rank_within_k"], ascending=[False, True])
        if fallback.empty:
            raise ValueError(
                f"No selected protocol found for field-map export. Check field-map settings and enabled active-count booleans."
            )
        row = fallback.iloc[0]
    else:
        row = target_rows.iloc[0]
    var_idx = normalized.variable_names.index(variable)
    active_sensors = tuple(int(x) for x in row["active_sensors"])
    global_idx, local_idx = _resolve_eval_index(eval_mask, cfg["field_map_eval_index"])
    actual = normalized.data_norm[global_idx] * normalized.std.reshape(1, -1) + normalized.mean.reshape(1, -1)
    room_w = normalized.room_width
    room_h = normalized.room_height
    coords = normalized.sensor_coords
    grid_nx, grid_ny = _resolve_field_map_grid_shape(room_w, room_h, cfg)
    power = float(cfg["field_map_idw_power"])

    panels: List[Tuple[str, str, np.ndarray]] = []
    ref_values = {sid: float(actual[i, var_idx]) for i, sid in enumerate(normalized.sensor_ids)}
    _, _, ref_surface = _idw_surface(coords, ref_values, room_w, room_h, grid_nx, grid_ny, power)
    panels.append(("reference_idw_all13", "All-13 IDW reference", ref_surface))

    if bool(cfg.get("field_map_include_sparse_idw_baseline", True)):
        sparse_values = {sid: float(actual[normalized.sensor_ids.index(sid), var_idx]) for sid in active_sensors}
        _, _, sparse_surface = _idw_surface(coords, sparse_values, room_w, room_h, grid_nx, grid_ny, power)
        panels.append(("sparse_idw_baseline", f"Sparse IDW baseline ({len(active_sensors)} active)", sparse_surface))

    for mode, model in models_by_mode.items():
        _, _, pred_grid = predict_dense_field_for_time(model, normalized, global_idx, active_sensors, device, cfg, grid_nx, grid_ny)
        panels.append((mode, f"{_mode_label(mode)} (dense query)", pred_grid[:, :, var_idx]))

    vmin = min(float(np.nanmin(surface)) for _, _, surface in panels)
    vmax = max(float(np.nanmax(surface)) for _, _, surface in panels)

    ncols = len(panels)
    fig = plt.figure(figsize=(4.6 * ncols + 0.8, 6.5), constrained_layout=False)
    gs = fig.add_gridspec(1, ncols + 1, width_ratios=[1] * ncols + [0.055], wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
    cax = fig.add_subplot(gs[0, ncols])
    cmap = plt.get_cmap("coolwarm")
    im = None

    sensor_value_lookup = {sid: float(actual[i, var_idx]) for i, sid in enumerate(normalized.sensor_ids)}
    for ax, (mode, panel_title, surface) in zip(axes, panels):
        im = ax.imshow(
            surface,
            origin="lower",
            extent=(0, room_w, 0, room_h),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="bilinear",
            zorder=0,
        )
        ax.set_title(panel_title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlim(0, room_w)
        ax.set_ylim(0, room_h)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.15)

        for sid, (x, y) in coords.items():
            if mode == "reference_idw_all13":
                marker_face = "white"
            else:
                marker_face = "black" if sid in active_sensors else "none"
            ax.scatter([x], [y], s=110, facecolor=marker_face, edgecolor="black", linewidth=1.0, zorder=3)
            ax.text(
                x + 0.02 * room_w,
                y + 0.02 * room_h,
                f"{sid}\\n{sensor_value_lookup[sid]:.2f}",
                fontsize=8.5,
                color="black",
                zorder=4,
                bbox=dict(boxstyle="round,pad=0.16", facecolor=(1, 1, 1, 0.35), edgecolor="none"),
            )

    if im is not None:
        cb = fig.colorbar(im, cax=cax)
        unit = normalized.variable_units.get(variable, "")
        cb.set_label(f"{variable.capitalize()} ({unit})" if unit else variable.capitalize(), fontsize=13)
        cb.ax.tick_params(labelsize=10)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", color="black", markerfacecolor="black", markersize=8, label="Active sensor"),
        plt.Line2D([0], [0], marker="o", linestyle="None", color="black", markerfacecolor="none", markersize=8, label="Inactive sensor"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=11)
    ts = str(normalized.time_idx[global_idx])
    fig.suptitle(
        f"Lab2 {variable.capitalize()} field map example — protocol {_pretty_subset_label(active_sensors)} — time {ts}",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.05, right=0.965, top=0.87, bottom=0.16, wspace=0.18)
    fig.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "protocol_name": str(row["protocol_name"]),
        "active_sensors": list(active_sensors),
        "active_count": int(len(active_sensors)),
        "time_global_index": int(global_idx),
        "time_eval_local_index": int(local_idx),
        "timestamp": ts,
        "variable": variable,
        "outfile": str(outfile),
    }



def _build_summary_markdown(selected_df: pd.DataFrame, regime_summary: pd.DataFrame, training_summary: pd.DataFrame, output_path: Path, cfg: Mapping[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Unified transfer benchmark (v0.7.2)")
    lines.append("")
    lines.append("This summary was generated by one single v0.7.2 pipeline.")
    lines.append("")
    lines.append("## Selected benchmark")
    lines.append("")
    lines.append("- Lab2 all 13 active")
    for k in [int(x) for x in cfg["selected_active_counts"]]:
        lines.append(f"- Top {int(cfg['top_n_per_k'])} subsets for k={k}, selected from {cfg['subset_search_source_mode']}")
    lines.append("")
    if not regime_summary.empty:
        lines.append("## Mean R² by regime")
        lines.append("")
        for variable in ["temperature", "humidity"]:
            var_df = regime_summary[regime_summary["variable"] == variable].copy()
            if var_df.empty:
                continue
            pivot = var_df.pivot_table(index="transfer_mode", columns="regime", values="mean_r2")
            cols = [c for c in ["all13"] + [f"top{int(cfg['top_n_per_k'])}_k{int(k)}" for k in cfg["selected_active_counts"]] if c in pivot.columns]
            lines.append(f"### {variable.capitalize()}")
            lines.append("")
            lines.append("| Mode | " + " | ".join(cols) + " |")
            lines.append("|---|" + "|".join(["---:"] * len(cols)) + "|")
            for mode in get_plot_mode_order(cfg, pivot.index.tolist()):
                vals = [pivot.loc[mode, c] if (mode in pivot.index and c in pivot.columns) else float("nan") for c in cols]
                fmt = lambda x: (f"{x:.3f}" if math.isfinite(x) else "—")
                lines.append(f"| {_mode_label(mode)} | " + " | ".join(fmt(v) for v in vals) + " |")
            lines.append("")
    if not selected_df.empty:
        lines.append("## Selected subsets")
        lines.append("")
        lines.append("| k | Rank | Protocol | Sensors | Source mode | Source score |")
        lines.append("|---:|---:|---|---|---|---:|")
        for row in selected_df[selected_df["protocol_name"] != "lab2_all13"].sort_values(["active_count", "rank_within_k"], ascending=[False, True]).itertuples(index=False):
            lines.append(f"| {row.active_count} | {row.rank_within_k} | {row.protocol_name} | {_pretty_subset_label(row.active_sensors)} | {row.source_transfer_mode} | {row.source_score:.3f} |")
        lines.append("")
    if not training_summary.empty:
        lines.append("## Training runs")
        lines.append("")
        lines.append("| Run | Transfer mode | Best epoch | Best val loss | Stopped early |")
        lines.append("|---|---|---:|---:|---|")
        for row in training_summary.itertuples(index=False):
            lines.append(f"| {row.run_name} | {row.transfer_train_mode} | {row.best_epoch} | {row.best_val_loss:.6f} | {row.stopped_early} |")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def _checkpoint_to_model(cfg: Mapping[str, Any], normalized: NormalizedLabData, ckpt: Mapping[str, Any], device: torch.device) -> nn.Module:
    model = build_model(cfg, len(normalized.sensor_ids), len(normalized.variable_names)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _ckpt_meta(ckpt: Mapping[str, Any], run_name: str, transfer_train_mode: str, best_path: Path, final_path: Path) -> Dict[str, Any]:
    return {
        "run_name": run_name,
        "tag": str(ckpt.get("tag", run_name)),
        "lab_name": str(ckpt.get("lab_name", "unknown")),
        "transfer_train_mode": transfer_train_mode,
        "best_epoch": int(ckpt.get("best_epoch", ckpt.get("epoch", 0))),
        "best_val_loss": float(ckpt.get("best_val_loss", ckpt.get("val_loss", float("nan")))),
        "stop_epoch": int(ckpt.get("stop_epoch", ckpt.get("epoch", 0))),
        "stopped_early": bool(ckpt.get("stopped_early", False)),
        "n_train_rows": int(ckpt.get("splits", {}).get("n_train", ckpt.get("splits", {}).get("n_adapt_train", 0))),
        "n_val_rows": int(ckpt.get("splits", {}).get("n_val", ckpt.get("splits", {}).get("n_adapt_val", 0))),
        "n_trainable_tensors": int(len(ckpt.get("trainable_param_names", []))),
        "trainable_param_names": ckpt.get("trainable_param_names", []),
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
    }


def _validate_config(cfg: Mapping[str, Any]) -> None:
    source_mode = str(cfg["subset_search_source_mode"])
    enabled = {
        "zero_shot": bool(cfg.get("run_lab2_zero_shot", False)),
        "head_only": bool(cfg.get("run_lab2_head_only", False)),
        "partial_finetune": bool(cfg.get("run_lab2_partial_finetune", False)),
        "full_finetune": bool(cfg.get("run_lab2_full_finetune", False)),
    }
    if bool(cfg.get("run_subset_search", True)):
        if source_mode not in enabled:
            raise ValueError(f"Unsupported subset_search_source_mode: {source_mode}")
        if not enabled[source_mode]:
            raise ValueError(f"subset_search_source_mode={source_mode} requires that mode to be enabled.")
        if len(get_enabled_active_counts(cfg)) == 0:
            raise ValueError("run_subset_search=True but no active counts are enabled. Turn on one or more of subset_search_enable_k1/k2/k3/k5.")


def main(cfg: Dict[str, Any] | None = None) -> None:
    cfg = copy.deepcopy(PARAMS if cfg is None else cfg)
    enabled_counts = get_enabled_active_counts(cfg)
    cfg["subset_search_active_counts"] = enabled_counts
    cfg["selected_active_counts"] = enabled_counts
    _validate_config(cfg)
    set_seed(int(cfg["seed"]))
    device = get_device(str(cfg["device"]))
    out_dir = Path(cfg["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("DeepONet v0.7.2 | unified comparable transfer benchmark")
    print("=" * 100)
    print(f"Output dir     : {out_dir}")
    print(f"Device         : {device}")
    print(f"Experiment     : {cfg['experiment_name']}")

    summary_json: Dict[str, Any] = {"config": copy.deepcopy(cfg), "files": {}, "training_runs": {}}
    training_rows: List[Dict[str, Any]] = []

    raw_lab1 = load_raw_lab_data(cfg, "lab1")
    lab1_split = build_lab1_blocked_split(raw_lab1.time_idx, cfg, int(cfg["seed"]) + int(cfg["lab1_split"]["seed_offset"]))
    lab1_train_idx = np.where(lab1_split.masks["train"])[0]
    lab1_mean = raw_lab1.data_raw[lab1_train_idx].mean(axis=(0, 1)).astype(np.float32)
    lab1_std = raw_lab1.data_raw[lab1_train_idx].std(axis=(0, 1)).astype(np.float32)
    lab1_std = np.where(lab1_std < 1e-8, 1.0, lab1_std).astype(np.float32)
    lab1_norm = normalize_raw_lab_data(raw_lab1, lab1_mean, lab1_std)
    summary_json["lab1_split"] = lab1_split.meta

    best_lab1_path = out_dir / str(cfg["best_checkpoint_name"])
    final_lab1_path = out_dir / str(cfg["final_checkpoint_name"])
    if bool(cfg.get("train_lab1_from_scratch", True)):
        if best_lab1_path.exists() and bool(cfg.get("reuse_existing_checkpoints", True)):
            print(f"Reusing existing Lab1 checkpoint: {best_lab1_path}")
        else:
            _, _, meta = train_model(cfg, lab1_norm, lab1_split, "train", "val", device, out_dir, str(cfg["best_checkpoint_name"]), str(cfg["final_checkpoint_name"]), float(cfg["lr"]), int(cfg["epochs"]), int(cfg["early_stopping_patience"]), float(cfg["early_stopping_min_delta"]), int(cfg["seed"]), None, "lab1_pretrain", None)
            training_rows.append(meta)
            summary_json["training_runs"][meta["run_name"]] = meta
    if not best_lab1_path.exists():
        raise FileNotFoundError(f"Lab1 checkpoint not found: {best_lab1_path}")
    lab1_ckpt = _load_checkpoint(best_lab1_path, device)
    lab1_model = _checkpoint_to_model(cfg, lab1_norm, lab1_ckpt, device)

    raw_lab2 = load_raw_lab_data(cfg, "lab2")
    lab2_norm = normalize_raw_lab_data(raw_lab2, lab1_mean, lab1_std)
    lab2_split = build_lab2_phase1_split(raw_lab2.time_idx, cfg, int(cfg["seed"]) + int(cfg["lab2_split"]["seed_offset"]))
    summary_json["lab2_split"] = lab2_split.meta

    available_models: Dict[str, nn.Module] = {}
    if bool(cfg.get("run_lab2_zero_shot", True)):
        available_models["zero_shot"] = lab1_model

    adaptations = [
        ("head_only", bool(cfg.get("run_lab2_head_only", True)), str(cfg["head_best_checkpoint_name"]), str(cfg["head_final_checkpoint_name"]), float(cfg["head_only_lr"]), int(cfg["head_only_epochs"]), int(cfg["head_only_patience"]), float(cfg["head_only_min_delta"]), int(cfg["seed"]) + 5100, "head_only", "head_only"),
        ("partial_finetune", bool(cfg.get("run_lab2_partial_finetune", True)), str(cfg["partial_best_checkpoint_name"]), str(cfg["partial_final_checkpoint_name"]), float(cfg["partial_finetune_lr"]), int(cfg["partial_finetune_epochs"]), int(cfg["partial_finetune_patience"]), float(cfg["partial_finetune_min_delta"]), int(cfg["seed"]) + 5200, "partial", "partial_finetune"),
        ("full_finetune", bool(cfg.get("run_lab2_full_finetune", True)), str(cfg["full_ft_best_checkpoint_name"]), str(cfg["full_ft_final_checkpoint_name"]), float(cfg["full_finetune_lr"]), int(cfg["full_finetune_epochs"]), int(cfg["full_finetune_patience"]), float(cfg["full_finetune_min_delta"]), int(cfg["seed"]) + 5300, "full", "full_finetune"),
    ]
    for display_mode, enabled, best_name, final_name, lr, epochs, patience, min_delta, seed, train_mode, model_key in adaptations:
        if not enabled:
            continue
        best_path = out_dir / best_name
        final_path = out_dir / final_name
        if best_path.exists() and bool(cfg.get("reuse_existing_checkpoints", True)):
            print(f"Reusing existing {display_mode} checkpoint: {best_path}")
            ckpt = _load_checkpoint(best_path, device)
            meta = _ckpt_meta(ckpt, f"lab2_{display_mode}", train_mode, best_path, final_path)
        else:
            _, _, meta = train_model(cfg, lab2_norm, lab2_split, "adapt_train", "adapt_val", device, out_dir, best_name, final_name, lr, epochs, patience, min_delta, seed, lab1_ckpt["model_state"], f"lab2_{display_mode}", train_mode)
            ckpt = _load_checkpoint(best_path, device)
        training_rows.append(meta)
        summary_json["training_runs"][meta["run_name"]] = meta
        available_models[model_key] = _checkpoint_to_model(cfg, lab2_norm, ckpt, device)

    training_summary_df = pd.DataFrame(training_rows)
    if not training_summary_df.empty:
        training_summary_df.to_csv(out_dir / str(cfg["training_runs_summary_name"]), index=False)
        summary_json["files"]["training_runs_summary"] = str(out_dir / str(cfg["training_runs_summary_name"]))

    if bool(cfg.get("run_manual_lab2_protocol_eval", False)):
        manual_protocols = build_manual_protocols(cfg, lab2_norm.sensor_ids)
        mm, ms, mp = [], [], []
        for mode, model in available_models.items():
            a, b, c = evaluate_active_protocols(model, lab2_norm, lab2_split.masks["test"], manual_protocols, device, mode, cfg)
            mm.append(a); ms.append(b); mp.append(c)
        if mm:
            pd.concat(mm, ignore_index=True).to_csv(out_dir / str(cfg["manual_lab2_metrics_name"]), index=False)
            pd.concat(ms, ignore_index=True).to_csv(out_dir / str(cfg["manual_lab2_sensor_summary_name"]), index=False)
            pd.concat(mp, ignore_index=True).to_csv(out_dir / str(cfg["manual_lab2_protocol_summary_name"]), index=False)
            summary_json["files"]["manual_lab2_metrics"] = str(out_dir / str(cfg["manual_lab2_metrics_name"]))

    selected_df = pd.DataFrame()
    subset_search_best_df = pd.DataFrame()
    subset_distribution_summary_df = pd.DataFrame()
    subset_gap_summary_df = pd.DataFrame()
    if bool(cfg.get("run_subset_search", True)):
        source_mode = str(cfg["subset_search_source_mode"])
        subset_protocols = build_generated_protocols(cfg, lab2_norm.sensor_ids)
        metrics_df, sensor_df, protocol_df = evaluate_active_protocols(
            available_models[source_mode],
            lab2_norm,
            lab2_split.masks["test"],
            subset_protocols,
            device,
            source_mode,
            cfg,
        )
        best_df = protocol_df.pivot_table(
            index=["transfer_mode", "protocol_name", "display_name", "active_total_nominal", "masked_total"],
            columns="variable",
            values="r2",
        ).reset_index()
        for col in ["temperature", "humidity"]:
            if col not in best_df.columns:
                best_df[col] = np.nan
        best_df = best_df.rename(
            columns={
                "active_total_nominal": "active_count",
                "temperature": "r2_temperature",
                "humidity": "r2_humidity",
            }
        )
        best_df["score"] = best_df[["r2_temperature", "r2_humidity"]].mean(axis=1)
        best_df = best_df.sort_values(["transfer_mode", "active_count", "score"], ascending=[True, True, False]).reset_index(drop=True)
        metrics_df.to_csv(out_dir / str(cfg["subset_search_metrics_name"]), index=False)
        sensor_df.to_csv(out_dir / str(cfg["subset_search_sensor_summary_name"]), index=False)
        protocol_df.to_csv(out_dir / str(cfg["subset_search_protocol_summary_name"]), index=False)
        best_df.to_csv(out_dir / str(cfg["subset_search_best_subsets_name"]), index=False)
        summary_json["files"]["subset_search_metrics"] = str(out_dir / str(cfg["subset_search_metrics_name"]))
        summary_json["files"]["subset_search_sensor_summary"] = str(out_dir / str(cfg["subset_search_sensor_summary_name"]))
        summary_json["files"]["subset_search_protocol_summary"] = str(out_dir / str(cfg["subset_search_protocol_summary_name"]))
        summary_json["files"]["subset_search_best_subsets"] = str(out_dir / str(cfg["subset_search_best_subsets_name"]))
        subset_search_best_df = best_df

        if bool(cfg.get("export_subset_distribution_analysis", True)):
            subset_distribution_summary_df, subset_gap_summary_df = build_subset_distribution_tables(protocol_df)
            if not subset_distribution_summary_df.empty:
                subset_distribution_summary_df.to_csv(out_dir / str(cfg["subset_distribution_summary_name"]), index=False)
                summary_json["files"]["subset_distribution_summary"] = str(out_dir / str(cfg["subset_distribution_summary_name"]))
            if bool(cfg.get("export_subset_gap_table", True)) and not subset_gap_summary_df.empty:
                subset_gap_summary_df.to_csv(out_dir / str(cfg["subset_gap_summary_name"]), index=False)
                summary_json["files"]["subset_gap_summary"] = str(out_dir / str(cfg["subset_gap_summary_name"]))
            if bool(cfg.get("export_subset_r2_boxplots", True)):
                _plot_subset_r2_boxplot(protocol_df, "temperature", out_dir / str(cfg["subset_r2_boxplot_temperature_name"]), cfg)
                _plot_subset_r2_boxplot(protocol_df, "humidity", out_dir / str(cfg["subset_r2_boxplot_humidity_name"]), cfg)
                summary_json["files"]["subset_r2_boxplot_temperature"] = str(out_dir / str(cfg["subset_r2_boxplot_temperature_name"]))
                summary_json["files"]["subset_r2_boxplot_humidity"] = str(out_dir / str(cfg["subset_r2_boxplot_humidity_name"]))

        selected_protocols = build_selected_protocols(best_df, int(cfg["top_n_per_k"]), [int(x) for x in cfg["selected_active_counts"]], source_mode)
        selected_df = selected_protocols_to_df(selected_protocols)
        selected_df.to_csv(out_dir / str(cfg["selected_protocols_name"]), index=False)
        summary_json["files"]["selected_protocols"] = str(out_dir / str(cfg["selected_protocols_name"]))

    regime_summary_df = pd.DataFrame()
    unified_protocol_summary_df = pd.DataFrame()
    if bool(cfg.get("run_unified_benchmark", True)):
        if selected_df.empty:
            raise ValueError("Unified benchmark requires selected protocols. Enable run_subset_search.")
        selected_protocols = [TransferProtocol(str(row.protocol_name), str(row.display_name), tuple(int(x) for x in row.active_sensors)) for row in selected_df.itertuples(index=False)]
        metric_parts: List[pd.DataFrame] = []
        sensor_parts: List[pd.DataFrame] = []
        protocol_parts: List[pd.DataFrame] = []
        if bool(cfg.get("run_lab1_reference", True)):
            lab1_metrics, lab1_sensor_summary, lab1_protocol_summary = evaluate_active_protocols(lab1_model, lab1_norm, lab1_split.masks["test"], selected_protocols, device, "lab1_reference", cfg)
            lab1_metrics.to_csv(out_dir / str(cfg["lab1_eval_metrics_name"]), index=False)
            summary_json["files"]["lab1_reference_metrics"] = str(out_dir / str(cfg["lab1_eval_metrics_name"]))
            metric_parts.append(lab1_metrics); sensor_parts.append(lab1_sensor_summary); protocol_parts.append(lab1_protocol_summary)
        for mode in ["zero_shot", "head_only", "partial_finetune", "full_finetune"]:
            if mode not in available_models:
                continue
            a, b, c = evaluate_active_protocols(available_models[mode], lab2_norm, lab2_split.masks["test"], selected_protocols, device, mode, cfg)
            metric_parts.append(a); sensor_parts.append(b); protocol_parts.append(c)
        unified_metrics_df = attach_protocol_metadata(pd.concat(metric_parts, ignore_index=True), selected_df)
        unified_sensor_summary_df = attach_protocol_metadata(pd.concat(sensor_parts, ignore_index=True), selected_df)
        unified_protocol_summary_df = attach_protocol_metadata(pd.concat(protocol_parts, ignore_index=True), selected_df)
        unified_metrics_df.to_csv(out_dir / str(cfg["unified_metrics_name"]), index=False)
        unified_sensor_summary_df.to_csv(out_dir / str(cfg["unified_sensor_summary_name"]), index=False)
        unified_protocol_summary_df.to_csv(out_dir / str(cfg["unified_protocol_summary_name"]), index=False)
        summary_json["files"]["unified_metrics"] = str(out_dir / str(cfg["unified_metrics_name"]))
        summary_json["files"]["unified_sensor_summary"] = str(out_dir / str(cfg["unified_sensor_summary_name"]))
        summary_json["files"]["unified_protocol_summary"] = str(out_dir / str(cfg["unified_protocol_summary_name"]))
        regime_summary_df, regime_pivot_df = build_regime_summary(unified_protocol_summary_df, selected_df, cfg)
        if not regime_summary_df.empty:
            regime_summary_df.to_csv(out_dir / str(cfg["benchmark_regime_summary_name"]), index=False)
            summary_json["files"]["benchmark_regime_summary"] = str(out_dir / str(cfg["benchmark_regime_summary_name"]))
        if not regime_pivot_df.empty:
            regime_pivot_df.to_csv(out_dir / str(cfg["benchmark_regime_pivot_r2_name"]), index=False)
            summary_json["files"]["benchmark_regime_pivot_r2"] = str(out_dir / str(cfg["benchmark_regime_pivot_r2_name"]))
        _plot_overview_bars(unified_protocol_summary_df, out_dir / str(cfg["overview_figure_name"]), cfg)
        _plot_sensor_map_panel_grid(unified_sensor_summary_df, "temperature", out_dir / str(cfg["sensor_map_temperature_name"]), cfg)
        _plot_sensor_map_panel_grid(unified_sensor_summary_df, "humidity", out_dir / str(cfg["sensor_map_humidity_name"]), cfg)
        summary_json["files"]["overview_figure"] = str(out_dir / str(cfg["overview_figure_name"]))
        summary_json["files"]["sensor_map_temperature"] = str(out_dir / str(cfg["sensor_map_temperature_name"]))
        summary_json["files"]["sensor_map_humidity"] = str(out_dir / str(cfg["sensor_map_humidity_name"]))
        if 5 in _enabled_subset_compare_targets(cfg):
            _plot_subset_compare(unified_protocol_summary_df, 5, out_dir / str(cfg["subset_compare_k5_name"]), cfg)
            summary_json["files"]["subset_compare_k5"] = str(out_dir / str(cfg["subset_compare_k5_name"]))
        if 3 in _enabled_subset_compare_targets(cfg):
            _plot_subset_compare(unified_protocol_summary_df, 3, out_dir / str(cfg["subset_compare_k3_name"]), cfg)
            summary_json["files"]["subset_compare_k3"] = str(out_dir / str(cfg["subset_compare_k3_name"]))
        if 2 in _enabled_subset_compare_targets(cfg):
            _plot_subset_compare(unified_protocol_summary_df, 2, out_dir / str(cfg["subset_compare_k2_name"]), cfg)
            summary_json["files"]["subset_compare_k2"] = str(out_dir / str(cfg["subset_compare_k2_name"]))
        if 1 in _enabled_subset_compare_targets(cfg):
            _plot_subset_compare(unified_protocol_summary_df, 1, out_dir / str(cfg["subset_compare_k1_name"]), cfg)
            summary_json["files"]["subset_compare_k1"] = str(out_dir / str(cfg["subset_compare_k1_name"]))
        if bool(cfg.get("export_field_maps", True)):
            field_models = {k: v for k, v in available_models.items() if k in {"zero_shot", "head_only", "partial_finetune", "full_finetune"}}
            if field_models:
                field_manifest = {
                    "temperature": _plot_example_field_map(lab2_norm, lab2_split.masks["test"], selected_df, field_models, device, cfg, "temperature", out_dir / str(cfg["field_map_temperature_name"])),
                    "humidity": _plot_example_field_map(lab2_norm, lab2_split.masks["test"], selected_df, field_models, device, cfg, "humidity", out_dir / str(cfg["field_map_humidity_name"])),
                }
                summary_json["field_map_manifest"] = field_manifest
                summary_json["files"]["field_map_temperature"] = str(out_dir / str(cfg["field_map_temperature_name"]))
                summary_json["files"]["field_map_humidity"] = str(out_dir / str(cfg["field_map_humidity_name"]))

    _build_summary_markdown(selected_df, regime_summary_df, training_summary_df, out_dir / str(cfg["summary_md_name"]), cfg)
    summary_json["files"]["summary_md"] = str(out_dir / str(cfg["summary_md_name"]))
    summary_json["files"]["summary_json"] = str(out_dir / str(cfg["summary_json_name"]))
    with open(out_dir / str(cfg["summary_json_name"]), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print("\nSaved artifacts")
    for key, value in summary_json["files"].items():
        print(f"  {key:28s}: {value}")


if __name__ == "__main__":
    main()
