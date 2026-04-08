from __future__ import annotations

import copy
import importlib.util
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd


# =============================================================================
# Config
# =============================================================================
PARAMS: Dict[str, Any] = {
    "phase1_run_dir": "./deepOnet/runs/lab1_to_lab2_v0.6.1",
    "phase2_run_dir": "./deepOnet/runs/lab1_to_lab2_v0.6.2",
    "output_dir": "./deepOnet/runs/lab1_to_lab2_v0.6.3",
    "helper_script": "./deepOnet/deepOnet_v0.6.2.py",
    "device": "auto",
    "top_n_per_k": 5,
    "selected_active_counts": [5, 3, 2],
    "include_lab1_reference_in_subset_figures": True,
    "include_partial_finetune": False,

    # Checkpoint names.
    "lab1_best_checkpoint_name": "thermodt_lab1_best.pt",
    "lab2_head_best_checkpoint_name": "thermodt_lab2_head_best.pt",
    "lab2_partial_best_checkpoint_name": "thermodt_lab2_partial_best.pt",
    "lab2_fullft_best_checkpoint_name": "thermodt_lab2_fullft_best.pt",

    # Input CSVs already produced by v0.6.1 / v0.6.2.
    "phase1_lab1_metrics_name": "thermodt_lab1_test_metrics_v061.csv",
    "phase1_lab2_metrics_name": "thermodt_phase1_lab2_metrics_v061.csv",
    "phase2_best_subsets_name": "thermodt_phase2_lab2_best_subsets_v062.csv",

    # Output CSV/JSON names.
    "selected_protocols_name": "selected_protocols_v063a.csv",
    "combined_metrics_name": "transfer_unified_metrics_v063a.csv",
    "combined_sensor_summary_name": "transfer_unified_sensor_summary_v063a.csv",
    "combined_protocol_summary_name": "transfer_unified_protocol_summary_v063a.csv",
    "summary_json_name": "transfer_unified_summary_v063a.json",
    "summary_md_name": "transfer_unified_summary_v063a.md",

    # Output figures.
    "overview_figure_name": "transfer_overview_top5_subsets_v063a.png",
    "sensor_map_temperature_name": "sensor_map_all13_temperature_v063a.png",
    "sensor_map_humidity_name": "sensor_map_all13_humidity_v063a.png",
    "subset_compare_k5_name": "transfer_compare_top5_k5_v063a.png",
    "subset_compare_k3_name": "transfer_compare_top5_k3_v063a.png",
    "subset_compare_k2_name": "transfer_compare_top5_k2_v063a.png",
}


# =============================================================================
# Fixed geometry (from the dataset paper)
# =============================================================================
LAB1_ROOM_W = 7.81
LAB1_ROOM_H = 7.82
LAB2_ROOM_W = 6.60
LAB2_ROOM_H = 12.60

LAB1_COORDS: Dict[int, Tuple[float, float]] = {
    1: (7.21, 0.60), 2: (3.88, 5.88), 3: (5.85, 5.88),
    4: (0.60, 0.60), 5: (1.92, 1.94), 6: (3.88, 3.91),
    7: (1.92, 3.91), 8: (5.85, 3.91), 9: (7.21, 7.22),
    10: (1.92, 5.88), 11: (0.60, 7.22), 12: (5.85, 1.94),
    13: (3.88, 1.94),
}
LAB2_COORDS: Dict[int, Tuple[float, float]] = {
    1: (5.70, 7.54), 2: (5.70, 10.17), 3: (3.30, 11.60),
    4: (3.30, 9.12), 5: (3.30, 6.34), 6: (3.30, 3.86),
    7: (3.30, 1.38), 8: (5.70, 4.91), 9: (5.70, 2.43),
    10: (0.90, 2.43), 11: (0.90, 10.17), 12: (0.90, 4.91),
    13: (0.90, 7.54),
}


# =============================================================================
# Data classes
# =============================================================================
@dataclass(frozen=True)
class SelectedProtocol:
    protocol_name: str
    display_name: str
    active_sensors: Tuple[int, ...]
    active_count: int
    rank_within_k: int | None
    source_score: float | None
    source_r2_temperature: float | None
    source_r2_humidity: float | None


# =============================================================================
# Helpers
# =============================================================================
def _load_helper_module(helper_path: Path):
    import sys

    spec = importlib.util.spec_from_file_location("deeponet_v062_helper", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_active_sensors(protocol_name: str) -> Tuple[int, ...]:
    if protocol_name == "lab2_all13":
        return tuple(range(1, 14))
    m = re.fullmatch(r"lab2_active_(\d+(?:_\d+)*)", str(protocol_name))
    if not m:
        raise ValueError(f"Unsupported protocol_name format: {protocol_name}")
    return tuple(int(x) for x in m.group(1).split("_"))


def _protocol_name_from_active(active: Sequence[int]) -> str:
    return "lab2_active_" + "_".join(str(int(x)) for x in active)


def _display_name_from_active(active: Sequence[int]) -> str:
    if len(active) == 13:
        return "Lab2 all 13 active"
    return "Lab2 active set " + str(list(int(x) for x in active))


def _pretty_subset_label(active: Sequence[int]) -> str:
    return "{" + ", ".join(str(int(x)) for x in active) + "}"


def _sensor_coords_for_lab(lab_name: str) -> Dict[int, Tuple[float, float]]:
    if lab_name == "lab1":
        return LAB1_COORDS
    if lab_name == "lab2":
        return LAB2_COORDS
    raise ValueError(f"Unsupported lab_name: {lab_name}")


def _room_dims_for_lab(lab_name: str) -> Tuple[float, float]:
    if lab_name == "lab1":
        return LAB1_ROOM_W, LAB1_ROOM_H
    if lab_name == "lab2":
        return LAB2_ROOM_W, LAB2_ROOM_H
    raise ValueError(f"Unsupported lab_name: {lab_name}")


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def _build_selected_protocols(best_subsets_df: pd.DataFrame, top_n_per_k: int, ks: Sequence[int]) -> List[SelectedProtocol]:
    head = best_subsets_df[best_subsets_df["transfer_mode"] == "head_only"].copy()
    if head.empty:
        raise ValueError("No head_only rows found in phase2 best-subsets CSV.")

    protocols: List[SelectedProtocol] = [
        SelectedProtocol(
            protocol_name="lab2_all13",
            display_name="Lab2 all 13 active",
            active_sensors=tuple(range(1, 14)),
            active_count=13,
            rank_within_k=None,
            source_score=None,
            source_r2_temperature=None,
            source_r2_humidity=None,
        )
    ]

    for k in [int(x) for x in ks]:
        sub = head[head["active_count"] == k].sort_values(["score", "r2_temperature", "r2_humidity"], ascending=False).head(int(top_n_per_k))
        if sub.empty:
            raise ValueError(f"No head_only subsets found for k={k}")
        for rank, row in enumerate(sub.itertuples(index=False), start=1):
            active = _parse_active_sensors(row.protocol_name)
            protocols.append(
                SelectedProtocol(
                    protocol_name=str(row.protocol_name),
                    display_name=_display_name_from_active(active),
                    active_sensors=active,
                    active_count=int(k),
                    rank_within_k=rank,
                    source_score=float(row.score),
                    source_r2_temperature=float(row.r2_temperature),
                    source_r2_humidity=float(row.r2_humidity),
                )
            )
    return protocols


def _selected_protocols_to_df(protocols: Sequence[SelectedProtocol]) -> pd.DataFrame:
    rows = []
    for p in protocols:
        rows.append({
            "protocol_name": p.protocol_name,
            "display_name": p.display_name,
            "active_sensors": list(p.active_sensors),
            "active_count": p.active_count,
            "masked_total": 13 - p.active_count,
            "rank_within_k": p.rank_within_k,
            "source_score": p.source_score,
            "source_r2_temperature": p.source_r2_temperature,
            "source_r2_humidity": p.source_r2_humidity,
        })
    return pd.DataFrame(rows)


def _protocols_for_eval(helper_mod, selected: Sequence[SelectedProtocol]):
    protocols = []
    for p in selected:
        protocols.append(helper_mod.TransferProtocol(protocol_name=p.protocol_name, display_name=p.display_name, active_sensors=list(p.active_sensors)))
    return protocols


def _evaluate_checkpoint(
    helper_mod,
    checkpoint: Mapping[str, Any],
    cfg: Mapping[str, Any],
    normalized,
    eval_mask: np.ndarray,
    selected_protocols: Sequence[SelectedProtocol],
    transfer_mode: str,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = helper_mod.build_model(cfg, len(normalized.sensor_ids), len(normalized.variable_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    helper_mod.PARAMS = dict(cfg)  # evaluate_active_protocols references module-level PARAMS
    return helper_mod.evaluate_active_protocols(model, normalized, eval_mask, _protocols_for_eval(helper_mod, selected_protocols), device, transfer_mode=transfer_mode)


def _idw_surface(
    coords_by_sensor: Mapping[int, Tuple[float, float]],
    values_by_sensor: Mapping[int, float],
    room_w: float,
    room_h: float,
    grid_nx: int = 220,
    grid_ny: int = 220,
    power: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sensor_ids = sorted(values_by_sensor.keys())
    pts = np.array([coords_by_sensor[s] for s in sensor_ids], dtype=float)
    vals = np.array([values_by_sensor[s] for s in sensor_ids], dtype=float)

    xs = np.linspace(0.0, float(room_w), int(grid_nx))
    ys = np.linspace(0.0, float(room_h), int(grid_ny))
    xx, yy = np.meshgrid(xs, ys)

    dx = xx[..., None] - pts[:, 0][None, None, :]
    dy = yy[..., None] - pts[:, 1][None, None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    # Exact sensor hits inherit exact values.
    exact = dist < 1e-12
    weights = 1.0 / np.maximum(dist, 1e-6) ** power
    z = (weights * vals[None, None, :]).sum(axis=2) / weights.sum(axis=2)
    if exact.any():
        hit_rows, hit_cols, hit_sensor_idx = np.where(exact)
        z[hit_rows, hit_cols] = vals[hit_sensor_idx]
    return xx, yy, z


def _plot_sensor_map_4panel(
    sensor_df: pd.DataFrame,
    variable: str,
    outfile: Path,
) -> None:
    mode_order = ["lab1_reference", "zero_shot", "head_only", "full_finetune"]
    title_map = {
        "lab1_reference": "LAB1 reference (all 13 active)",
        "zero_shot": "LAB2 zero-shot (all 13 active)",
        "head_only": "LAB2 head-only (all 13 active)",
        "full_finetune": "LAB2 full fine-tune (all 13 active)",
    }
    lab_by_mode = {
        "lab1_reference": "lab1",
        "zero_shot": "lab2",
        "head_only": "lab2",
        "full_finetune": "lab2",
    }

    frames = []
    for mode in mode_order:
        sub = sensor_df[(sensor_df["transfer_mode"] == mode) & (sensor_df["protocol_name"] == "lab2_all13") & (sensor_df["variable"] == variable)].copy()
        if not sub.empty:
            frames.append(sub)
    if not frames:
        raise ValueError(f"No all-13 sensor rows found for variable={variable}")

    vmin = min(float(df["r2"].min()) for df in frames)
    vmax = max(float(df["r2"].max()) for df in frames)
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-3

    fig = plt.figure(figsize=(18.5, 6.6), constrained_layout=False)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.055], wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    cax = fig.add_subplot(gs[0, 4])
    cmap = plt.get_cmap("RdYlGn")
    im = None

    for ax, mode in zip(axes, mode_order):
        sub = sensor_df[(sensor_df["transfer_mode"] == mode) & (sensor_df["protocol_name"] == "lab2_all13") & (sensor_df["variable"] == variable)].copy()
        lab_name = lab_by_mode[mode]
        coords = _sensor_coords_for_lab(lab_name)
        room_w, room_h = _room_dims_for_lab(lab_name)
        ax.set_title(title_map[mode], fontsize=15, fontweight="bold", pad=10)
        ax.set_xlim(0, room_w)
        ax.set_ylim(0, room_h)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.15)

        if sub.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=13)
            continue

        values_by_sensor = {int(row.sensor): float(row.r2) for row in sub.itertuples(index=False)}
        xx, yy, zz = _idw_surface(coords, values_by_sensor, room_w, room_h)
        im = ax.imshow(
            zz,
            origin="lower",
            extent=(0, room_w, 0, room_h),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="bilinear",
            zorder=0,
        )

        for sid, (x, y) in coords.items():
            if sid not in values_by_sensor:
                continue
            val = values_by_sensor[sid]
            ax.scatter([x], [y], s=260, facecolor="none", edgecolor="black", linewidth=1.2, zorder=3)
            ax.text(
                x + 0.05 * room_w / 6.6,
                y + 0.03 * room_h / 12.6,
                f"{sid}\n{val:.3f}",
                fontsize=10.5,
                color="black",
                zorder=4,
                bbox=dict(boxstyle="round,pad=0.18", facecolor=(1, 1, 1, 0.25), edgecolor="none"),
            )

    if im is not None:
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Sensor R²", fontsize=14)
        cb.ax.tick_params(labelsize=11)

    fig.suptitle(f"LAB1 → LAB2 all-13 sensor transfer map ({variable})", fontsize=22, fontweight="bold", y=0.97)
    fig.subplots_adjust(left=0.04, right=0.965, bottom=0.10, top=0.88, wspace=0.18)
    fig.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_overview_bars(protocol_summary: pd.DataFrame, outfile: Path, include_lab1_reference: bool = True) -> None:
    ks = [13, 5, 3, 2]
    category_labels = {13: "All 13 active", 5: "Top 5 (k=5) mean", 3: "Top 5 (k=3) mean", 2: "Top 5 (k=2) mean"}

    mode_order = ["zero_shot", "head_only", "full_finetune"]
    mode_labels = {
        "zero_shot": "LAB2 zero-shot",
        "head_only": "LAB2 head-only",
        "full_finetune": "LAB2 full fine-tune",
        "lab1_reference": "LAB1 reference",
    }
    colors = {
        "lab1_reference": "#1f77b4",
        "zero_shot": "#ff7f0e",
        "head_only": "#9467bd",
        "full_finetune": "#2ca02c",
    }
    if include_lab1_reference:
        mode_order = ["lab1_reference"] + mode_order

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), sharey=False)
    width = 0.18 if include_lab1_reference else 0.22
    x = np.arange(len(ks), dtype=float)

    for ax, variable in zip(axes, ["temperature", "humidity"]):
        sub = protocol_summary[protocol_summary["variable"] == variable].copy()
        for i, mode in enumerate(mode_order):
            vals: List[float] = []
            for k in ks:
                g = sub[(sub["transfer_mode"] == mode) & (sub["active_total_nominal"] == k)]
                vals.append(float(g["r2"].mean()) if not g.empty else float("nan"))
            offs = (i - (len(mode_order) - 1) / 2.0) * width
            bars = ax.bar(x + offs, vals, width=width, label=mode_labels[mode], color=colors[mode], alpha=0.95)
            for bar, val in zip(bars, vals):
                if math.isfinite(val):
                    ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.004, f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels([category_labels[k] for k in ks], fontsize=10)
        ax.set_ylabel("R²")
        ax.set_title(f"{variable.capitalize()} transfer summary", fontsize=15, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        # Dynamic y-limits from present values.
        vals_present = sub["r2"].dropna().to_numpy(dtype=float)
        if len(vals_present):
            ymin = max(0.0, float(vals_present.min()) - 0.05)
            ymax = min(1.01, float(vals_present.max()) + 0.04)
        else:
            ymin, ymax = 0.0, 1.0
        ax.set_ylim(ymin, ymax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.01), ncol=len(mode_order), frameon=False, fontsize=11)
    fig.suptitle("Unified LAB1/LAB2 transfer benchmark on selected sparse subsets", fontsize=20, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.18, wspace=0.14)
    fig.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_subset_compare(protocol_summary: pd.DataFrame, active_count: int, outfile: Path, include_lab1_reference: bool = True) -> None:
    sub = protocol_summary[protocol_summary["active_total_nominal"] == int(active_count)].copy()
    # Exclude all13 here; only top-5 chosen protocols for this k.
    sub = sub[sub["protocol_name"] != "lab2_all13"].copy()
    if sub.empty:
        raise ValueError(f"No protocol summary rows found for active_count={active_count}")

    # Order by head-only score.
    order = (
        sub[(sub["transfer_mode"] == "head_only") & (sub["variable"] == "temperature")][["protocol_name", "score"]]
        if "score" in sub.columns else None
    )

    # Derive protocol order from head-only mean over variables if score absent.
    order = (
        sub[sub["transfer_mode"] == "head_only"].groupby("protocol_name")["r2"].mean().sort_values(ascending=False).index.tolist()
    )
    label_map = {
        pn: _pretty_subset_label(_parse_active_sensors(pn))
        for pn in order
    }

    mode_order = ["zero_shot", "head_only", "full_finetune"]
    mode_labels = {
        "zero_shot": "LAB2 zero-shot",
        "head_only": "LAB2 head-only",
        "full_finetune": "LAB2 full fine-tune",
        "lab1_reference": "LAB1 reference",
    }
    colors = {
        "lab1_reference": "#1f77b4",
        "zero_shot": "#ff7f0e",
        "head_only": "#9467bd",
        "full_finetune": "#2ca02c",
    }
    if include_lab1_reference:
        mode_order = ["lab1_reference"] + mode_order

    fig, axes = plt.subplots(2, 1, figsize=(15.5, 9.0), sharex=True)
    width = 0.18 if include_lab1_reference else 0.22
    x = np.arange(len(order), dtype=float)

    for ax, variable in zip(axes, ["temperature", "humidity"]):
        var_df = sub[sub["variable"] == variable].copy()
        for i, mode in enumerate(mode_order):
            vals = []
            for pn in order:
                g = var_df[(var_df["transfer_mode"] == mode) & (var_df["protocol_name"] == pn)]
                vals.append(float(g["r2"].mean()) if not g.empty else float("nan"))
            offs = (i - (len(mode_order) - 1) / 2.0) * width
            bars = ax.bar(x + offs, vals, width=width, label=mode_labels[mode], color=colors[mode], alpha=0.95)
            for bar, val in zip(bars, vals):
                if math.isfinite(val):
                    ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.004, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("R²")
        ax.set_title(f"{variable.capitalize()} — top 5 subsets with k={active_count}", fontsize=14.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        vals_present = var_df["r2"].dropna().to_numpy(dtype=float)
        ymin = max(0.0, float(vals_present.min()) - 0.05) if len(vals_present) else 0.0
        ymax = min(1.01, float(vals_present.max()) + 0.04) if len(vals_present) else 1.0
        ax.set_ylim(ymin, ymax)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([label_map[pn] for pn in order], fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.01), ncol=len(mode_order), frameon=False, fontsize=11)
    fig.suptitle(f"Unified subset comparison for k={active_count}", fontsize=20, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.17, hspace=0.30)
    fig.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_summary_markdown(
    selected_df: pd.DataFrame,
    protocol_summary: pd.DataFrame,
    output_path: Path,
    include_lab1_reference: bool,
) -> None:
    lines: List[str] = []
    lines.append("# Unified transfer benchmark (v0.6.3a)")
    lines.append("")
    lines.append("This report re-evaluates the saved v0.6.1 / v0.6.2 checkpoints on a single common benchmark:")
    lines.append("")
    lines.append("- LAB2 all 13 active")
    for k in [5, 3, 2]:
        lines.append(f"- Top {len(selected_df[selected_df['active_count'] == k])} head-only subsets for k={k}")
    lines.append("")

    mode_order = ["zero_shot", "head_only", "full_finetune"]
    if include_lab1_reference:
        mode_order = ["lab1_reference"] + mode_order

    lines.append("## Mean R² by regime")
    lines.append("")
    for variable in ["temperature", "humidity"]:
        lines.append(f"### {variable.capitalize()}")
        lines.append("")
        lines.append("| Mode | All13 | Top5 k=5 mean | Top5 k=3 mean | Top5 k=2 mean |")
        lines.append("|---|---:|---:|---:|---:|")
        var_df = protocol_summary[protocol_summary["variable"] == variable]
        for mode in mode_order:
            vals = []
            for k in [13, 5, 3, 2]:
                g = var_df[(var_df["transfer_mode"] == mode) & (var_df["active_total_nominal"] == k)]
                vals.append(float(g["r2"].mean()) if not g.empty else float("nan"))
            fmt = lambda x: (f"{x:.3f}" if math.isfinite(x) else "—")
            lines.append(f"| {mode} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} | {fmt(vals[3])} |")
        lines.append("")

    lines.append("## Selected subsets")
    lines.append("")
    lines.append("| k | Rank | Protocol | Sensors | Head-only source score |")
    lines.append("|---:|---:|---|---|---:|")
    for row in selected_df[selected_df["protocol_name"] != "lab2_all13"].sort_values(["active_count", "rank_within_k"], ascending=[False, True]).itertuples(index=False):
        lines.append(f"| {row.active_count} | {row.rank_within_k} | {row.protocol_name} | {_pretty_subset_label(row.active_sensors)} | {row.source_score:.3f} |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================
def main(cfg: Dict[str, Any] | None = None) -> None:
    cfg = copy.deepcopy(cfg or PARAMS)
    root = Path.cwd()
    phase1_dir = (root / str(cfg["phase1_run_dir"])).resolve()
    phase2_dir = (root / str(cfg["phase2_run_dir"])).resolve()
    out_dir = (root / str(cfg["output_dir"])).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    helper_path = (root / str(cfg["helper_script"])).resolve()

    helper = _load_helper_module(helper_path)
    device = helper.get_device(str(cfg["device"]))

    print("=" * 96)
    print("DeepONet v0.6.3a | unified LAB1/LAB2 transfer re-evaluation")
    print("=" * 96)
    print(f"Phase1 run dir : {phase1_dir}")
    print(f"Phase2 run dir : {phase2_dir}")
    print(f"Output dir     : {out_dir}")
    print(f"Device         : {device}")

    # ------------------------------------------------------------------
    # Load selection CSV and choose unified protocol set.
    # ------------------------------------------------------------------
    best_subsets_path = phase2_dir / str(cfg["phase2_best_subsets_name"])
    best_subsets = pd.read_csv(best_subsets_path)
    selected_protocols = _build_selected_protocols(best_subsets, int(cfg["top_n_per_k"]), [int(x) for x in cfg["selected_active_counts"]])
    selected_df = _selected_protocols_to_df(selected_protocols)
    selected_df.to_csv(out_dir / str(cfg["selected_protocols_name"]), index=False)

    print(f"Selected protocols: {len(selected_protocols)} total")
    for k in [13, 5, 3, 2]:
        n = int((selected_df["active_count"] == k).sum())
        if n:
            print(f"  active_count={k}: {n}")

    # ------------------------------------------------------------------
    # Load checkpoints.
    # ------------------------------------------------------------------
    lab1_ckpt = _load_checkpoint(phase1_dir / str(cfg["lab1_best_checkpoint_name"]))
    head_ckpt = _load_checkpoint(phase2_dir / str(cfg["lab2_head_best_checkpoint_name"]))
    full_ckpt = _load_checkpoint(phase1_dir / str(cfg["lab2_fullft_best_checkpoint_name"]))
    partial_ckpt = None
    if bool(cfg.get("include_partial_finetune", False)):
        partial_ckpt = _load_checkpoint(phase2_dir / str(cfg["lab2_partial_best_checkpoint_name"]))

    phase_cfg = copy.deepcopy(lab1_ckpt["config"])
    helper.PARAMS = dict(phase_cfg)
    helper.set_seed(int(phase_cfg["seed"]))

    # ------------------------------------------------------------------
    # Load and normalize data exactly as in transfer training.
    # ------------------------------------------------------------------
    raw_lab1 = helper.load_raw_lab_data(phase_cfg, "lab1")
    raw_lab2 = helper.load_raw_lab_data(phase_cfg, "lab2")
    lab1_mean = np.asarray(lab1_ckpt["stats"]["mean"], dtype=np.float32)
    lab1_std = np.asarray(lab1_ckpt["stats"]["std"], dtype=np.float32)
    lab1_norm = helper.normalize_raw_lab_data(raw_lab1, lab1_mean, lab1_std)
    lab2_norm = helper.normalize_raw_lab_data(raw_lab2, lab1_mean, lab1_std)

    lab1_split = helper.build_lab1_blocked_split(raw_lab1.time_idx, phase_cfg, int(phase_cfg["seed"]) + int(phase_cfg["lab1_split"]["seed_offset"]))
    lab2_split = helper.build_lab2_phase1_split(raw_lab2.time_idx, phase_cfg, int(phase_cfg["seed"]) + int(phase_cfg["lab2_split"]["seed_offset"]))

    # ------------------------------------------------------------------
    # Evaluate models on the unified protocol set.
    # ------------------------------------------------------------------
    all_metrics: List[pd.DataFrame] = []
    all_sensor_summary: List[pd.DataFrame] = []
    all_protocol_summary: List[pd.DataFrame] = []

    print("\nEvaluating LAB1 reference on LAB1 test ...")
    m, s, p = _evaluate_checkpoint(helper, lab1_ckpt, phase_cfg, lab1_norm, lab1_split.masks["test"], selected_protocols, "lab1_reference", device)
    all_metrics.append(m)
    all_sensor_summary.append(s)
    all_protocol_summary.append(p)

    print("Evaluating LAB2 zero-shot ...")
    m, s, p = _evaluate_checkpoint(helper, lab1_ckpt, phase_cfg, lab2_norm, lab2_split.masks["test"], selected_protocols, "zero_shot", device)
    all_metrics.append(m)
    all_sensor_summary.append(s)
    all_protocol_summary.append(p)

    print("Evaluating LAB2 head-only ...")
    m, s, p = _evaluate_checkpoint(helper, head_ckpt, phase_cfg, lab2_norm, lab2_split.masks["test"], selected_protocols, "head_only", device)
    all_metrics.append(m)
    all_sensor_summary.append(s)
    all_protocol_summary.append(p)

    print("Evaluating LAB2 full fine-tune ...")
    m, s, p = _evaluate_checkpoint(helper, full_ckpt, phase_cfg, lab2_norm, lab2_split.masks["test"], selected_protocols, "full_finetune", device)
    all_metrics.append(m)
    all_sensor_summary.append(s)
    all_protocol_summary.append(p)

    if partial_ckpt is not None:
        print("Evaluating LAB2 partial fine-tune ...")
        m, s, p = _evaluate_checkpoint(helper, partial_ckpt, phase_cfg, lab2_norm, lab2_split.masks["test"], selected_protocols, "partial_finetune", device)
        all_metrics.append(m)
        all_sensor_summary.append(s)
        all_protocol_summary.append(p)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    sensor_summary_df = pd.concat(all_sensor_summary, ignore_index=True)
    protocol_summary_df = pd.concat(all_protocol_summary, ignore_index=True)

    # Attach selected-protocol metadata for easier plotting / summaries.
    meta_cols = selected_df[["protocol_name", "active_count", "rank_within_k", "source_score", "source_r2_temperature", "source_r2_humidity"]].copy()
    protocol_summary_df = protocol_summary_df.merge(meta_cols, on="protocol_name", how="left")
    sensor_summary_df = sensor_summary_df.merge(meta_cols[["protocol_name", "active_count", "rank_within_k"]], on="protocol_name", how="left")
    metrics_df = metrics_df.merge(meta_cols[["protocol_name", "active_count", "rank_within_k"]], on="protocol_name", how="left")
    protocol_summary_df["active_total_nominal"] = protocol_summary_df["active_count"]

    metrics_df.to_csv(out_dir / str(cfg["combined_metrics_name"]), index=False)
    sensor_summary_df.to_csv(out_dir / str(cfg["combined_sensor_summary_name"]), index=False)
    protocol_summary_df.to_csv(out_dir / str(cfg["combined_protocol_summary_name"]), index=False)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\nGenerating figures ...")
    _plot_overview_bars(protocol_summary_df, out_dir / str(cfg["overview_figure_name"]), include_lab1_reference=bool(cfg["include_lab1_reference_in_subset_figures"]))
    _plot_sensor_map_4panel(sensor_summary_df, "temperature", out_dir / str(cfg["sensor_map_temperature_name"]))
    _plot_sensor_map_4panel(sensor_summary_df, "humidity", out_dir / str(cfg["sensor_map_humidity_name"]))
    _plot_subset_compare(protocol_summary_df, 5, out_dir / str(cfg["subset_compare_k5_name"]), include_lab1_reference=bool(cfg["include_lab1_reference_in_subset_figures"]))
    _plot_subset_compare(protocol_summary_df, 3, out_dir / str(cfg["subset_compare_k3_name"]), include_lab1_reference=bool(cfg["include_lab1_reference_in_subset_figures"]))
    _plot_subset_compare(protocol_summary_df, 2, out_dir / str(cfg["subset_compare_k2_name"]), include_lab1_reference=bool(cfg["include_lab1_reference_in_subset_figures"]))

    # ------------------------------------------------------------------
    # Summary outputs.
    # ------------------------------------------------------------------
    summary_json = {
        "phase1_run_dir": str(phase1_dir),
        "phase2_run_dir": str(phase2_dir),
        "output_dir": str(out_dir),
        "selected_protocols": json.loads(selected_df.to_json(orient="records")),
        "lab1_split_meta": lab1_split.meta,
        "lab2_split_meta": lab2_split.meta,
        "files": {
            "selected_protocols": str(out_dir / str(cfg["selected_protocols_name"])),
            "metrics": str(out_dir / str(cfg["combined_metrics_name"])),
            "sensor_summary": str(out_dir / str(cfg["combined_sensor_summary_name"])),
            "protocol_summary": str(out_dir / str(cfg["combined_protocol_summary_name"])),
            "overview_figure": str(out_dir / str(cfg["overview_figure_name"])),
            "sensor_map_temperature": str(out_dir / str(cfg["sensor_map_temperature_name"])),
            "sensor_map_humidity": str(out_dir / str(cfg["sensor_map_humidity_name"])),
            "subset_compare_k5": str(out_dir / str(cfg["subset_compare_k5_name"])),
            "subset_compare_k3": str(out_dir / str(cfg["subset_compare_k3_name"])),
            "subset_compare_k2": str(out_dir / str(cfg["subset_compare_k2_name"])),
        },
    }
    with open(out_dir / str(cfg["summary_json_name"]), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    _build_summary_markdown(selected_df, protocol_summary_df, out_dir / str(cfg["summary_md_name"]), include_lab1_reference=bool(cfg["include_lab1_reference_in_subset_figures"]))

    print("\nSaved artifacts")
    print(f"  Selected protocols : {out_dir / str(cfg['selected_protocols_name'])}")
    print(f"  Metrics            : {out_dir / str(cfg['combined_metrics_name'])}")
    print(f"  Sensor summary     : {out_dir / str(cfg['combined_sensor_summary_name'])}")
    print(f"  Protocol summary   : {out_dir / str(cfg['combined_protocol_summary_name'])}")
    print(f"  Figures directory  : {out_dir}")


if __name__ == "__main__":
    main()
