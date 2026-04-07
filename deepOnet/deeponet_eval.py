"""
DeepONet — Evaluation on held-out test set
===========================================
Loads the best trained model and evaluates it on the 3-day test window
that was never seen during training.

For each sensor s (1–13):
  - Branch input = all 13 sensor values at time t, with sensor s masked to 0
  - Trunk input  = normalised (x, y) coordinate of sensor s
  - Prediction   = reconstructed temperature at sensor s

Outputs
-------
  deeponet_eval_metrics.csv        — RMSE / MAE / R² per sensor
  deeponet_eval_timeseries.png     — predicted vs actual time series (all sensors)
  deeponet_eval_scatter.png        — scatter + fit per sensor
  deeponet_eval_floormap.png       — RMSE and MAE scores on the room floor map

Usage
-----
    python deeponet_eval.py
    python deeponet_eval.py --model deeponet_final.pth   # use final instead of best
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.interpolate import RBFInterpolator

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
#  Model  (must match train script)
# ─────────────────────────────────────────────────────────────────────────────
def make_mlp(in_dim, hidden, out_dim, depth):
    layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(depth - 2):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    def __init__(self, n_sensors=13, coord_dim=2, hidden=128, p=128, depth=4):
        super().__init__()
        self.branch = make_mlp(n_sensors, hidden, p, depth)
        self.trunk  = make_mlp(coord_dim, hidden, p, depth)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, u, x):
        return (self.branch(u) * self.trunk(x)).sum(-1) + self.bias.squeeze()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_sensor(model, branch_norm, coord, device, chunk=8192):
    """Predict one sensor: branch_norm has that sensor already zeroed."""
    model.eval()
    u   = torch.tensor(branch_norm, dtype=torch.float32, device=device)
    x   = torch.tensor(coord,       dtype=torch.float32, device=device).expand(len(u), -1)
    out = []
    for i in range(0, len(u), chunk):
        out.append(model(u[i:i+chunk], x[i:i+chunk]).cpu().numpy())
    return np.concatenate(out)


def r2(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main(model_file='deeponet_best.pth'):
    device = (torch.device('mps')  if torch.backends.mps.is_available()  else
              torch.device('cuda') if torch.cuda.is_available()           else
              torch.device('cpu'))

    print(f"\n{'='*60}")
    print(f"  DeepONet Evaluation  |  model={model_file}  |  device={device}")
    print(f"{'='*60}")

    # ── Load metadata ──────────────────────────────────────────────────────
    meta_path = OUT_DIR / 'deeponet_meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found — run deeponet_train.py first.")
    with open(meta_path) as f:
        meta = json.load(f)

    stats        = meta['stats']
    test_meta    = meta['test_meta']
    sensor_ids   = meta['sensor_ids']           # [1..13]
    sensor_coords= {int(k): v for k, v in meta['sensor_coords'].items()}
    room_w, room_h = meta['room_dims']
    data_dir     = Path(meta['data_dir'])
    t_mean       = stats['t_mean']
    t_std        = stats['t_std']
    scale        = stats['scale']
    test_indices = test_meta['test_indices']     # row indices in the full dataset

    print(f"\n  Test windows ({len(test_meta['windows'])}):")
    for w in test_meta['windows']:
        print(f"    {w['start']}  →  {w['end']}")
    print(f"  Total test steps : {test_meta['n_test']:,}")

    # ── Load test data ─────────────────────────────────────────────────────
    print("\n[1] Loading test data ...")
    fpath    = data_dir / 'Temperature_5min.xlsx'
    df_full  = pd.read_excel(fpath)
    time_idx = pd.to_datetime(pd.read_excel(data_dir / 'Time_5min.xlsx')['tin'])

    # Drop NaN timesteps (gap between s0 and s1) — must match train script
    cols     = [f'tem{i}' for i in sensor_ids]
    raw_full = np.stack([df_full[col].values for col in cols], axis=1) / scale
    valid    = ~np.isnan(raw_full).any(axis=1)
    raw_all  = raw_full[valid]
    time_idx = time_idx[valid].reset_index(drop=True)

    time_test = time_idx.iloc[test_indices].reset_index(drop=True)

    # Raw test array (N_test, 13) in °C
    raw_test  = raw_all[test_indices]
    norm_test = (raw_test - t_mean) / t_std    # normalised

    # Outdoor temperature for test indices
    outdoor_full = df_full['temE'].values[valid]
    outdoor_test = outdoor_full[test_indices].astype(float)

    coords_norm = np.array([(sensor_coords[i][0] / room_w,
                              sensor_coords[i][1] / room_h) for i in sensor_ids])  # (13, 2)

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\n[2] Loading model from {model_file} ...")
    ckpt  = torch.load(OUT_DIR / model_file, map_location=device)
    model = DeepONet(n_sensors=len(sensor_ids)).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    val_loss_str = f"{ckpt['val_loss']:.6f}" if 'val_loss' in ckpt else '?'
    print(f"  Loaded (epoch={ckpt.get('epoch', '?')}, val_loss={val_loss_str})")

    # ── Predict each sensor ────────────────────────────────────────────────
    print("\n[3] Predicting all sensors on test set ...")
    records = []
    preds   = {}   # sensor_id -> predicted °C array
    actuals = {}   # sensor_id -> actual °C array

    for s_idx, s_id in enumerate(sensor_ids):
        # Mask target sensor in branch
        branch_masked = norm_test.copy()
        branch_masked[:, s_idx] = 0.0

        pred_norm = predict_sensor(model, branch_masked,
                                   coords_norm[s_idx:s_idx+1], device)
        pred_c  = pred_norm * t_std + t_mean      # denormalise → °C
        actual_c = raw_test[:, s_idx]

        rmse = float(np.sqrt(np.mean((pred_c - actual_c) ** 2)))
        mae  = float(np.mean(np.abs(pred_c - actual_c)))
        r2_  = float(r2(actual_c, pred_c))

        preds[s_id]   = pred_c
        actuals[s_id] = actual_c
        records.append(dict(sensor=s_id, RMSE=rmse, MAE=mae, R2=r2_))
        print(f"  Sensor {s_id:>2} :  RMSE={rmse:.4f}°C  MAE={mae:.4f}°C  R²={r2_:.4f}")

    # ── Save metrics CSV ───────────────────────────────────────────────────
    metrics_df = pd.DataFrame(records).set_index('sensor')
    csv_path   = OUT_DIR / 'deeponet_eval_metrics.csv'
    metrics_df.to_csv(csv_path)
    print(f"\n  Saved metrics : {csv_path}")
    print(f"\n  ── Summary ──")
    print(f"  Mean RMSE : {metrics_df['RMSE'].mean():.4f}°C")
    print(f"  Mean MAE  : {metrics_df['MAE'].mean():.4f}°C")
    print(f"  Mean R²   : {metrics_df['R2'].mean():.4f}")

    # ── Time-series plot (all 13 sensors) ──────────────────────────────────
    print("\n[4] Saving plots ...")
    n_cols = 3
    n_rows = (len(sensor_ids) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.8), sharex=True)
    axes = axes.flatten()

    x_axis      = np.arange(len(raw_test))
    n_per_window = test_meta['test_hours'] * 12   # 6 h × 12 samples/h = 72

    def hour_to_period(h):
        if   h <  6: return 'Night'
        elif h <  8: return 'Early AM'
        elif h < 12: return 'Morning'
        elif h < 14: return 'Noon'
        elif h < 17: return 'Afternoon'
        elif h < 21: return 'Evening'
        else:        return 'Night'

    # Dual x-axis ticks: sample index + short date + time-of-day period
    tick_pos    = [i * n_per_window for i in range(len(test_meta['windows']))]
    tick_labels = []
    for i, w in enumerate(test_meta['windows']):
        date_short = pd.Timestamp(w['start']).strftime('%b %-d')
        period     = hour_to_period(w['start_hour'])
        tick_labels.append(f"{i * n_per_window}\n{date_short}\n{period}")
    tick_pos.append(len(raw_test))
    tick_labels.append(str(len(raw_test)))

    for s_idx, s_id in enumerate(sensor_ids):
        ax = axes[s_idx]

        # ── Outdoor temperature shading (right y-axis) ────────────────────
        ax_r = ax.twinx()
        ax_r.fill_between(x_axis, outdoor_test, alpha=0.12, color='orange')
        ax_r.plot(x_axis, outdoor_test, color='darkorange', lw=0.6, alpha=0.5,
                  label='Outdoor T')
        ax_r.tick_params(axis='y', labelsize=6, labelcolor='darkorange')
        ax_r.set_ylim(outdoor_test.min() - 2, outdoor_test.max() + 2)
        # Only label the right axis on the rightmost column
        if (s_idx + 1) % n_cols == 0 or s_idx == len(sensor_ids) - 1:
            ax_r.set_ylabel('Outdoor T (°C)', fontsize=7, color='darkorange')
        else:
            ax_r.set_yticklabels([])

        # ── Indoor actual vs predicted (left y-axis) ──────────────────────
        ax.plot(x_axis, actuals[s_id], color='tomato',    lw=0.8, alpha=0.8,
                label='Actual', zorder=3)
        ax.plot(x_axis, preds[s_id],   color='steelblue', lw=0.8, alpha=0.9,
                label='Predicted', zorder=3)
        ax.set_facecolor('none')   # let outdoor shading show through
        r = records[s_idx]
        ax.set_title(f"Sensor {s_id}  |  RMSE={r['RMSE']:.3f}°C  R²={r['R2']:.3f}",
                     fontsize=9)
        ax.set_ylabel('Indoor T (°C)', fontsize=8)
        ax.tick_params(axis='both', labelsize=7)

        # Vertical lines at window boundaries
        for wp in tick_pos[1:-1]:
            ax.axvline(wp, color='gray', lw=0.5, ls='--', alpha=0.5, zorder=1)

        if s_idx == 0:
            lines_l, labels_l = ax.get_legend_handles_labels()
            lines_r, labels_r = ax_r.get_legend_handles_labels()
            ax.legend(lines_l + lines_r, labels_l + labels_r, fontsize=7)

    # Apply dual x-axis labels to all bottom-row axes
    for ax in axes:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=6.5)
        ax.set_xlabel('Sample index  /  Date–time', fontsize=7)

    # Hide unused subplots
    for ax in axes[len(sensor_ids):]:
        ax.set_visible(False)

    fig.suptitle(f'DeepONet — Test Set Reconstruction  '
                 f'({len(test_meta["windows"])} windows × {test_meta["test_hours"]}h = '
                 f'{test_meta["n_test"]:,} timesteps)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    ts_path = OUT_DIR / 'deeponet_eval_timeseries.png'
    plt.savefig(ts_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved : {ts_path}")

    # ── Scatter plot (all 13 sensors) ──────────────────────────────────────
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for s_idx, s_id in enumerate(sensor_ids):
        ax  = axes[s_idx]
        act = actuals[s_id]
        prd = preds[s_id]

        ax.scatter(act, prd, s=3, alpha=0.3, color='steelblue', rasterized=True)

        lims = [min(act.min(), prd.min()) - 0.5, max(act.max(), prd.max()) + 0.5]
        ax.plot(lims, lims, 'r--', lw=1.2, label='Ideal')

        coeffs = np.polyfit(act, prd, 1)
        x_line = np.linspace(lims[0], lims[1], 200)
        ax.plot(x_line, np.polyval(coeffs, x_line), 'k-', lw=1.2,
                label=f'Fit: {coeffs[0]:.2f}x+{coeffs[1]:.2f}')

        r = records[s_idx]
        ax.set_title(f"Sensor {s_id}  RMSE={r['RMSE']:.3f}°C  R²={r['R2']:.3f}", fontsize=9)
        ax.set_xlabel('Actual T (°C)', fontsize=8)
        ax.set_ylabel('Predicted T (°C)', fontsize=8)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    for ax in axes[len(sensor_ids):]:
        ax.set_visible(False)

    fig.suptitle('DeepONet — Predicted vs Actual (Test Set, per Sensor)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    sc_path = OUT_DIR / 'deeponet_eval_scatter.png'
    plt.savefig(sc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved : {sc_path}")

    # ── Floor map: RMSE and MAE per sensor location ────────────────────────
    GATEWAY  = (1.63, 7.73)
    GRID_RES = 300
    gx = np.linspace(0, room_w, GRID_RES)
    gy = np.linspace(0, room_h, GRID_RES)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

    coords_m = np.array([sensor_coords[i] for i in sensor_ids])   # (13, 2) in metres
    rmse_vals = np.array([r['RMSE'] for r in records])
    mae_vals  = np.array([r['MAE']  for r in records])
    r2_vals   = np.array([r['R2']   for r in records])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'DeepONet — Reconstruction Error on Floor Map\n'
        f'{len(test_meta["windows"])} × {test_meta["test_hours"]}h test windows  '
        f'({test_meta["n_test"]:,} timesteps)',
        fontsize=12, fontweight='bold'
    )

    # RdYlGn: red=bad, green=good
    # RMSE/MAE: low is good → reverse the colormap so low=green, high=red
    panels = [
        ('RMSE (°C)',  rmse_vals, 'RdYlGn_r'),
        ('MAE (°C)',   mae_vals,  'RdYlGn_r'),
        ('R²',         r2_vals,   'RdYlGn'),
    ]

    for ax, (label, values, cmap) in zip(axes, panels):
        rbf = RBFInterpolator(coords_m, values, kernel='thin_plate_spline', smoothing=0)
        Z   = rbf(grid_pts).reshape(GX.shape)

        im = ax.imshow(Z, origin='lower', extent=[0, room_w, 0, room_h],
                       cmap=cmap, aspect='equal', interpolation='bilinear')

        # Sensor dots coloured by their score
        sc = ax.scatter(coords_m[:, 0], coords_m[:, 1],
                        c=values, cmap=cmap, vmin=Z.min(), vmax=Z.max(),
                        s=160, edgecolors='black', linewidths=0.9, zorder=5)

        # Sensor number + score labels
        for sid, (x, y), v in zip(sensor_ids, coords_m, values):
            ax.annotate(f'{sid}\n{v:.3f}', (x, y),
                        textcoords='offset points', xytext=(6, 4),
                        fontsize=7, fontweight='bold', color='white', zorder=6,
                        path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        # Gateway marker
        ax.plot(*GATEWAY, marker='^', color='white', markersize=9,
                markeredgecolor='black', markeredgewidth=0.9, zorder=6)
        ax.annotate('GW', GATEWAY, textcoords='offset points',
                    xytext=(5, 4), fontsize=7.5, fontweight='bold', color='white',
                    zorder=6, path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        # Room boundary
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_edgecolor('black')

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label, fontsize=9)

        ax.set_title(label, fontsize=11)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlim(0, room_w)
        ax.set_ylim(0, room_h)
        ax.set_xticks(np.arange(0, room_w + 1, 2))
        ax.set_yticks(np.arange(0, room_h + 1, 2))

    plt.tight_layout()
    fm_path = OUT_DIR / 'deeponet_eval_floormap.png'
    plt.savefig(fm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved : {fm_path}")

    print("\n  Done.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='deeponet_best.pth',
                        help='Model file inside deepOnet/ (default: deeponet_best.pth)')
    args = parser.parse_args()
    main(model_file=args.model)
