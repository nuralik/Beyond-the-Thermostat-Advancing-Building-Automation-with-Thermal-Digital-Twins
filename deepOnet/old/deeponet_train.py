"""
DeepONet — Spatial Thermal Field Reconstruction, Lab1
=======================================================
  Data    : data/lab1_5min/Temperature_5min.xlsx  (all 13 sensors)
  Goal    : predict any sensor temperature from its (x, y) coordinate,
            given the other sensors as input (with masking augmentation).

  Masking augmentation
  --------------------
  For every (time t, target sensor s) training sample:
    - sensor s is always zeroed in the branch input (we are predicting it)
    - MASK_EXTRA additional sensors are randomly zeroed (robustness)

  Test split
  ----------
  8 non-contiguous 6-hour windows spread across the full date range.
  These timesteps are held out entirely — never seen during training or
  validation. Evaluation is in deeponet_eval.py.

  Validation split
  ----------------
  VAL_FRAC of the remaining (non-test) timesteps are held out for
  validation. Best model is saved based on validation loss.

Usage
-----
    python deeponet_train.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data' / 'lab1_5min'
OUT_DIR  = Path(__file__).parent

# ── Room & sensor geometry (Lab1) ─────────────────────────────────────────────
ROOM_W, ROOM_H = 7.81, 7.82
SENSOR_COORDS  = {
     1: (7.21, 0.60),  2: (3.88, 5.88),  3: (5.85, 5.88),
     4: (0.60, 0.60),  5: (1.92, 1.94),  6: (3.88, 3.91),
     7: (1.92, 3.91),  8: (5.85, 3.91),  9: (7.21, 7.22),
    10: (1.92, 5.88), 11: (0.60, 7.22), 12: (5.85, 1.94),
    13: (3.88, 1.94),
}
ALL_IDS = list(range(1, 14))   # sensors 1–13

# ── Test windows — 8 × 6-hour blocks, never seen by the model ─────────────────
# Each entry: (date YYYY-MM-DD, start_hour)
# Start hours are varied to cover morning / midday / evening / night
TEST_WINDOWS = [
    ('2022-06-15',  8),   # morning
    ('2022-06-20', 14),   # afternoon
    ('2022-07-06', 10),   # mid-morning
    ('2022-07-10', 20),   # evening
    ('2022-08-05',  6),   # early morning
    ('2022-08-25', 12),   # noon
    ('2022-09-04', 16),   # late afternoon
    ('2022-09-12', 22),   # night
]
TEST_HOURS = 6   # hours per window  →  8 × 6 = 48 h total

# ── Validation fraction (of non-test timesteps) ───────────────────────────────
VAL_FRAC = 0.15

# ── Hyper-parameters ──────────────────────────────────────────────────────────
P          = 128    # branch / trunk output width (dot-product dim)
HIDDEN     = 128    # hidden layer width
DEPTH      = 4      # layers per sub-network
LR         = 1e-3
BATCH      = 4096
EPOCHS     = 300
MASK_EXTRA = 1      # extra sensors randomly zeroed per sample (besides target)

DEVICE = (torch.device('mps')  if torch.backends.mps.is_available()  else
          torch.device('cuda') if torch.cuda.is_available()           else
          torch.device('cpu'))


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset  (masking augmentation)
# ─────────────────────────────────────────────────────────────────────────────
class DeepONetDataset(Dataset):
    """
    Each sample: (masked_branch, trunk_coord, target_value)

    mask_extra > 0  → training mode  (random extra sensors zeroed each call)
    mask_extra = 0  → validation mode (only target sensor zeroed, deterministic)

    Shapes
    ------
    branch : (N, 13)   coords : (13, 2)   labels : (N, 13)
    """
    def __init__(self, branch: np.ndarray, coords: np.ndarray,
                 labels: np.ndarray, mask_extra: int = MASK_EXTRA):
        self.branch     = torch.tensor(branch, dtype=torch.float32)
        self.coords     = torch.tensor(coords, dtype=torch.float32)
        self.labels     = torch.tensor(labels, dtype=torch.float32)
        self.N, self.S  = labels.shape
        self.mask_extra = mask_extra

    def __len__(self):
        return self.N * self.S

    def __getitem__(self, idx):
        t = idx // self.S
        s = idx  % self.S

        branch = self.branch[t].clone()
        branch[s] = 0.0   # always mask the target sensor

        if self.mask_extra > 0:
            others = [i for i in range(self.S) if i != s]
            extra  = np.random.choice(others, size=self.mask_extra, replace=False)
            branch[extra] = 0.0

        return branch, self.coords[s], self.labels[t, s]


# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────
def make_mlp(in_dim: int, hidden: int, out_dim: int, depth: int) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(depth - 2):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    def __init__(self, n_sensors: int = 13, coord_dim: int = 2,
                 hidden: int = HIDDEN, p: int = P, depth: int = DEPTH):
        super().__init__()
        self.branch = make_mlp(n_sensors, hidden, p, depth)
        self.trunk  = make_mlp(coord_dim, hidden, p, depth)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return (self.branch(u) * self.trunk(x)).sum(-1) + self.bias.squeeze()


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading & splitting
# ─────────────────────────────────────────────────────────────────────────────
def build_test_mask(time_idx: pd.DatetimeIndex) -> np.ndarray:
    """Returns boolean mask — True for every timestep inside a test window."""
    mask = np.zeros(len(time_idx), dtype=bool)
    for date_str, start_hour in TEST_WINDOWS:
        t0 = pd.Timestamp(f'{date_str} {start_hour:02d}:00:00')
        t1 = t0 + pd.Timedelta(hours=TEST_HOURS)
        mask |= ((time_idx >= t0) & (time_idx < t1)).values
    return mask


def load_data(val_seed: int = 42):
    """
    Returns
    -------
    branch_train, coords, labels_train  : training arrays (normalised)
    branch_val,   coords, labels_val    : validation arrays (normalised)
    stats      : dict(t_mean, t_std, scale)
    test_meta  : dict saved to disk for deeponet_eval.py
    """
    fpath = DATA_DIR / 'Temperature_5min.xlsx'
    scale = 100.0

    print(f"  Loading {fpath.name} ...", end=' ', flush=True)
    df       = pd.read_excel(fpath)
    time_idx = pd.to_datetime(pd.read_excel(DATA_DIR / 'Time_5min.xlsx')['tin'])
    print(f"{len(df):,} timesteps")

    # Stack all 13 sensor columns → (N, 13) in °C
    cols     = [f'tem{i}' for i in ALL_IDS]
    raw_full = np.stack([df[col].values for col in cols], axis=1) / scale

    # Drop NaN timesteps (gap between s0 and s1)
    valid_mask = ~np.isnan(raw_full).any(axis=1)
    raw      = raw_full[valid_mask]
    time_idx = time_idx[valid_mask].reset_index(drop=True)
    if (~valid_mask).sum():
        print(f"  Dropped {(~valid_mask).sum():,} NaN timesteps (data gap)")

    # ── Test mask (8 × 6-hour windows) ───────────────────────────────────────
    test_mask      = build_test_mask(time_idx)
    remaining_idx  = np.where(~test_mask)[0]   # indices not in test

    n_test = int(test_mask.sum())
    print(f"  Test windows   : {len(TEST_WINDOWS)} × {TEST_HOURS}h = {n_test:,} timesteps")

    # ── Validation split (random, from non-test only) ─────────────────────────
    rng     = np.random.default_rng(val_seed)
    n_val   = int(len(remaining_idx) * VAL_FRAC)
    val_idx = rng.choice(remaining_idx, size=n_val, replace=False)
    val_set = set(val_idx.tolist())

    train_idx = np.array([i for i in remaining_idx if i not in val_set])

    print(f"  Training steps : {len(train_idx):,}")
    print(f"  Val steps      : {len(val_idx):,}  ({VAL_FRAC*100:.0f}% of non-test)")

    # Normalise using training statistics only
    t_mean = float(raw[train_idx].mean())
    t_std  = float(raw[train_idx].std())
    raw_norm = (raw - t_mean) / t_std

    branch_train = raw_norm[train_idx]
    labels_train = raw_norm[train_idx]
    branch_val   = raw_norm[val_idx]
    labels_val   = raw_norm[val_idx]

    # Normalised coordinates
    coords = np.array([(SENSOR_COORDS[i][0] / ROOM_W,
                        SENSOR_COORDS[i][1] / ROOM_H) for i in ALL_IDS])  # (13, 2)

    stats = dict(t_mean=t_mean, t_std=t_std, scale=scale)

    # Metadata for eval script
    test_windows_meta = [
        {'date': d, 'start_hour': h,
         'start': f'{d} {h:02d}:00', 'end': f'{d} {h+TEST_HOURS:02d}:00'}
        for d, h in TEST_WINDOWS
    ]
    test_meta = dict(
        windows=test_windows_meta,
        test_hours=TEST_HOURS,
        test_indices=[int(i) for i in np.where(test_mask)[0]],
        n_test=n_test,
    )

    return branch_train, branch_val, coords, labels_train, labels_val, stats, test_meta


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(model: DeepONet, train_loader: DataLoader,
          val_loader: DataLoader) -> tuple:
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for u, x, y in train_loader:
            u, x, y = u.to(DEVICE), x.to(DEVICE), y.to(DEVICE)
            optimiser.zero_grad()
            loss = criterion(model(u, x), y)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(y)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for u, x, y in val_loader:
                u, x, y = u.to(DEVICE), x.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(u, x), y).item() * len(y)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step()

        # Save best model by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'train_loss': epoch_loss,
                'val_loss': best_val_loss,
            }, OUT_DIR / 'deeponet_best.pth')
            flag = '  *** best val ***'
        else:
            flag = ''

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:>4}/{EPOCHS}  "
              f"train={epoch_loss:.6f}  val={val_loss:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.0f}s{flag}")

    return train_losses, val_losses


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"  DeepONet  |  data=lab1_5min  |  device={DEVICE}")
    print(f"{'='*60}")

    # ── Load & split data ──────────────────────────────────────────────────
    print("\n[1] Loading & splitting data ...")
    branch_train, branch_val, coords, labels_train, labels_val, stats, test_meta = load_data()

    print(f"  Training pairs : {len(branch_train) * len(ALL_IDS):,}  "
          f"({len(ALL_IDS)} sensors × {len(branch_train):,} timesteps)")
    print(f"  Val pairs      : {len(branch_val) * len(ALL_IDS):,}")
    print(f"  Device         : {DEVICE}")

    # ── Save metadata for eval script ──────────────────────────────────────
    meta = dict(stats=stats, test_meta=test_meta,
                sensor_ids=ALL_IDS,
                sensor_coords=SENSOR_COORDS,
                room_dims=[ROOM_W, ROOM_H],
                data_dir=str(DATA_DIR))
    meta_path = OUT_DIR / 'deeponet_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved meta     : {meta_path}")

    # ── Datasets / DataLoaders ─────────────────────────────────────────────
    train_ds = DeepONetDataset(branch_train, coords, labels_train, mask_extra=MASK_EXTRA)
    val_ds   = DeepONetDataset(branch_val,   coords, labels_val,   mask_extra=0)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=0, pin_memory=(str(DEVICE) != 'cpu'))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                              num_workers=0, pin_memory=(str(DEVICE) != 'cpu'))

    # ── Model ──────────────────────────────────────────────────────────────
    print("\n[2] Building model ...")
    model = DeepONet(n_sensors=len(ALL_IDS)).to(DEVICE)
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\n[3] Training for {EPOCHS} epochs ...")
    train_losses, val_losses = train(model, train_loader, val_loader)

    # ── Save final model ───────────────────────────────────────────────────
    final_path = OUT_DIR / 'deeponet_final.pth'
    torch.save({
        'model_state': model.state_dict(),
        'epochs': EPOCHS,
        'final_train_loss': train_losses[-1],
        'final_val_loss':   val_losses[-1],
        'train_loss_curve': train_losses,
        'val_loss_curve':   val_losses,
    }, final_path)
    print(f"\n  Saved final model : {final_path}")
    print(f"  Best model (val)  : {OUT_DIR / 'deeponet_best.pth'}")
    print(f"\n  Run deeponet_eval.py to evaluate on the held-out test set.")
    print(f"  Done.\n")


if __name__ == '__main__':
    main()
