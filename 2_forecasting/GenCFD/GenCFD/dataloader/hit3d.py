import os
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset


def _as_bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y")


class HIT3D(Dataset):
    """
    Single-trajectory, sliding-window dataset for GenCFD (HIT, 128^3).

    Expects a single .npy file at {root}/trajectory.npy with shape:
        [C=4, T=204, X=128, Y=128, Z=128]  (dtype float32 recommended)

    Splits (half-open indices in original timeline):
        - train: [4, 164)  # skip first 4; 160 frames for training
        - val  : [164, 184)
        - test : [184, 200)

    Each item returns:
        - init_cond:   history of length lb, stacked along channels
                       shape [4*lb, 128, 128, 128]
        - lead_time:   scalar; either integer in [1..lf] or normalized to [0,1]
        - target_cond: frame at t_anchor + k (k = lead), shape [4, 128, 128, 128]
        - (optional) target_block: next lf frames, shape [lf, 4, 128, 128, 128]

    Normalization:
        - mean.npy/std.npy at {root} with per-channel stats over the *train window*.
        - If normalize_inputs=True, both init_cond and targets use the same z-score.

    Runtime knobs (env vars so you don’t have to modify train_gencfd):
        HIT3D_LB=<int>                 # default 1
        HIT3D_LF=<int>                 # default 20
        HIT3D_NORM_LEAD={0,1}          # default 1 (normalize to [0,1])
        HIT3D_STACK={0,1}              # keep for future; we stack channels by default
        HIT3D_RET_BLOCK={0,1}          # also return 'target_block' if 1
        HIT3D_TRAJ_PATH=<path>         # override {root}/trajectory.npy
        HIT3D_STATS_PATH=<path>        # where mean.npy/std.npy live (defaults to root)
    """

    # default split bounds (half-open): (start, end)
    DEFAULT_BOUNDS = {
        "train": (4, 164),
        "val":   (164, 204),
        "test":  (164, 204),
    }

    def __init__(
        self,
        root: str = "GenCFD/data/HIT3D_128",
        split: str = "train",
        padding_method: str = "circular",   # kept for GenCFD's CLI compatibility
        normalize_inputs: bool = True,
        lb: Optional[int] = None,           # lookback length
        lf: Optional[int] = None,           # forecast horizon
        bounds: Optional[Tuple[int, int]] = None,  # override half-open split bounds
        dtype = np.float32,
    ):
        self.root = Path(root)
        self.split = split
        self.padding_method = padding_method
        self.normalize_inputs = normalize_inputs
        self.dtype = dtype

        # Read knobs from env if not passed
        self.lb = lb if lb is not None else int(os.getenv("HIT3D_LB", "1"))
        self.lf = lf if lf is not None else int(os.getenv("HIT3D_LF", "20"))
        self.normalize_lead = _as_bool_env("HIT3D_NORM_LEAD", True)
        self.return_block   = _as_bool_env("HIT3D_RET_BLOCK", False)

        # Locate files
        print(f'root: {self.root}')
        traj_path = os.getenv("HIT3D_TRAJ_PATH", str(self.root / "trajectory.npy"))
        self.traj_path = Path(traj_path)
        if not self.traj_path.exists():
            raise FileNotFoundError(f"trajectory.npy not found at {self.traj_path}")

        stats_dir = Path(os.getenv("HIT3D_STATS_PATH", str(self.root)))
        mean_path = stats_dir / "MEAN.npy"
        std_path  = stats_dir / "STD.npy"
        if self.normalize_inputs:
            if not (mean_path.exists() and std_path.exists()):
                raise FileNotFoundError(
                    f"mean/std not found. Expected {mean_path} and {std_path}.\n"
                    f"Run the stats script provided in the README snippet."
                )
            self.mean = np.load(mean_path).astype(self.dtype)  # [4]
            self.std  = np.load(std_path).astype(self.dtype)   # [4]
            self.mean = self.mean.reshape(4)
            self.std  = self.std.reshape(4)
        else:
            self.mean = None
            self.std = None

        # Memory-map the trajectory (RAM friendly)
        self.traj = np.load(self.traj_path, mmap_mode="r")  # [4, 204, 128,128,128]
        if self.traj.ndim != 5 or self.traj.shape[0] != 4 or self.traj.shape[1] < 50:
            raise ValueError(f"Bad trajectory shape {self.traj.shape}; expected [4, 204, 128,128,128].")

        # Bounds
        if bounds is None:
            if self.split not in self.DEFAULT_BOUNDS:
                raise ValueError(f"Unknown split '{self.split}'.")
            start, end = self.DEFAULT_BOUNDS[self.split]
        else:
            start, end = bounds

        T = self.traj.shape[1]
        if not (0 <= start < end <= T):
            raise ValueError(f"Invalid bounds {start}:{end} for T={T}.")

        self.start, self.end = start, end  # half-open

        # Pre-compute valid anchors s such that:
        # we can take lb history ending at s, and *any* lead in [1..lf] stays in-bounds.
        # History frames: [s-lb+1, ..., s]
        # Targets up to s + lf
        first_anchor = start + self.lb - 1
        last_anchor  = (end - 1) - self.lf
        if last_anchor < first_anchor:
            raise ValueError(
                f"Window {start}:{end} too short for lb={self.lb}, lf={self.lf}."
            )
        self.anchors: List[int] = list(range(first_anchor, last_anchor + 1))

        self.input_channel = 4 * self.lb
        _, _, nx, ny, nz = self.traj.shape
        self.output_shape       = (nx, ny, nz)
        self.spatial_resolution = (nx, ny, nz)
        self.output_channel = 4



    def __len__(self) -> int:
        return len(self.anchors)

    def _z(self, arr: np.ndarray) -> np.ndarray:
        # arr: [C, X, Y, Z]
        if not self.normalize_inputs:
            return arr.astype(self.dtype)
        return ((arr - self.mean[:, None, None, None]) /
                (self.std[:,  None, None, None] + 1e-6)).astype(self.dtype)

    def _get_lead(self) -> int:
        # Uniform lead in [1..lf]
        return int(np.random.randint(1, self.lf + 1))

    def __getitem__(self, idx: int):
        s = self.anchors[idx]         # anchor time index (end of history)
        k = self._get_lead()          # lead

        # History indices [s-lb+1 ... s], each -> [4, X, Y, Z]
        hist = self.traj[:, s - self.lb + 1 : s + 1]  # [4, lb, X, Y, Z]
        # Stack history along channels to keep model unchanged:
        # -> [4*lb, X, Y, Z]
        hist_stacked = np.concatenate(
            [hist[:, i] for i in range(self.lb)], axis=0
        )

        # Target at lead k
        tgt = self.traj[:, s + k]  # [4, X, Y, Z]

        # Optional future block of length lf
        if self.return_block:
            block = self.traj[:, s + 1 : s + 1 + self.lf]  # [4, lf, X, Y, Z]
            block = np.moveaxis(block, 1, 0)               # [lf, 4, X, Y, Z]
            block = np.stack([self._z(block[i]) for i in range(self.lf)], axis=0)
        else:
            block = None

        # Normalize (same stats for history & targets)
        init_cond = self._z(hist_stacked)   # [4*lb, 128,128,128]
        target    = self._z(tgt)            # [4,    128,128,128]

        # lead_time scalar
        if self.normalize_lead:
            lead_time = float(k / self.lf)
        else:
            lead_time = float(k)

        # sample = {
        #     "init_cond":   torch.from_numpy(init_cond).float(),
        #     "lead_time":   torch.tensor(lead_time, dtype=torch.float32),
        #     "target_cond": torch.from_numpy(target).float(),
        #     # for convenience/debugging:
        #     "lead_index":  torch.tensor(k, dtype=torch.int32),
        # }
        sample = {
            # GenCFD’s denoising_model.py expects this exact key
            "initial_cond": torch.from_numpy(init_cond).float(),   # [in_channels, 128,128,128]

            # Time-conditioning key name used across the codebase
            "lead_time":    torch.tensor(lead_time, dtype=torch.float32),  # scalar

            # Likely-consumed target key (naming varies by dataset);
            # we provide both "target" and "target_cond" for compatibility.
            "target":       torch.from_numpy(target).float(),      # [4, 128,128,128]
            "target_cond":  torch.from_numpy(target).float(),

            # for convenience/debugging (not used by trainer)
            "lead_index":   torch.tensor(k, dtype=torch.int32),
        }


        if block is not None:
            sample["target_block"] = torch.from_numpy(block).float()  # [lf, 4, ...]
        return sample
