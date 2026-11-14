"""Strictly causal adapter around the pinned official TAPNext++ implementation."""
from pathlib import Path
import sys
import time
import numpy as np
import psutil
import torch
from pipeline.common import ROOT, digest

VENDOR = ROOT / 'work/track1_observation_revision/tapnet_source'
CHECKPOINT = ROOT / 'work/track1_observation_revision/tapnextpp_512.ckpt'
EXPECTED_SIZE = 2532283010
EXPECTED_MD5 = 'eea4fffa043f4f28503c130d826b116c'


def resources():
    rss = psutil.Process().memory_info().rss
    available = psutil.virtual_memory().available
    gpu = torch.mps.driver_allocated_memory() if torch.backends.mps.is_available() else 0
    # Apple GPU allocations and RSS may overlap: report separately, not as a
    # purported unique-memory sum. Available memory is the system safety floor.
    if rss > 18 * 2**30 or gpu > 18 * 2**30 or available < 6 * 2**30:
        raise MemoryError(f'Neural tracker guard RSS={rss}, GPU={gpu}, available={available}')
    return dict(rss_gib=rss / 2**30, gpu_driver_gib=gpu / 2**30, available_gib=available / 2**30)


class Tracker:
    def __init__(self, device='mps'):
        if CHECKPOINT.stat().st_size != EXPECTED_SIZE or digest(CHECKPOINT, 'md5') != EXPECTED_MD5:
            raise ValueError('Checkpoint incomplete or publisher checksum mismatch')
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('Metal is unavailable in this process; do not silently benchmark CPU')
        resources()
        torch.set_num_threads(8)
        if device == 'mps':
            torch.mps.set_per_process_memory_fraction(.5)
        sys.path.insert(0, str(VENDOR))
        from tapnet.tapnextpp.votsp2026.model import TAPNextPP
        self.model = TAPNextPP.from_checkpoint(CHECKPOINT, device=device,
                                              half_precision=False, input_resolution=512)
        self.state = None
        self.last_timestamp = None
        self.shape = None
        self.device = device
        resources()

    def reset(self):
        self.state = None
        self.last_timestamp = None
        self.shape = None

    def step(self, rgb, timestamp_us, queries=None):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError('Expected uint8 RGB frame')
        if self.last_timestamp is not None and timestamp_us <= self.last_timestamp:
            raise ValueError('Frame timestamps must increase strictly')
        if self.shape is not None and rgb.shape != self.shape:
            raise ValueError('Native image dimensions changed')
        if self.state is None:
            q = np.asarray(queries, np.float32)
            if q.ndim != 2 or q.shape[1] != 2 or not len(q) or not np.isfinite(q).all():
                raise ValueError('Finite initialization points required')
            h, w = rgb.shape[:2]
            if (q < 0).any() or (q[:, 0] >= w).any() or (q[:, 1] >= h).any():
                raise ValueError('Query outside source image')
        elif queries is not None:
            raise ValueError('Silent reseeding is forbidden; explicitly reset the track')
        resources()
        start = time.perf_counter()
        with torch.inference_mode():
            xy, visible, state = self.model.track_frame(rgb[..., ::-1], queries, self.state, autocast=False)
        if self.device == 'mps':
            torch.mps.synchronize()
        if not np.isfinite(xy).all():
            raise FloatingPointError('Nonfinite neural point predictions')
        self.state, self.last_timestamp, self.shape = state, int(timestamp_us), rgb.shape
        return xy, visible, time.perf_counter() - start
