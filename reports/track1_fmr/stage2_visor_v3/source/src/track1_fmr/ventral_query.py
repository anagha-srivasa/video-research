"""Explicit structured filtering and lazy, caller-supplied semantic verification."""
import numpy as np
from .contracts import EpistemicTag


def query(tubes, *, minimum_red=None, verifier=None, label=None, max_candidates=5):
    if max_candidates < 1 or (minimum_red is not None and not np.isfinite(minimum_red)):
        raise ValueError('Invalid query bounds')
    if (verifier is None) != (label is None):
        raise ValueError('A label requires a verifier and vice versa')
    matches = []
    for tube in tubes:
        if minimum_red is not None and tube.algebraic_sig[5] < minimum_red:
            continue
        matches.append(tube)
    # Truncation is explicit: discarded candidates make completeness unknown.
    truncated = len(matches) > max_candidates
    chosen = matches[:max_candidates]
    results = []
    for tube in chosen:
        tag = tube.epistemic_tag
        score = None
        if verifier is not None and tube.best_view_patch is not None:
            score = float(verifier(tube.best_view_patch, label))
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Verifier must return a finite score in [0,1]')
        results.append(dict(track_id=tube.track_id, start=tube.t_start, end=tube.t_end,
                            evidence=tag.value, semantic_score=score,
                            semantic_label=label if score is not None else None,
                            physical_contact=None))
    return dict(candidates=results, total_candidates=len(matches), truncated=truncated,
                identity_certified=False, contact_certified=False)


class YOLOEVerifier:
    """Lazy optional YOLOE adapter; the checkpoint must already exist locally."""
    def __init__(self, checkpoint, device=None):
        from pathlib import Path
        import os
        from pipeline.common import ROOT
        path = Path(checkpoint).resolve()
        if not path.is_file() or not path.is_relative_to(ROOT):
            raise FileNotFoundError('Supply a local project-scoped YOLOE checkpoint')
        os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'work/track1_fmr_config'))
        os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
        from ultralytics import YOLOE, settings
        import torch
        settings.update({'weights_dir': str(ROOT / 'work')})
        self.model = YOLOE(str(path))
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device)
        self.weights_dir = ROOT / 'work'
        self.label = None

    def __call__(self, patch, label):
        if label != self.label:
            if not (self.weights_dir / 'mobileclip2_b.ts').is_file():
                raise FileNotFoundError('Locally acquired MobileCLIP2 encoder required')
            self.model.set_classes([label])
            self.label = label
        result = self.model.predict(source=patch[..., ::-1], device=self.device, verbose=False)[0]
        return float(result.boxes.conf.max()) if len(result.boxes) else 0.
