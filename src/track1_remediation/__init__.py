"""Bounded Track 1 candidates. No release replacement or Track 2 training."""
from pathlib import Path
from track1_retrain.common import ROOT,VIDEOS,CFG,split,save,array,LOG,pool,Monitor,guard
from pipeline.common import atomic_json,digest
R=ROOT/'reports/track1_remediation';D=ROOT/'derived/track1_remediation';W=ROOT/'work/track1_remediation'
SOURCES={'v2':ROOT/'derived/track1_dataset_v2','context':ROOT/'derived/track1_context_v3'}
for p in [R,D,W]:p.mkdir(exist_ok=True)
