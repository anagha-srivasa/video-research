"""Blinded Track 1 reference milestone. No model promotion or Track 2 training."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REF=ROOT/'references/track1';WORK=ROOT/'work/track1_reference';REPORT=ROOT/'reports/track1_reference'
for path in [REF,WORK,REPORT]:path.mkdir(exist_ok=True)
