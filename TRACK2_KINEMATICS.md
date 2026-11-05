# Kinematics-only Track 2 prototype

The bounded six-run extension is complete. Kinematic GRU/MLP predictors improve on CV but do not beat constant position across the required horizons. Full Track 2 remains on hold.

Read [results and report corrections](reports/track2_kinematics/RESULTS.md) or open [the dashboard](reports/track2_kinematics/dashboard.html).

Code: `src/track2_kinematics`. Locked configuration: `configs/track2_kinematics.json`. Models: `derived/track2_kinematics`. Scratch/fold records: `work/track2_kinematics`. The frozen Track 1 and original feasibility artifacts are preserved.

```python
from track2_kinematics.fit import load
from track2_kinematics.model import predict

model, norm, checkpoint = load(
    'derived/track2_kinematics/models/kinematic_mlp.17.pt'
)
# history: (batch, 4, 4), finite intrinsic-normalized x,y,vx,vy.
# Caller must enforce consecutive 250 ms hops, causal input support,
# and the same source episode/epoch, as the training adapter does.
# output = predict(model, norm, history)  # (batch, 3, 4): 250/500/1000 ms.
```

This is an exploratory candidate API, not a default training release or independently validated physical-motion predictor. It outputs no calibrated uncertainty, contact events or geometry. Model instances loaded through this API are CPU float32; numerical postprocessing uses float64.
