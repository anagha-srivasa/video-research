"""Open research result dashboard, source-bound artifact manifest and reproduction."""
import html,json
import numpy as np
from .common import *

def write():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 spatial=json.loads((REPORT/'spatial.json').read_text());appearance=json.loads((REPORT/'appearance.json').read_text());search=json.loads((REPORT/'search.json').read_text());functional=json.loads((REPORT/'functional.json').read_text());evaluation=json.loads((REPORT/'evaluation.json').read_text());verification=json.loads((REPORT/'verification.json').read_text());status=json.loads((OUT/'status.json').read_text())
 fig,axes=plt.subplots(1,3,figsize=(16,4.6));sv=[r for r in evaluation['spatial'] if r['split']=='validation'];axes[0].bar([r['method'] for r in sv],[r['boundary_fscore_2px'] for r in sv],color=['#8b96a9','#3699d5','#40b99b']);axes[0].set_title('Paired manual validation boundary F2');axes[0].set_ylim(0,1);axes[0].tick_params(axis='x',rotation=20)
 methods=['locked_v2_fitter','refit_cubic','refit_with_neutral_detail','nearest','linear'];protocols=['single','gap100','endpoint100'];x=np.arange(3)
 used=[]
 for method in methods:
  rs=[r for r in evaluation['temporal'] if r['split']=='validation' and r['model']==16 and r['method']==method]
  if rs:used.append((method,{r['protocol']:r['position_nrmse'] for r in rs}))
 for j,(method,values) in enumerate(used):axes[1].bar(x+(j-(len(used)-1)/2)*.15,[values.get(p,np.nan) for p in protocols],width=.15,label=method)
 axes[1].set_xticks(x,protocols);axes[1].set_title('M16 source-position holdout NRMSE');axes[1].legend(fontsize=6);axes[1].set_ylabel('Lower is better')
 rs=functional['videos'];labels=['M16','M32'];attempted=[sum(r['attempted'] for r in rs if r['model']==n) for n in [16,32]];supported=[sum(r['valid'] for r in rs if r['model']==n) for n in [16,32]];geometry=[sum(r['complete_certified_geometry'] for r in rs if r['model']==n) for n in [16,32]]
 axes[2].bar(np.arange(2)-.25,attempted,.25,label='all attempted');axes[2].bar(np.arange(2),supported,.25,label='numerically supported');axes[2].bar(np.arange(2)+.25,geometry,.25,label='complete + certified geometry');axes[2].set_xticks([0,1],labels);axes[2].set_title('Keep attempt and acceptance denominators');axes[2].legend(fontsize=7)
 fig.suptitle('Track 1 on dataset v2 · exploratory validation · source-contour inputs');fig.tight_layout();fig.savefig(REPORT/'quality.png',dpi=160,pil_kwargs={'compress_level':0});plt.close(fig)
 def table(records,columns):
  header=''.join('<th>'+html.escape(k)+'</th>' for k in columns);body=''.join('<tr>'+''.join('<td>'+html.escape(f'{r[k]:.6f}' if isinstance(r.get(k),float) else str(r.get(k)))+'</td>' for k in columns)+'</tr>' for r in records);return '<table><tr>'+header+'</tr>'+body+'</table>'
 counts=dict(observations=sum(r['observations'] for r in spatial['videos']),attempted=sum(r['attempted'] for r in functional['videos']),valid=sum(r['valid'] for r in functional['videos']),complete_certified_geometry=sum(r['complete_certified_geometry'] for r in functional['videos']))
 page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Track 1 dataset-v2 refit</title><style>body{font:16px system-ui;max-width:1350px;margin:32px auto;padding:20px;background:#101820;color:#e5edf4}a{color:#8ed6ff}p{line-height:1.6}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:9px;text-align:left;border-bottom:1px solid #43505d}img{max-width:100%}.status{background:#23384a;padding:16px}</style><h1>Track 1 refit on dataset v2</h1><p class="status">Open development release. Track 2 remains on hold. Numerical readiness and independent scientific acceptance are separate.</p>'''
 page+=f'<p>{counts["observations"]:,} eligible observations from 28,591 uncompressed native RGB frames. CPU float64 refits compare 16 controls/35 channels with bounded compound 32 controls/67 channels; four Legendre coefficients, one-second histories and quarter-second hops remain fixed. Frozen MobileNetV3-Small features are rebuilt for every eligible observation; separate source-masked object/context PCA projections use 128 dimensions and training participants only.</p>'
 page+='<img src="quality.png" alt="Paired spatial quality, temporal source holdouts and attempt denominators"><h2>Paired manual reconstruction</h2>'+table(sv,['method','masks','iou','boundary_fscore_2px'])
 page+='<p>The encoder receives the source contour. These are paired manual masks within the new partition, not the old full validation cohort or unseen RGB segmentation. The covariance and motion targets remain source-annotation measurements.</p><h2>Matched temporal reconstruction</h2>'+table([r for r in evaluation['temporal'] if r['split']=='validation'],['model','protocol','method','probes','position_nrmse','mask_iou'])
 page+='<p>Each model compares methods on identical retained observations within its supported cohort; M16 and M32 cohorts can differ. Shape holdout inputs are re-encoded after removing targets. Parameter choices use the five training participants only; the two repeatedly used validation participants remain exploratory. Interpolation keeps all retained knots and can outperform compact cubic compression. These tests do not establish online forecasting accuracy.</p>'
 final_model=json.loads((folder('calibration')/'model.json').read_text());family_rows=[]
 for n,model in final_model.items():
  for name,option in model['options']['families'].items():family_rows.append(dict(model=n,family=name,regularization=option['regularization'],loss=option['loss'],detail_count=model['detail']['count']))
 page+='<h2>Final training-only settings</h2>'+table(family_rows,['model','family','regularization','loss','detail_count'])+'<p>Shape settings minimize training-mask reconstruction error. Position/scale settings have per-training-video/protocol safeguards. These objective refinements followed exploratory validation diagnostics; this is not an untouched final evaluation. <a href="geometry_selection.json">Selection evidence</a>.</p>'
 page+='<h2>Full stored states and kinematics</h2>'+table(functional['videos'],['video_id','model','attempted','valid','certified_geometry','complete_certified_geometry','kinematic_supported'])
 page+='<p>Functional derivatives describe fitted history. Local constant velocity comes from a separate recent-observation estimator. Its covariance is conditional and uncalibrated. Optional endpoint-neutral detail affects positions inside history only and has zero endpoint position/velocity contribution. Geometry rollback affects a separate certified geometry view; its covariance is unavailable. Full masks are source-complete only when every retained source ring is represented.</p>'
 page+='<h2>Sequence availability</h2>'+table([dict(index=k,sequences=v) for k,v in verification['sequences'].items()],['index','sequences'])
 page+='<p>Sequences never cross missing hops or recorded source/correspondence epochs. Two-second clips limit longer sequences; zero-length sixteen-state indices remain empty instead of being fabricated. Provisional annotation-chain identity is not independently certified. Every paired source observation has a selection ledger.</p>'
 page+='<h2>Uncertainty and limits</h2>'+table(evaluation.get('uncertainty',[]),['split','model','source_position_marginal95_coverage'])
 page+='<p>Coverage is approximate source-position marginal coverage. It does not calibrate human label uncertainty, physical velocity, joint state consistency or detail/guarded geometry. Dense masks may depend on future annotated endpoints; no verified causal RGB measurement pipeline is claimed. Independent contact/identity/velocity references and fresh locked participants remain missing. No public VISOR benchmark parity or scientific freeze is claimed.</p><p><a href="search.json">Train-only search</a> · <a href="spatial.json">Spatial build</a> · <a href="appearance.json">Full appearance/PCA</a> · <a href="functional.json">Functional build</a> · <a href="evaluation.json">Matched evaluation</a> · <a href="verification.json">Numerical checks and sequence counts</a> · <a href="manifest.json">Artifact/source manifest</a></p></html>'
 (REPORT/'dashboard.html').write_text(page)
 doc=f'''# Track 1 retraining on dataset v2

This open development release is in `derived/track1_dataset_v2`. Numerical status: `derived/track1_dataset_v2/status.json`. Scientific acceptance remains false; Track 2 remains on hold.

{counts['observations']:,} selected source observations were refitted. There are {counts['attempted']:,} attempted model/window records, {counts['valid']:,} numerically supported states and {counts['complete_certified_geometry']:,} states with represented source completeness plus a decoder-bound continuous geometry certificate. The last condition concerns retained source geometry, not independently verified physical topology or identity.

The fixed-16 comparison stores `(4,35)` coefficients and `(4,68)` GL4 xi. The compound-32 ablation stores `(4,67)` coefficients and `(4,132)` xi, with ring allocation/roles/parents. Both use one-second histories and 0.25-second hops. Coefficients and stored tensors are float32; fitting/covariance calculations use CPU float64. Appearance stores separate `(4,2,128)` object/context coefficients; PCA and temporal selection use training participants only. The frozen CNN uses Metal. No neural backbone fine-tuning is part of this architecture.

The raw coefficients retain their fitted history derivatives. The separate local-CV kinematic state is authoritative for the declared source-position motion estimator. It is not independently calibrated physical motion. Endpoint-neutral history detail does not change the coarse endpoint or its velocity. Geometry coefficients and certificates form a separate view; no calibrated geometry covariance is transferred after rollback.

```python
from track1_retrain.store import Dataset
import pyarrow.parquet as pq
rows = pq.read_table('derived/track1_dataset_v2/functional/P02_122.m16.parquet').to_pylist()
row = next(r for r in rows if r['valid'])
with Dataset() as ds:
    state = ds.get_state(row['video_id'], row['entity_track_id'], row['window_end_us'], model=16)
    assert state['coefficients'].shape == (4,35)
    assert state['xi'].shape == (4,68)
```

Use `model=32` explicitly for the compound ablation. `require_geometry=True` additionally requires represented source completeness and a bound certificate. `ds.geometry(state, timestamp_us)` uses the certified normalized decoder within the history interval. Reader memory maps at most two video/model shards. Invalid states are rejected by default.

Sequence indices are split by participant and available at lengths 4 and 16. They never bridge missing hops or source/correspondence epochs. Current counts: {json.dumps(verification['sequences'])}. Do not assume a sixteen-state sequence exists in short clips.

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
 python3 -m track1_retrain.run --stage all
```

Stages: spatial, appearance, search, refine-selection, functional, evaluate, verify. CPU spatial chunks and complete appearance videos have hash-checked checkpoints. Temporary scripts/logs are in `work/track1_retrain`; uncompressed data/tensors and reports remain under this project root. Memory guards stop before an OOM; the controlled initial mmap-reader stop and subsequent fix remain in resource logs.

Read `reports/track1_retrain/dashboard.html` for paired reconstruction and support. Validation is exploratory; source-mask interpolation is retrospective, and source centroid motion can include occlusion/deformation/camera effects. Independent motion/contact/identity labels, fresh locked evaluation, full joint uncertainty, and long RGB sequence coverage remain open. Frozen v1/v2 artifacts remain unchanged.
'''
 (ROOT/'TRACK1_RETRAIN.md').write_text(doc)
 files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in OUT.rglob('*') if p.is_file()}
 atomic_json(REPORT/'manifest.json',dict(version=CFG['version'],state='open_not_finalized',dataset_manifest_sha256=digest(ROOT/'reports/dataset_v2/manifest.json'),code=code(),files=files,source_observation_mode='retrospective',scientific_acceptance=False,track2_started=False))
