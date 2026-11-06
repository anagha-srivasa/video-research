import json
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT,atomic_json
from track1_retrain.common import save
from track2_feasibility.data import Data,BASE
from track1_encoder_revision.runtime import CompiledGeometry
from track1.numerics import basis

CFG=json.loads((ROOT/'configs/track2_geometry.json').read_text())
D=ROOT/'derived/track2_geometry';R=ROOT/'reports/track2_geometry';W=ROOT/'work/track2_geometry'


def descriptors(rings,roles,scale):
    """Translation-invariant endpoint shape summaries; no future target access."""
    if not rings or not np.isfinite(scale) or scale<=0:raise ValueError('Invalid geometry')
    points=np.concatenate(rings)*scale
    if not np.isfinite(points).all():raise ValueError('Nonfinite geometry')
    extent=np.ptp(points,axis=0);cov=np.cov(points.T,bias=True)
    areas=[abs(float(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))*.5))*scale**2 for p in rings]
    outer=sum(a for a,role in zip(areas,roles) if role=='outer');holes=sum(a for a,role in zip(areas,roles) if role=='hole')
    return np.array([np.log(max(extent[0],1e-8)),np.log(max(extent[1],1e-8)),np.log(max(outer-holes,1e-16)),
        cov[0,0]/max(np.trace(cov),1e-16),cov[0,1]/max(np.trace(cov),1e-16),
        holes/max(outer,1e-16),sum(role=='outer' for role in roles),sum(role=='hole' for role in roles)])


def prepare():
    data=Data();features=np.zeros((len(data.samples),4,8),np.float32);mask=np.zeros((len(data.samples),4),bool);ledger=[]
    for video in sorted({s['video_id'] for s in data.samples}):
        rows=pq.read_table(BASE/'functional'/f'{video}.m32.parquet').to_pylist();lookup={(r['entity_track_id'],r['window_end_us']):r for r in rows}
        a=np.load(BASE/'functional'/f'{video}.m32.geometry_coefficients.npy',mmap_mode='r');cache={}
        for i,s in enumerate(data.samples):
            if s['video_id']!=video:continue
            for j,t in enumerate(s['input_times_us']):
                key=s['entity_track_id'],t;r=lookup.get(key);reason=None
                if r is None or not r['geometry_training_eligible']:reason='missing_complete_certified_geometry'
                else:
                    if r['episode_id']!=s['episode_id'] or r['latest_input_us']>t:raise ValueError('Geometry episode or causality mismatch')
                    if key not in cache:
                        c=np.array(a[r['row_index']],np.float32);curves=CompiledGeometry(c,r,512).evaluate(t)
                        cache[key]=descriptors(curves,r['roles'],float(np.exp((basis(1.)@c)[-1])))
                    features[i,j]=cache[key];mask[i,j]=True
                ledger.append(dict(sequence_id=s['sequence_id'],video_id=video,split=s['split'],history_step=j,timestamp_us=t,
                    available=bool(mask[i,j]),reason=reason,source_epoch=r['epoch'] if r else None))
        a._mmap.close()
    np.save(D/'geometry.npy',features,allow_pickle=False);np.save(D/'geometry_mask.npy',mask,allow_pickle=False)
    save(R/'geometry_ledger.parquet',ledger)
    atomic_json(R/'support.json',dict(records=[dict(split=part,histories=len(data.indices(part,usable=False)),
        all_four_geometry=int(mask[data.indices(part,usable=False)].all(1).sum()),
        any_geometry=int(mask[data.indices(part,usable=False)].any(1).sum()),
        no_geometry=int((~mask[data.indices(part,usable=False)].any(1)).sum())) for part in ('train','validation')],
        missing_geometry_excluded=False,geometry_sampled_endpoint_only=True,no_contact_constraints=True,source_labels_retrospective=True))


def functional(data):
    x=data.x[:,:,:140].astype(float).reshape(-1,4,4,35)
    # Remove the absolute centroid from the constant Legendre mode only.
    x[:,:,0,:2]-=data.x[:,:,140:142]
    return x.reshape(-1,4,140)


def normalize(data,ids,geo,mask):
    from track2_kinematics.model import normalization
    if not len(ids) or any(data.split[ids]!='train'):raise ValueError('Training participants only')
    norm=normalization(data,ids);x=functional(data);seen=set();unique=[];g=[]
    for i in ids:
        for j,key in enumerate(data.samples[i]['input_state_ids']):
            if key in seen:continue
            seen.add(key);unique.append(x[i,j])
            if mask[i,j]:g.append(geo[i,j])
    x=np.array(unique);mean=x.mean(0);scale=np.maximum(x.std(0),.01)
    _,_,vh=np.linalg.svd((x-mean)/scale,full_matrices=False);projection=vh[:16]
    # Remove SVD sign ambiguity for checkpoint reproducibility.
    for row in projection:
        if row[np.argmax(abs(row))]<0:row*=-1
    if len(g)<16:raise ValueError('Insufficient training geometry for bounded ablation')
    g=np.array(g,float)
    norm.update(functional_mean=mean.tolist(),functional_scale=scale.tolist(),projection=projection.tolist(),
        geometry_mean=g.mean(0).tolist(),geometry_scale=np.maximum(g.std(0),.01).tolist(),unique_input_states=len(unique),geometry_states=len(g))
    return norm


def features(data,norm,geo,mask,family):
    from track2_kinematics.model import relative
    if family not in CFG['families']:raise ValueError('Unknown feature arm')
    out=np.zeros((len(data.samples),4,29),np.float32)
    out[:,:,:4]=relative(data.x[:,:,140:144])/np.array(norm['x_scale'])
    if family!='kinematics':out[:,:,4:20]=((functional(data)-norm['functional_mean'])/norm['functional_scale'])@np.array(norm['projection']).T
    if family in ('availability','geometry'):out[:,:,28]=mask
    if family=='geometry':
        # Mask before arithmetic: missing sentinel payloads have no influence.
        standardized=(np.where(mask[:,:,None],geo,np.array(norm['geometry_mean']))-norm['geometry_mean'])/norm['geometry_scale']
        out[:,:,20:28]=standardized
    if not np.isfinite(out).all():raise ValueError('Nonfinite features')
    return out
