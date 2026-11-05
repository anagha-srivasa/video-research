"""Frame-amortised frozen MobileNet pooling and train-only 128-D PCA projection."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json,sys,subprocess,time,logging
import numpy as np
import torch
import pyarrow.parquet as pq
import duckdb
import queue,threading
from pipeline.common import ROOT,digest,atomic_json
from .artifacts import output_dir,load_shard,save_shard,manifest,memory_guard
from .numerics import spline_basis
from .spatial import soft_masks

LOG=logging.getLogger('track1')

def backbone():
    from torchvision.models import mobilenet_v3_small
    weight=ROOT/'work/track1_weights/mobilenet_v3_small-047dcff4.pth'
    if not digest(weight).startswith('047dcff4'):raise ValueError('Official MobileNet weight SHA256 prefix mismatch')
    torch.manual_seed(41);model=mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True));model.eval();model.requires_grad_(False)
    return model.features

def read_exact(stream,n):
    data=bytearray()
    while len(data)<n:
        chunk=stream.read(n-len(data))
        if not chunk:raise RuntimeError('Decoder pipe closed during frame payload')
        data.extend(chunk)
    return data

def frame_stream(cfg,video,timestamps):
    with duckdb.connect() as db:
        mapping=dict(db.execute('SELECT pts_raw,timestamp_us FROM read_parquet(?) WHERE timestamp_us IN (SELECT unnest(?))',[str(ROOT/cfg['input']/f'frames.{video}.parquet'),list(map(int,timestamps))]).fetchall())
    request=ROOT/f'work/track1_decode.{video}.json'
    atomic_json(request,dict(path=str(ROOT/f'dataset/videos/{video}.MP4'),pts=mapping,width=448,height=256))
    process=subprocess.Popen([sys.executable,'-m','track1.decoder',str(request)],stdout=subprocess.PIPE)
    try:
        while True:
            line=process.stdout.readline()
            if not line:break
            header=json.loads(line);rgb=np.frombuffer(read_exact(process.stdout,header['bytes']),np.uint8).reshape(header['shape'])
            yield header['timestamp_us'],rgb
        if process.wait()!=0:raise RuntimeError('Sequential hardware decode failed: '+video)
    finally:
        if process.poll() is None:process.terminate();process.wait()

def extract_video(cfg,choice,model,device,monitor):
    video=choice['video_id'];out=ROOT/'work/track1_features';out.mkdir(exist_ok=True)
    features_path=out/f'{video}.npy';valid_path=out/f'{video}.valid.npy'
    spatial=output_dir(cfg,'spatial');rows=pq.read_table(spatial/f'{video}.parquet').to_pylist();s=load_shard(spatial/f'{video}.pt')
    calibration=json.loads((output_dir(cfg,'calibration')/'model.json').read_text());by_time=defaultdict(list)
    for row in rows:
        if row['spatial_valid']:by_time[row['timestamp_us']].append(row)
    # Resumable video-level checkpoint; transient full feature maps never reach disk.
    checkpoint=out/f'{video}.json';fingerprint=dict(spatial_sha256=digest(spatial/f'{video}.pt'),calibration_sha256=digest(output_dir(cfg,'calibration')/'model.json'),weights_sha256=digest(ROOT/'work/track1_weights/mobilenet_v3_small-047dcff4.pth'))
    if checkpoint.exists() and json.loads(checkpoint.read_text()).get('inputs')==fingerprint and features_path.exists() and valid_path.exists():
        LOG.info('%s appearance pooled feature checkpoint reused',video);return
    temp=features_path.with_suffix('.npy.tmp');pooled=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=(len(rows),576));pooled[:]=0
    valid=np.zeros(len(rows),bool);mass=np.zeros((len(rows),2),np.float32)
    phi=spline_basis();start=time.monotonic();done=0;batch=[]
    holes={}
    with duckdb.connect() as db:
        for uid,rings,aw,ah in db.execute('SELECT mask_uid,contours_json,annotation_width,annotation_height FROM read_parquet(?) WHERE hole_count>0',[str(ROOT/cfg['input']/f'masks.{video}.parquet')]).fetchall():
            holes[uid]=[np.asarray(r['points'],float)*[rows[0]['width']/aw,rows[0]['height']/ah] for r in json.loads(rings) if r['role']=='hole']
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    def pool_mask(row,h,w):
        idx=row['row_index'];curve=phi@s['controls'][idx]
        x=(np.arange(w)+.5)*row['width']/w;y=(np.arange(h)+.5)*row['height']/h
        grid=np.stack(np.meshgrid(x,y),axis=-1).reshape(-1,2)
        sigma=np.sqrt(s['normal_variance'][idx]*calibration['kappa_cov'])
        core,halo=soft_masks(curve,sigma,grid,tau=max(row['width']/448,1.))
        # A dominant outer spline cannot encode source holes. Remove them explicitly
        # from appearance support, with the same conservative uncertainty margin.
        for hole in holes.get(row['mask_uid'],[]):
            hole_core,hole_halo=soft_masks(hole,np.full(len(hole),max(float(np.median(sigma)),1.)),grid,tau=max(row['width']/448,1.))
            core*=1-hole_halo;halo*=1-hole_core
        return idx,core.astype(np.float32),float(core.sum()),float(halo.sum())
    def process_batch(items,executor):
        rgb=np.stack([x[1] for x in items]);tensor=torch.from_numpy(rgb).permute(0,3,1,2).to(device,dtype=torch.float32)/255
        with torch.inference_mode():maps=model((tensor-mean)/std)
        h,w=maps.shape[-2:]
        for j,(t,_) in enumerate(items):
            selected=by_time[t];masks=list(executor.map(lambda row:pool_mask(row,h,w),selected))
            weights=torch.from_numpy(np.stack([x[1] for x in masks])).to(device)
            with torch.inference_mode():value=(weights@maps[j].reshape(576,-1).T)/weights.sum(1,keepdim=True).clamp_min(1e-8)
            values=value.cpu().numpy()
            for k,(idx,_,core_mass,halo_mass) in enumerate(masks):
                mass[idx]=[core_mass,halo_mass]
                if core_mass>1e-4 and np.isfinite(values[k]).all():pooled[idx]=values[k];valid[idx]=True
        del maps,tensor
    # The bounded frame batch and CPU mask threads overlap decoding/Metal work without RGB duplication.
    fifo=queue.Queue(maxsize=cfg['appearance_batch']*2);stop=threading.Event()
    def produce():
        try:
            for item in frame_stream(cfg,video,sorted(by_time)):
                while not stop.is_set():
                    try:fifo.put(item,timeout=.5);break
                    except queue.Full:pass
                if stop.is_set():return
            fifo.put(None)
        except BaseException as error:fifo.put(error)
    producer=threading.Thread(target=produce,daemon=True);producer.start()
    def consume():
        while True:
            item=fifo.get()
            if item is None:return
            if isinstance(item,BaseException):raise item
            yield item
    with ThreadPoolExecutor(max_workers=min(12,cfg['workers'])) as executor:
      try:
        for t,rgb in consume():
            batch.append((t,rgb))
            if len(batch)>=cfg['appearance_batch']:
                process_batch(batch,executor);done+=len(batch);batch=[]
                if done%256==0:monitor.check();LOG.info('%s appearance %d/%d frames %.1f fps',video,done,len(by_time),done/max(time.monotonic()-start,1e-9))
        if batch:process_batch(batch,executor);done+=len(batch)
      finally:stop.set();producer.join(timeout=10)
    pooled.flush();del pooled;temp.replace(features_path);np.save(valid_path,valid);np.save(out/f'{video}.mass.npy',mass)
    atomic_json(checkpoint,dict(inputs=fingerprint,frames=done,observations=len(rows),usable=int(valid.sum()),elapsed_seconds=time.monotonic()-start,device=str(device),feature_shape=[576,8,14],resize='full RGB resized to 448x256 without crop; native to feature-grid scale explicit',core_kappa=1.,halo_kappa=2.))
    LOG.info('%s appearance extraction COMPLETE frames=%d usable=%d',video,done,valid.sum())

def fit_projection(cfg,choices):
    from scipy.linalg import eigh
    from threadpoolctl import threadpool_limits
    source=ROOT/'work/track1_features';train=[c for c in choices if c['video_id'] in cfg['engineering_splits']['train'] and c['visor_split']=='train']
    total=np.zeros(576);cross=np.zeros((576,576));n=0;progress=[]
    with threadpool_limits(limits=16):
        for c in train:
            v=c['video_id'];a=np.load(source/f'{v}.npy',mmap_mode='r');valid=np.load(source/f'{v}.valid.npy')
            for begin in range(0,len(a),4096):
                x=a[begin:begin+4096][valid[begin:begin+4096]].astype(float);n+=len(x);total+=x.sum(0);cross+=x.T@x
            mean=total/n;cov=cross/n-np.outer(mean,mean);values=eigh(cov,eigvals_only=True)
            progress.append(dict(video_id=v,observations=n,variance=float(np.trace(cov)),top128_fraction=float(values[-128:].sum()/max(values.sum(),1e-20))))
            LOG.info('Appearance PCA trained observations=%d top128 variance=%.4f',n,progress[-1]['top128_fraction'])
        values,vectors=eigh(cov);order=np.argsort(values)[::-1];values=values[order];vectors=vectors[:,order]
    # Eigenvector sign canonicalisation makes repeated CPU fits reproducible.
    for j in range(vectors.shape[1]):
        if vectors[np.argmax(np.abs(vectors[:,j])),j]<0:vectors[:,j]*=-1
    out=output_dir(cfg,'appearance');save_shard(out/'projection.pt',dict(mean=mean.astype(np.float32),components=vectors[:,:128].astype(np.float32),eigenvalues=values.astype(np.float32)))
    evaluations={}
    for c in choices:
        v=c['video_id'];a=np.load(source/f'{v}.npy',mmap_mode='r');valid=np.load(source/f'{v}.valid.npy');embedding=np.zeros((len(a),128),np.float16);errors=np.zeros(6);count=0
        for begin in range(0,len(a),4096):
            stop=min(begin+4096,len(a));good=valid[begin:stop];x=a[begin:stop].astype(float)-mean;projection=x@vectors[:,:128]
            embedding[begin:stop]=projection.astype(np.float16)
            if good.any():
                energy=np.sum(x[good]**2,axis=1);count+=int(good.sum())
                for j,k in enumerate([0,8,16,32,64,128]):errors[j]+=float(np.sum(np.maximum(energy-np.sum(projection[good,:k]**2,axis=1),0)))
        embedding[~valid]=0;path=out/f'{v}.pt';save_shard(path,dict(embedding=embedding,valid=valid,core_halo_mass=np.load(source/f'{v}.mass.npy')))
        evaluations[v]=dict(valid_observations=count,reconstruction_mse_by_components={str(k):float(e/max(count*576,1)) for k,e in zip([0,8,16,32,64,128],errors)})
        manifest(cfg,'appearance',v,[path],evaluations[v])
    report=dict(training_videos=[c['video_id'] for c in train],training_observations=n,dimension=128,frozen_backbone='torchvision MobileNetV3-Small ImageNet1K V1',weights_sha256=digest(ROOT/'work/track1_weights/mobilenet_v3_small-047dcff4.pth'),projection='Train-only PCA, unwhitened; no backbone fine-tuning',progress=progress,evaluations=evaluations)
    atomic_json(out/'projection.json',report);manifest(cfg,'appearance','projection',[out/'projection.pt',out/'projection.json'])
    return report
