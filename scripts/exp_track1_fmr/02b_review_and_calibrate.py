"""Locked local VLM review. Model-generated references are never human adjudication."""
import argparse
import base64
import json
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT, atomic_json, digest
from track1_belief import data


def request(base, route, payload=None, timeout=600):
    req=urllib.request.Request(base+route,data=None if payload is None else json.dumps(payload).encode(),
                               headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)


def prepare(report):
    report.mkdir(exist_ok=False,parents=True);images=report/'images';images.mkdir()
    metadata=json.loads((ROOT/'reports/track1_belief/anchor_data.json').read_text())['episodes']
    index=pq.read_table(ROOT/'dataset/context_v3/frames.parquet').to_pylist();tasks=[];data.W=ROOT/'work/track1_fmr'
    for video in ['P08_09','P18_03']:
        meta=next(m for m in metadata if m['video_id']==video)
        with np.load(ROOT/meta['path']) as z:
            times=z['timestamp_us'][z['timestamp_us']>meta['manual_seed_us']]
        chosen=set(times[np.linspace(0,len(times)-1,150).astype(int)].tolist())
        frames=sorted([f for f in index if f['video_id']==video and f['timestamp_us'] in chosen],key=lambda f:f['timestamp_us'])
        if len(frames)!=150:raise ValueError('Need 150 exact native frames per participant')
        for n,rgb,_ in data.native_frames(video,frames,meta['episode_id']+'.vlm_review'):
            timestamp=frames[n]['timestamp_us'];path=images/f'{meta["episode_id"]}_{timestamp}.jpg'
            cv2.imwrite(str(path),cv2.cvtColor(cv2.resize(rgb,(768,432)),cv2.COLOR_RGB2BGR))
            tasks.append(dict(episode_id=meta['episode_id'],video_id=video,timestamp_us=timestamp,
                target_description=meta['entity_name'],target_id=meta['episode_id']+':'+meta['entity_name'],
                image=str(path.relative_to(ROOT)),image_sha256=digest(path)))
    atomic_json(report/'manifest.json',dict(tasks=tasks,participants=['P08','P18'],role='locked model-reviewed validation',
        development_participants=['P02','P03','P04','P11','P22'],human_adjudicated=False,
        reference_state_definition='visible foreground hand/manipulator including attached forearm and glove, model-estimated mask centroid and tight bbox',
        blinded_to_tracker=True,code_sha256=digest(Path(__file__))))


def parse_labels(content,batch_size):
    starts=[i for i in [content.find('{'),content.find('[')] if i>=0]
    if not starts:raise ValueError('No JSON labels')
    parsed=json.JSONDecoder().raw_decode(content[min(starts):])[0]
    entries=parsed if isinstance(parsed,list) else parsed['labels'];labels={}
    for label in entries:
        index=label['index']
        if type(index) is not int or index in labels or not 0<=index<batch_size:raise ValueError('Invalid/duplicate frame index')
        if not isinstance(label.get('visible'),bool) or not 0<=float(label['confidence'])<=1:raise ValueError('Invalid visibility/confidence')
        if label['visible']:
            if not all(type(label.get(k)) in (float,int) and np.isfinite(label[k]) and 0<=label[k]<=1 for k in ['x','y']):raise ValueError('Invalid centroid')
            box=np.asarray(label.get('bbox'),float)
            if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box<0) or np.any(box>1) or box[0]>=box[2] or box[1]>=box[3]:raise ValueError('Invalid foreground bbox')
            if not box[0]<=label['x']<=box[2] or not box[1]<=label['y']<=box[3]:raise ValueError('Centroid outside bbox')
        labels[index]=label
    if len(labels)!=batch_size:raise ValueError('Missing frame label')
    return labels


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report',default='reports/track1_fmr/stage2_vlm_review_v2_support')
    ap.add_argument('--base-url',default='http://127.0.0.1:1234');ap.add_argument('--model')
    ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--limit',type=int)
    ap.add_argument('--reuse-tasks',type=Path)
    ap.add_argument('--batch-size',type=int,default=4)
    args=ap.parse_args();report=ROOT/args.report
    if not report.resolve().is_relative_to(ROOT/'reports/track1_fmr'):raise ValueError('Report outside FMR reports')
    if not (report/'manifest.json').exists():
        if args.reuse_tasks is None:prepare(report)
        else:
            source=args.reuse_tasks if args.reuse_tasks.is_absolute() else ROOT/args.reuse_tasks
            manifest=json.loads(source.read_text());manifest.update(reference_state_definition='visible foreground hand/manipulator support including attached visible forearm, bbox and area centroid; model estimate',
                source_manifest_sha256=digest(source),code_sha256=digest(Path(__file__)),human_adjudicated=False)
            report.mkdir(parents=True,exist_ok=False);atomic_json(report/'manifest.json',manifest)
    if args.prepare_only:return
    models=request(args.base_url,'/v1/models');atomic_json(report/'server_models.json',models)
    available=[m['id'] for m in models['data']]
    model=args.model or next((m for m in available if 'qwen3.8' in m.lower()),None)
    if model is None or model not in available:raise ValueError('Requested Qwen3.8 model is not served')
    manifest=json.loads((report/'manifest.json').read_text());tasks=manifest['tasks'];completed=[]
    (report/'responses').mkdir(exist_ok=True)
    if not 1<=args.batch_size<=6:raise ValueError('Batch size must be 1..6')
    pending=[]
    for task in tasks[:args.limit]:
        out=report/'responses'/f"{task['episode_id']}_{task['timestamp_us']}.json"
        if out.exists():
            previous=json.loads(out.read_text())
            if previous['label'] is not None:completed.append(previous);continue
            try:
                content=''.join(item['content'] for item in previous['raw']['output'] if item['type']=='message')
                labels=parse_labels(content,previous['request_settings']['batch_size'])
                original_sha=digest(out);attempts=report/'invalid_attempts';attempts.mkdir(exist_ok=True)
                out.rename(attempts/(out.stem+f'.attempt{len(list(attempts.glob(out.stem+"*")))}.json'))
                previous.update(label=labels[previous['request_settings']['batch_index']],validation_error=None,
                    recovered_from_saved_raw_sha256=original_sha,normalizer_code_sha256=digest(Path(__file__)))
                atomic_json(out,previous);completed.append(previous);continue
            except (ValueError,KeyError,TypeError):pass
            attempts=report/'invalid_attempts';attempts.mkdir(exist_ok=True)
            out.rename(attempts/(out.stem+f'.attempt{len(list(attempts.glob(out.stem+"*")))}.json'))
        pending.append(task)
    for offset in range(0,len(pending),args.batch_size):
        batch=pending[offset:offset+args.batch_size]
        prompt=('Independently inspect each numbered egocentric image and find its named hand/manipulator. '
            'Use the whole visible foreground support: hand plus attached visible forearm, including any worn glove. '
            'Exclude background and unrelated body regions. Do NOT cut the forearm off at an anatomical wrist. '
            'Estimate the AREA CENTROID of this visible foreground mask (not a joint or bounding-box center), '
            'and its tight bounding box [left,top,right,bottom]. Coordinates normalized [0,1] in each image. '
            'If absent/occluded/ambiguous abstain. No tracker output or source masks are supplied. '
            'Return JSON only: {"labels": [{"index":0,"visible":true,"x":0.5,"y":0.5,'
            '"bbox":[0.2,0.2,0.8,0.8],"confidence":0.9}, ...]}. One label for each index. No explanation.')
        inputs=[dict(type='text',content=prompt)]
        for index,task in enumerate(batch):
            image=ROOT/task['image']
            if digest(image)!=task['image_sha256']:raise ValueError('Locked image changed')
            inputs.extend([dict(type='text',content=f"Image index {index}: target {task['target_description']}"),
                dict(type='image',data_url='data:image/jpeg;base64,'+base64.b64encode(image.read_bytes()).decode())])
        payload=dict(model=model,temperature=0,max_output_tokens=900,reasoning='off',store=False,input=inputs)
        raw=request(args.base_url,'/api/v1/chat',payload)
        content=''.join(x['content'] for x in raw['output'] if x['type']=='message')
        labels={};parse_error=None
        try:
            labels=parse_labels(content,len(batch))
        except (ValueError,KeyError,TypeError) as exc:parse_error=str(exc);labels={}
        for index,task in enumerate(batch):
            row=dict(task=task,model=model,prompt=prompt,raw=raw,label=labels.get(index),
                validation_error=parse_error if parse_error else (None if index in labels else 'Missing frame label'),
                human_adjudicated=False,reference_kind='local_vlm_silver',reviewer_code_sha256=digest(Path(__file__)),
                request_settings=dict(route='/api/v1/chat',reasoning='off',temperature=0,batch_size=len(batch),batch_index=index),
                manifest_sha256=digest(report/'manifest.json'))
            atomic_json(report/'responses'/f"{task['episode_id']}_{task['timestamp_us']}.json",row);completed.append(row)
        atomic_json(report/'progress.json',dict(completed=len(completed),total=len(tasks),model=model,
            valid=sum(r['label'] is not None for r in completed)))
        print(f"review {len(completed)}/{len(tasks)} valid={sum(r['label'] is not None for r in completed)}",flush=True)
    accepted=[r for r in completed if r['label'] is not None and r['label']['confidence']>=.8]
    atomic_json(report/'result.json',dict(reviewed=len(completed),total=len(tasks),accepted=len(accepted),model=model,
        human_adjudicated=False,reference_kind='local_vlm_silver',independent_human_nees=None,independent_human_idf1=None,
        response_hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted((report/'responses').glob('*.json'))}))


if __name__=='__main__':main()
