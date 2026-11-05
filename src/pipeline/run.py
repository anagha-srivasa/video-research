from __future__ import annotations
import argparse
import fcntl
import json
import logging
from pathlib import Path
import sys
import subprocess
import time
import yaml
from .common import ROOT,SCHEMAS,LOG,atomic_json,source_paths,digest,config_hash,code_hash,stage_manifest,tool_versions,verify_publisher_file
from .video import index_video
from .visor import prepare_dense,normalize_visor
from .semantic import normalize_actions,normalize_camera,build_events,build_windows
from .qa import audit_and_render

STAGES=['audit_raw','index_video_pts','prepare_visor_dense','normalize_visor','normalize_actions','normalize_camera','build_event_evidence','build_window_index','run_qa']
ALIASES={'normalise_visor':'normalize_visor','normalise_camera':'normalize_camera','normalise_actions':'normalize_actions',
         'normalise_contacts':'normalize_visor','build_entity_scopes':'normalize_visor','build_mask_geometry':'normalize_visor','align_annotation_time':'normalize_visor'}

def audit_raw(cfg,choice):
    v=choice['video_id'];inputs=source_paths(cfg,v)
    missing=[str(p.relative_to(ROOT)) for p in inputs if not p.exists()]
    if missing:raise FileNotFoundError('Raw sources incomplete: '+str(missing))
    if digest(inputs[0],'md5')!=choice['official_md5']:raise ValueError('Publisher MD5 mismatch: '+v)
    publisher=ROOT/cfg['raw']/'metadata/VISOR_publisher_info.json'
    verified=[]
    if publisher.exists():
        metadata=json.loads(publisher.read_text());lookup={}
        def visit(value):
            if isinstance(value,dict):
                if 'sha1' in value:lookup[value['id']]=value
                for child in value.values():visit(child)
            elif isinstance(value,list):
                for child in value:visit(child)
        visit(metadata)
        base='2v6cgv1x04ol22qp9rm9x2j6a7/'
        paths=[(inputs[1],base+f"GroundTruth-SparseAnnotations/annotations/{choice['visor_split']}/{v}.json"),
               (inputs[2],base+f"Interpolations-DenseAnnotations/{choice['visor_split']}/{v}_interpolations.zip"),
               (ROOT/cfg['raw']/'visor/frame_mapping.json',base+'frame_mapping.json')]
        for local,key in paths:
            expected=lookup[key]
            verify_publisher_file(local,expected)
            verified.append(str(local.relative_to(ROOT)))
    for p in inputs:
        if p.suffix=='.json':json.loads(p.read_text())
    return stage_manifest(cfg,v,'audit_raw',inputs,[],dict(source_complete=True,publisher_visor_sha1_verified=verified))

def reusable(cfg,v,stage):
    path=ROOT/cfg['output']/f'{stage}.{v}.manifest.json'
    if not path.exists():return False
    m=json.loads(path.read_text())
    if not m['completed'] or m['config_sha256']!=config_hash(cfg) or m['pipeline_sha256']!=code_hash() or m['tool_versions']!=tool_versions():return False
    return all((ROOT/p).exists() and digest(ROOT/p)==h for p,h in {**m['source_files'],**m['outputs']}.items())

def run_one(cfg,choice,stages=None,resume=False,render=True):
    out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True);v=choice['video_id']
    for stage in stages or STAGES:
        if resume and reusable(cfg,v,stage):LOG.info('%s: %s SKIP verified identical sources/config/code/outputs',v,stage);continue
        start=time.monotonic();LOG.info('%s: %s START',v,stage)
        if stage=='audit_raw':audit_raw(cfg,choice)
        elif stage=='index_video_pts':subprocess.run([sys.executable,'-m','pipeline.video','--config-json',json.dumps(cfg),'--choice-json',json.dumps(choice)],cwd=ROOT,check=True)
        elif stage=='prepare_visor_dense':prepare_dense(cfg,v)
        elif stage=='normalize_visor':normalize_visor(cfg,choice)
        elif stage=='normalize_actions':normalize_actions(cfg,choice)
        elif stage=='normalize_camera':normalize_camera(cfg,choice)
        elif stage=='build_event_evidence':build_events(cfg,choice)
        elif stage=='build_window_index':build_windows(cfg,choice)
        elif stage=='run_qa':audit_and_render(cfg,choice,render=render)
        LOG.info('%s: %s PASS elapsed=%.1fs',v,stage,time.monotonic()-start)

def reproduce(cfg,choice):
    v=choice['video_id'];copy=dict(cfg,output='work/repro_v1')
    qa_path=ROOT/'reports'/f'qa.{v}.json'
    original_qa=qa_path.read_bytes() if qa_path.exists() else None
    run_one(copy,choice,render=False)
    if original_qa is not None:qa_path.write_bytes(original_qa)
    comparisons={}
    for table in SCHEMAS:
        a=ROOT/cfg['output']/f'{table}.{v}.parquet';b=ROOT/copy['output']/a.name
        comparisons[table]=dict(original=digest(a),rebuilt=digest(b),identical=digest(a)==digest(b))
    passed=all(r['identical'] for r in comparisons.values())
    atomic_json(ROOT/'reports'/f'reproducibility.{v}.json',dict(video_id=v,passed=passed,tables=comparisons,method='Independent full rebuild into work/repro_v1; bytewise Parquet SHA256 comparison.'))
    if not passed:raise ValueError('Reproducibility failed: '+v)
    LOG.info('%s: independent rebuild has identical hashes for all %d tables',v,len(comparisons))

def refresh_status(cfg,selection):
    out=ROOT/cfg['output'];videos=[]
    for choice in selection:
        v=choice['video_id'];qa=ROOT/'reports'/f'qa.{v}.json';repro=ROOT/'reports'/f'reproducibility.{v}.json';review=ROOT/'reports'/f'visual_review.{v}.json'
        complete=all((out/f'{table}.{v}.parquet').exists() for table in SCHEMAS)
        q=json.loads(qa.read_text()) if qa.exists() else {}
        r=json.loads(repro.read_text()) if repro.exists() else {}
        visual=json.loads(review.read_text()) if review.exists() else {}
        hashes={table:digest(out/f'{table}.{v}.parquet') for table in SCHEMAS} if complete else {}
        reproducible=complete and r.get('passed',False) and all(r.get('tables',{}).get(t,{}).get('original')==h for t,h in hashes.items())
        reviewed=complete and visual.get('passed',False) and visual.get('table_hashes')==hashes
        image=ROOT/'reports'/f'qa.{v}.jpg'
        reviewed=reviewed and image.exists() and visual.get('image_sha256')==digest(image)
        automated=q.get('automated_pass',False) and q.get('pipeline_sha256')==code_hash() and q.get('config_sha256')==config_hash(cfg)
        videos.append(dict(video_id=v,complete=complete,automated_pass=automated,reproducible=reproducible,visual_review=reviewed))
    ready=all(v['complete'] and v['automated_pass'] and v['reproducible'] and v['visual_review'] for v in videos)
    status=dict(schema_version=cfg['schema_version'],ingestion_ready=ready,videos=videos,
        readiness_scope='Architecture-neutral evidence; explicit unresolved/quarantined data retained. Camera resize/crop remains unverified; no metric object geometry.',
        config_sha256=config_hash(cfg),pipeline_sha256=code_hash())
    atomic_json(out/'status.json',status)
    atomic_json(out/'provenance.json',dict(schema_version=cfg['schema_version'],raw=str(cfg['raw']),license=cfg['license'],
        source_urls=['https://epic-kitchens.github.io/','https://doi.org/10.5523/bris.2v6cgv1x04ol22qp9rm9x2j6a7','https://epic-kitchens.github.io/epic-fields/site'],
        correction_tool_commit=json.loads((ROOT/'work/vendor_support/commit.json').read_text())['sha'],
        table_layout='<table>.<video_id>.parquet',config=cfg,pipeline_sha256=code_hash(),
        time_conventions='Native MP4 frame PTS minus first valid PTS; integer microseconds. Dense ordinal transport and camera anchor-time interpolation are explicitly inferred.',
        limits=['No fitted B-splines, wavelets, Legendre coefficients, depth, metric reconstruction, or learned confidence.','Source identities are provisional scopes, not verified tracks.','Missing segmentation is not absence; contact changes are interval-censored observations.']))
    return status

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/pilot_v1.yaml');p.add_argument('--stage',default='all')
    p.add_argument('--video');p.add_argument('--gold-only',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--verify-reproducibility',action='store_true');p.add_argument('--refresh-status',action='store_true')
    args=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',stream=sys.stdout)
    cfg=yaml.safe_load((ROOT/args.config).read_text())
    for key in ['raw','output','staging']:
        if not (ROOT/cfg[key]).resolve().is_relative_to(ROOT):raise ValueError('All paths must remain inside video_research')
    lock=(ROOT/'work/pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    selection=json.loads((ROOT/cfg['raw']/'metadata/pilot_selection.json').read_text())
    if args.refresh_status:print(json.dumps(refresh_status(cfg,selection),indent=2));return
    choices=[c for c in selection if not args.video or c['video_id']==args.video]
    if args.gold_only:choices=[c for c in choices if c['video_id']==cfg['gold_video']]
    choices.sort(key=lambda c:(c['video_id']!=cfg['gold_video'],c['video_id']))
    if not choices:raise ValueError('No matching pilot video')
    stage=ALIASES.get(args.stage,args.stage)
    if stage!='all' and stage not in STAGES:raise ValueError('Unknown stage '+stage)
    start=time.monotonic()
    atomic_json(ROOT/cfg['output']/'status.json',dict(schema_version=cfg['schema_version'],ingestion_ready=False,reason='processing_or_selective_rebuild'))
    try:
        for i,choice in enumerate(choices,1):
            LOG.info('VIDEO %d/%d %s',i,len(choices),choice['video_id'])
            run_one(cfg,choice,None if stage=='all' else [stage],args.resume)
            if args.verify_reproducibility and stage=='all':reproduce(cfg,choice)
            refresh_status(cfg,selection)
    except BaseException:
        LOG.exception('Pipeline interrupted/failed; raw source files were not modified. Partial table writes are unpublished; readiness remains false.')
        raise
    LOG.info('Completed requested pipeline stages elapsed=%.1fs; readiness=%s',time.monotonic()-start,refresh_status(cfg,selection)['ingestion_ready'])

if __name__=='__main__':main()
