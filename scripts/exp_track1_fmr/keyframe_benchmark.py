"""v2 execution path; reference data are confined to scoring and initial query binding."""
import importlib.util
import json
import threading
import time
from pathlib import Path
import cv2
import numpy as np
import psutil
from pipeline.common import ROOT,atomic_json,digest
from track1_fmr.keyframe_identity import KeyframeDetector,KeyframeObservationIdentityTracker
from track1_neural_observation.benchmark import ArrayUnpickler,official_metrics

METHODS=('keyframe_strict','keyframe_lineage')


def support(e):
    if e.observation.mask is not None:
        if e.evidence=='detector':return e.observation.mask.copy()
        # Measured support-box displacement, not the filtered centroid offset.
        shift=(e.box[:2]-e.observation.box[:2])*160
        return cv2.warpAffine(e.observation.mask.astype(np.uint8),np.array([[1,0,shift[0]],[0,1,shift[1]]],np.float32),(160,160),flags=cv2.INTER_NEAREST).astype(bool)
    m=np.zeros((160,160),bool);lo=np.clip((e.box[:2]*160).astype(int),0,160);hi=np.clip(np.ceil(e.box[2:]*160).astype(int),0,160);m[lo[1]:hi[1],lo[0]:hi[0]]=True;return m


def bind_current(states,point):
    candidates=[]
    x,y=np.clip((np.asarray(point)*160).astype(int),0,159)
    for tid,e in states.items():
        if e.observed and np.all(point>=e.box[:2]) and np.all(point<=e.box[2:]) and support(e)[y,x]:
            candidates.append((float(np.prod(e.box[2:]-e.box[:2])),tid))
    return min(candidates)[1] if candidates else -1


def execute(args,helpers):
    revision=args.revision or ('v4' if args.keyframe_stride==10 else 'v3' if args.keyframe_stride==8 else 'v2')
    methods=METHODS+('hybrid_strict','hybrid_lineage') if revision=='v4' else METHODS
    helpers.isolated();out=ROOT/'reports/track1_fmr'/args.tag;out.mkdir(exist_ok=False)
    import torch
    torch.set_num_threads(4);cv2.setNumThreads(4)
    started=time.perf_counter();stop=threading.Event();resource=dict(peak=0,failure=None)
    root=psutil.Process()
    def monitor():
        while not stop.wait(.05):
            try:
                rss=0
                for proc in [root]+root.children(recursive=True):
                    try:rss+=proc.memory_info().rss
                    except psutil.NoSuchProcess:pass
                resource['peak']=max(resource['peak'],rss)
                if rss>18*2**30 or psutil.virtual_memory().available<6*2**30:resource['failure']='Memory guard exceeded'
            except psutil.Error as exc:resource['failure']=str(exc)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    def guard():
        if resource['failure']:raise MemoryError(resource['failure'])
    try:
        dataset=ROOT/'work/track1_observation_revision/tapvid_davis/tapvid_davis.pkl'
        receipt=json.loads((dataset.parent/'receipt.json').read_text())
        if digest(dataset)!=receipt['uncompressed_sha256']:raise ValueError('Dataset changed')
        with dataset.open('rb') as f:cases=ArrayUnpickler(f).load()
        names=sorted(cases)[:args.videos]
        selection_path=ROOT/'reports/track1_fmr/stage2_visor_v3/protocol.json';selection=json.loads(selection_path.read_text())
        for rel,sha in selection['input_hashes'].items():
            if digest(ROOT/rel)!=sha:raise ValueError('Reference changed')
        sources=[Path(__file__),ROOT/'scripts/exp_track1_fmr/03_eval_observation_identity.py',ROOT/'scripts/exp_track1_fmr/04_verify_observation_identity.py',*sorted((ROOT/'src/track1_fmr').glob('*.py')),
                 ROOT/'src/track1_belief/data.py',ROOT/'src/track1_encoder_revision/source_stream.py',ROOT/'src/track1_neural_observation/benchmark.py']
        metadata_path=ROOT/'reports/track1_belief/anchor_data.json';frame_path=ROOT/'dataset/context_v3/frames.parquet'
        protocol=dict(kind='keyframe_'+revision,revision=revision,methods=methods,case_names=names,dataset_sha256=digest(dataset),keyframe_stride=args.keyframe_stride,
            detector_confidence=.1,detector_size='native-capped at 480, rounded to 32' if revision=='v4' else 480 if revision=='v3' else 640,grid=160,anchor_resolution='native-capped at 320' if revision=='v4' else 320,anchors=12 if revision=='v3' else 8,fb_max_px=1.2 if revision=='v3' else 1.5,min_inliers=3 if revision=='v4' else 4,patch_ncc_min=.65 if revision=='v3' else None,lk_window=9 if revision=='v3' else 13 if revision=='v4' else 21,detector_heads=1 if revision in ('v3','v4') else 2,lk_covariance_policy='CV plus measured-displacement diffusion; no LK conditioning' if revision in ('v3','v4') else 'Condition on measured LK displacement',
            code_hashes={str(p.resolve().relative_to(ROOT)):digest(p) for p in sources},reference_input_hashes=selection['input_hashes'],reference_selection_sha256=digest(selection_path),
            checkpoint_hashes={str(p.relative_to(ROOT)):digest(p) for p in ([ROOT/'work/yoloe-26s-seg-pf.pt',ROOT/'work/yoloe-26s-seg.pt'] if revision=='v4' else [ROOT/'work/yoloe-26s-seg.pt',ROOT/'work/mobileclip2_b.ts'] if revision=='v3' else [ROOT/'work/yoloe-26s-seg-pf.pt',ROOT/'work/yoloe-26s-seg.pt',ROOT/'work/mobileclip2_b.ts'])},
            native_metadata_sha256=digest(metadata_path),native_frame_index_sha256=digest(frame_path),
            observation_policy='detector or >=3 FB-verified anchors; hybrid particle additionally requires its own FB verification and observed carrier support' if revision=='v4' else 'detector or >=4 FB-verified image correspondences; predictions never observed',
            point_policy='Initial query only; incremental inverse-distance anchor displacement, detector-box affine fallback after anchor loss. Lost anchors never predicted forward. No later reference point input.',
            lineage='Causal provisional split/merge edges; smallest active descendant support containing prior transported point. Strict IDs also scored.',
            q_units='continuous white acceleration spectral density',tracker_q=2.40508678,tracker_floor=.012 if revision in ('v3','v4') else .00979414048,lk_diffusion_density=.012**2 if revision=='v4' else .015**2 if revision=='v3' else None,gray_scope='One 320x320 pyramid per frame, optional caller-provided buffer; no parallel DorsalFluxExtractor in this carrier',covariance_eigenvalue_cap=None if revision in ('v3','v4') else .25,
            timing='Isolated CPU; particle LK, gating and reseeding included on DAVIS; initial query binding and metric scoring excluded; native has no queries; decoder and cold warmup excluded' if revision=='v4' else 'Isolated CPU, shared resident detector, every runtime keyframe/flow/association cost included; query evaluation excluded; decoder excluded; warmup separate',
            physical_identity_verified=False,human_adjudicated=False,scientific_acceptance=False,query_particle_protocol='Only first query coordinates enter particles; particles never enter carrier association. Fallback/reseed hypotheses are not observed until a later FB-verified step. Combined-query LK timing included.' if revision=='v4' else None,anti_hijack_area_guard='actual foreground mass <0.06 cannot match >3x mass' if revision=='v4' else None)
        if args.reuse_native_from:
            if not args.reuse_native_from.replace('_','').isalnum():raise ValueError('Invalid reuse tag')
            old=ROOT/'reports/track1_fmr'/args.reuse_native_from;old_protocol=json.loads((old/'protocol.json').read_text())
            assert old_protocol['checkpoint_hashes']==protocol['checkpoint_hashes'] and old_protocol['keyframe_stride']==protocol['keyframe_stride']
            for rel,sha in protocol['code_hashes'].items():
                if rel.startswith('src/') and rel in old_protocol['code_hashes']:assert old_protocol['code_hashes'][rel]==sha,rel
            assert old_protocol['native_metadata_sha256']==protocol['native_metadata_sha256'] and old_protocol['native_frame_index_sha256']==protocol['native_frame_index_sha256']
            protocol['native_reused_from']=dict(tag=args.reuse_native_from,protocol_sha256=digest(old/'protocol.json'),result_sha256=digest(old/'result.json'),scope='Identical runtime sources, checkpoint hashes, keyframe stride and native inputs. Query-evaluation-only correction; immutable native measurements and their original isolated timings copied unchanged.')
        atomic_json(out/'protocol.json',protocol)
        for rel in protocol['code_hashes']:
            dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
        detector=KeyframeDetector(revision=revision);detector.warmup();guard()
        atomic_json(out/'warmup.json',dict(seconds=detector.warmup_seconds,text_cache_manifest=str(detector.cache_manifest.relative_to(ROOT)) if detector.cache_manifest else None,text_cache_sha256=digest(detector.cache_manifest) if detector.cache_manifest else None))
        rows=[];visor=[];all_timings=[];native=[]
        metric=official_metrics()
        for ci,name in enumerate([] if args.visor_only else names):
            helpers.isolated();case=cases[name];video=case['video'];target=np.asarray(case['points'],float);hidden=np.asarray(case['occluded'],bool)
            keep=(~hidden).any(1);target=target[keep];hidden=hidden[keep];first=np.argmax(~hidden,1);count,frames=hidden.shape
            model=KeyframeObservationIdentityTracker(args.keyframe_stride,revision=revision)
            positions={m:np.full((count,frames,2),-1e4,np.float32) for m in methods}
            observed={m:np.zeros((count,frames),bool) for m in methods};carriers={m:np.zeros((count,frames),bool) for m in methods}
            evidence={m:np.zeros((count,frames),np.uint8) for m in methods};active_history={m:np.full((count,frames),-1,int) for m in methods};active={m:np.full(count,-1,int) for m in methods};fixed={m:np.full(count,-1,int) for m in methods}
            points={m:np.zeros((count,2)) for m in methods};frame_rows=[];times=[];trace=[];particle_trace=[];adoption_trace=[]
            for frame,rgb in enumerate(video):
                guard();tick=time.perf_counter();states=model.step(rgb,frame/30,detector);elapsed=time.perf_counter()-tick;times.append(elapsed);all_timings.append(elapsed)
                record=dict(frame=frame,**model.last_telemetry);frame_rows.append(record)
                particle_trace.extend(dict(frame=frame,**x) for x in model.last_particle_trace);adoption_trace.extend(model.last_adoptions)
                for e in states.values():
                    trace.append(dict(frame=frame,id=e.track_id,source=e.evidence,inliers=len(e.fb_errors),fb_max=float(e.fb_errors.max()) if len(e.fb_errors) else None,ncc_min=float(e.ncc_scores.min()) if len(e.ncc_scores) else None,old_anchors=e.displacement_old.tolist(),new_anchors=e.displacement_new.tolist(),mean=e.belief.mean[:2].tolist(),shape=e.belief.shape_scatter.tolist(),box=e.box.tolist(),previous_box=None if e.previous_box is None else e.previous_box.tolist()))
                for m in methods:
                    for k in np.flatnonzero(first==frame):
                        tid=bind_current(states,target[k,frame]);fixed[m][k]=active[m][k]=tid;points[m][k]=target[k,frame]
                        if m.startswith('hybrid') and tid>=0:model.bind_query_point((0 if m=='hybrid_strict' else count)+int(k),tid,target[k,frame],lineage=m.endswith('lineage'))
                    for k in np.flatnonzero((first<frame)&(active[m]>=0)):
                        if m.startswith('hybrid'):
                            particle=model.query_particles[(0 if m=='hybrid_strict' else count)+int(k)]
                            tid=particle.track_id;active[m][k]=tid
                            if tid in states:
                                active_history[m][k,frame]=tid;carriers[m][k,frame]=True
                                if particle.observed:observed[m][k,frame]=True;positions[m][k,frame]=particle.position;evidence[m][k,frame]=3
                            continue
                        tid=int(active[m][k]);old=states.get(tid)
                        if m=='keyframe_lineage':tid=model.follow(tid,points[m][k]);active[m][k]=tid
                        e=states.get(tid)
                        if e is None:continue
                        # Preserve local query offset across detector reseeding, using this frame's verified old/new anchors.
                        mover=old if old is not None else e
                        points[m][k]=model.transport_entity(points[m][k],mover)
                        active_history[m][k,frame]=tid;carriers[m][k,frame]=True
                        if e.observed:
                            observed[m][k,frame]=True;positions[m][k,frame]=points[m][k];evidence[m][k,frame]=1 if e.evidence=='detector' else 2
                if frame%25==0:print(f'{name} {frame+1}/{frames}; elapsed={time.perf_counter()-started:.1f}s',flush=True)
            queries=np.array([[t,target[k,t,1]*256,target[k,t,0]*256] for k,t in enumerate(first)],np.float32)
            case_rows=[];arrays={}
            for m in methods:
                row=helpers.point_row(m,positions[m],observed[m],carriers[m],target,hidden,first,fixed[m]);row.update(video=name,mean_component_ms=1000*np.mean(times))
                official=metric(queries[None],hidden[None],(target*256)[None],(~observed[m])[None],(positions[m]*256)[None],query_mode='first')
                row['official_tapvid_observed_only']={k:float(v[0]) for k,v in official.items()};rows.append(row);case_rows.append(row)
                arrays.update({m+'_position':positions[m],m+'_observed':observed[m],m+'_carrier':carriers[m],m+'_fixed_ids':fixed[m],m+'_evidence':evidence[m],m+'_active_ids':active_history[m]})
            np.savez_compressed(out/(name+'.points.npz'),**arrays)
            atomic_json(out/(name+'.trace.json'),dict(frames=frame_rows,entities=trace,lineage=model.lineage,particles=particle_trace,adoptions=adoption_trace))
            atomic_json(out/(name+'.json'),dict(frames=frames,queries=count,rows=case_rows,timings=times,points_sha256=digest(out/(name+'.points.npz')),trace_sha256=digest(out/(name+'.trace.json'))))
            atomic_json(out/'progress.json',dict(completed=ci+1,total=len(names),elapsed_seconds=time.perf_counter()-started));print('completed',ci+1,len(names),name,flush=True)
        spec=importlib.util.spec_from_file_location('visor_reference',ROOT/'scripts/exp_track1_fmr/02a_eval_visor_sparse.py');reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
        from track1_belief import data
        data.W=ROOT/'work/track1_fmr'
        for ep in ([] if args.skip_visor else selection['episodes']):
            model=KeyframeObservationIdentityTracker(args.keyframe_stride,revision=revision);bound={m:-1 for m in methods};point={m:None for m in methods};bytime={r['timestamp_us']:r for r in ep['references']}
            for index,rgb,_ in data.native_frames(ep['video_id'],ep['frames'],ep['video_id']+'.keyframe'):
                guard();states=model.step(rgb,ep['frames'][index]['timestamp_us']/1e6,detector)
                for m in methods:
                    if point[m] is not None:
                        old=states.get(bound[m]);tid=model.follow(bound[m],point[m]) if m.endswith('lineage') else bound[m]
                        if old is not None:point[m]=model.transport_entity(point[m],old)
                        bound[m]=tid
                ts=ep['frames'][index]['timestamp_us']
                if ts not in bytime:continue
                truth=reference.raster(bytime[ts]);yy,xx=np.nonzero(truth);centre=np.array([xx.mean()/160,yy.mean()/160])
                for m in methods:
                    if index==0:bound[m]=bind_current(states,centre);point[m]=centre.copy()
                    e=states.get(bound[m]);pred=support(e) if e is not None and e.observed else np.zeros_like(truth)
                    direct=max([float((support(x)&truth).sum()/max(1,(support(x)|truth).sum())) for x in states.values() if x.evidence=='detector'] or [0.])
                    visor.append(dict(video_id=ep['video_id'],timestamp_us=ts,method=m,initialization_frame=index==0,fixed_id=bound[m],detector_best_mask_iou=direct,observed_fixed_id_mask_recall=float((pred&truth).sum()/truth.sum())))
            atomic_json(out/'visor_progress.json',visor);print('completed VISOR',ep['video_id'],flush=True)
        # New native cohort: inference reads RGB and timestamps only, never source centres or VLM labels.
        if args.reuse_native_from:
            import shutil
            old=ROOT/'reports/track1_fmr'/args.reuse_native_from;old_result=json.loads((old/'result.json').read_text());native=old_result['native']
            resource['peak']=max(resource['peak'],old_result['sampled_process_tree_peak_rss_gib']*2**30)
            for entry in native:
                name=entry['episode_id']+'.observations.npz';assert digest(old/name)==entry['sha256'];shutil.copy2(old/name,out/name)
            atomic_json(out/'native_progress.json',native)
        elif not args.visor_only and not args.skip_native:
            import pyarrow.parquet as pq
            metadata=json.loads(metadata_path.read_text())['episodes'];frames_all=pq.read_table(frame_path).to_pylist()
            for meta in metadata:
                source=ROOT/meta['path']
                if digest(source)!=meta['sha256']:raise ValueError('Source ledger changed')
                with np.load(source,allow_pickle=False) as z:timestamps=z['timestamp_us'];timestamps=timestamps[timestamps>=meta['manual_seed_us']]
                requested=set(timestamps.tolist());frames=sorted([f for f in frames_all if f['video_id']==meta['video_id'] and f['timestamp_us'] in requested],key=lambda f:f['timestamp_us'])
                if [f['timestamp_us'] for f in frames]!=timestamps.tolist():raise ValueError('Native PTS mismatch')
                model=KeyframeObservationIdentityTracker(args.keyframe_stride,revision=revision);values=[];shapes=[];boxes=[];hits=[];sources=[];motions=[];motion_dt=[];inliers=[];fb_max=[];ncc_min=[];timings=[]
                for n,rgb,_ in data.native_frames(meta['video_id'],frames,meta['episode_id']+'.keyframe'):
                    guard()
                    if n%100==0:helpers.isolated();print('native',meta['episode_id'],n,len(frames),flush=True)
                    tick=time.perf_counter();states=model.step(rgb,frames[n]['timestamp_us']/1e6,detector);timings.append(time.perf_counter()-tick)
                    for tid,e in states.items():
                        if not e.observed:continue
                        values.append([e.belief.time,tid,*e.measurement,*e.measurement_covariance.ravel(),*e.belief.mean,*e.belief.covariance.ravel()]);shapes.append(e.belief.shape_scatter);boxes.append(e.box.copy());hits.append(e.hits);sources.append(e.evidence);motions.append(e.motion_delta.copy());motion_dt.append(e.motion_dt);inliers.append(len(e.fb_errors));fb_max.append(float(e.fb_errors.max()) if len(e.fb_errors) else -1.);ncc_min.append(float(e.ncc_scores.min()) if len(e.ncc_scores) else -1.)
                path=out/(meta['episode_id']+'.observations.npz')
                np.savez_compressed(path,values=np.asarray(values).reshape(-1,28),support_shapes=np.asarray(shapes).reshape(-1,2,2),support_boxes=np.asarray(boxes).reshape(-1,4),observed_counts=np.asarray(hits),evidence=np.asarray(sources),motion_delta=np.asarray(motions).reshape(-1,2),motion_dt=np.asarray(motion_dt),fb_inliers=np.asarray(inliers),fb_max=np.asarray(fb_max),ncc_min=np.asarray(ncc_min))
                native.append(dict(episode_id=meta['episode_id'],frames=len(frames),observations=len(values),sha256=digest(path),mean_ingestion_ms=1000*np.mean(timings),timings=timings));atomic_json(out/'native_progress.json',native)
        summaries=[]
        for m in methods:
            selected=[r for r in rows if r['method']==m];den=sum(r['visible_point_times'] for r in selected)
            if not den:continue
            summaries.append(dict(method=m,visible_point_times=den,observed_visible_coverage=sum(r['observed_visible_point_times'] for r in selected)/den,observed_point_agreement_8px=sum(r['observed_correct_8px_times'] for r in selected)/den,recovered_returns=sum(r['recovered_returns_5frame'] for r in selected),mean_ingestion_ms=1000*np.mean(all_timings)))
        atomic_json(out/'result.json',dict(protocol_sha256=digest(out/'protocol.json'),summaries=summaries,rows=rows,visor=visor,native=native,sampled_process_tree_peak_rss_gib=resource['peak']/2**30,elapsed_seconds=time.perf_counter()-started,scientific_acceptance=False))
    except Exception as exc:
        atomic_json(out/'failure.json',dict(error=repr(exc),peak_rss_gib=resource['peak']/2**30));raise
    finally:stop.set();thread.join()


def verify(out):
    protocol=json.loads((out/'protocol.json').read_text());result=json.loads((out/'result.json').read_text())
    methods=tuple(protocol['methods'])
    assert digest(out/'protocol.json')==result['protocol_sha256']
    for rel,sha in protocol['code_hashes'].items():assert digest(out/'source'/rel)==sha,rel
    for group in ('checkpoint_hashes','reference_input_hashes'):
        for rel,sha in protocol[group].items():assert digest(ROOT/rel)==sha,rel
    dataset=ROOT/'work/track1_observation_revision/tapvid_davis/tapvid_davis.pkl';assert digest(dataset)==protocol['dataset_sha256']
    with dataset.open('rb') as f:cases=ArrayUnpickler(f).load()
    if protocol.get('native_reused_from'):
        reused=protocol['native_reused_from'];old=ROOT/'reports/track1_fmr'/reused['tag']
        assert digest(old/'protocol.json')==reused['protocol_sha256'] and digest(old/'result.json')==reused['result_sha256']
    metadata_path=ROOT/'reports/track1_belief/anchor_data.json';frame_path=ROOT/'dataset/context_v3/frames.parquet'
    assert digest(metadata_path)==protocol['native_metadata_sha256'] and digest(frame_path)==protocol['native_frame_index_sha256']
    metadata=json.loads(metadata_path.read_text())['episodes']
    for ep in metadata:assert digest(ROOT/ep['path'])==ep['sha256']
    frozen=json.loads((ROOT/'reports/track1_fmr/stage1_holdout_v4/protocol.json').read_text())
    assert protocol['code_hashes']['src/track1_fmr/dorsal_flux.py']==frozen['source_hashes']['src/track1_fmr/dorsal_flux.py']
    totals={m:dict(visible=0,observed=0,correct=0,returns=0,recovered=0,squared_error=0.) for m in methods};latencies=[];frames_total=0;queries_total=0;hashes={}
    rows={(r['video'],r['method']):r for r in result['rows']}
    for name in protocol['case_names']:
        case=cases[name];target=np.asarray(case['points'],float);hidden=np.asarray(case['occluded'],bool);keep=(~hidden).any(1);target=target[keep];hidden=hidden[keep];first=np.argmax(~hidden,1)
        count,frames=hidden.shape;frames_total+=frames;queries_total+=count
        eligible=(np.arange(frames)[None,:]>first[:,None])&~hidden
        marker=json.loads((out/(name+'.json')).read_text());trace=json.loads((out/(name+'.trace.json')).read_text());path=out/(name+'.points.npz')
        assert digest(path)==marker['points_sha256'];assert digest(out/(name+'.trace.json'))==marker['trace_sha256'];hashes[path.name]=digest(path)
        table={(r['frame'],r['id']):r for r in trace['entities']}
        particle_table={(r['frame'],r['query_id']):r for r in trace.get('particles',[])}
        for r in trace['entities']:
            if r['source']=='fb_lk':
                assert r['inliers']>=protocol.get('min_inliers',4) and r['fb_max']<protocol['fb_max_px']
                if protocol.get('revision')=='v3':assert r['ncc_min']>=.65
        for edge in trace['lineage']:
            assert edge['parent']!=edge['child'] and edge['time']>=0 and edge['kind'] in ('split_hypothesis','merge_hypothesis')
        latencies.extend(marker['timings'])
        with np.load(path,allow_pickle=False) as saved:
            for m in methods:
                p=saved[m+'_position'];seen=saved[m+'_observed'];carrier=saved[m+'_carrier'];source=saved[m+'_evidence'];active=saved[m+'_active_ids'];ids=saved[m+'_fixed_ids']
                assert p.shape==(count,frames,2) and seen.shape==(count,frames) and np.isfinite(p).all()
                assert not (seen&~carrier).any() and np.all(p[~seen]==-1e4) and np.array_equal(seen,source>0)
                assert not (carrier&(ids<0)[:,None]).any()
                for k,t in zip(*np.nonzero(seen)):
                    r=table[(int(t),int(active[k,t]))]
                    if m.startswith('hybrid'):
                        qid=int(k)+(0 if m=='hybrid_strict' else count);pr=particle_table[(int(t),qid)]
                        assert source[k,t]==3 and pr['observed'] and pr['fb_valid'] and pr['fb_error']<1.5 and pr['previous_seed_valid']
                        assert r['source'] in ('detector','fb_lk') and pr['support_distance']<=6.25
                        d=np.asarray(pr['candidate'])-r['mean'];assert float(d@np.linalg.solve(r['shape'],d))<=6.25
                        assert np.allclose(p[k,t],pr['candidate'])
                    else:assert r['source']==('detector' if source[k,t]==1 else 'fb_lk')
                # Independently replay query transport using only the initial query and
                # saved image-evidence geometry. Later ground-truth points are scoring only.
                for k in range(count):
                    tid=int(ids[k]);point=target[k,first[k]].copy()
                    if m.startswith('hybrid'):
                        if tid<0:continue
                        qid=k+(0 if m=='hybrid_strict' else count);fallback=point.copy();seed=True
                        for t in range(int(first[k])+1,frames):
                            pr=particle_table[(t,qid)];current=int(pr['track_id'])
                            assert np.allclose(pr['previous_position'],point) and pr['previous_seed_valid']==seed
                            if current!=tid:
                                assert m=='hybrid_lineage' and any(edge['parent']==tid and edge['child']==current and round(edge['time']*30)==t for edge in trace['lineage'])
                            mover=table.get((t,tid),table.get((t,current)))
                            if mover is not None:
                                oa=np.asarray(mover['old_anchors'],np.float32);na=np.asarray(mover['new_anchors'],np.float32)
                                if len(oa)>=3:
                                    w=1/np.maximum(np.linalg.norm(oa-fallback,axis=1),1e-4)**2;w/=w.sum();fallback=fallback+(w[:,None]*(na-oa)).sum(0)
                                elif mover['source']=='detector' and mover.get('previous_box') is not None:
                                    before=np.asarray(mover['previous_box']);after=np.asarray(mover['box']);extent=before[2:]-before[:2]
                                    if np.all(extent>1e-6):fallback=after[:2]+(fallback-before[:2])/extent*(after[2:]-after[:2])
                            e=table.get((t,current));candidate=np.asarray(pr['candidate'],np.float32);inside=False
                            if e is not None:
                                d=candidate-e['mean'];inside=float(d@np.linalg.solve(e['shape'],d))<=6.25
                            accepted=bool(seed and pr['fb_valid'] and e is not None and e['source'] in ('detector','fb_lk') and inside)
                            assert accepted==pr['observed']==bool(seen[k,t])
                            if accepted:point=candidate.copy();fallback=point.copy();seed=True
                            else:
                                point=fallback.copy();seed=False
                                if e is not None and e['source']=='detector':
                                    d=point-e['mean'];seed=bool(float(d@np.linalg.solve(e['shape'],d))<=6.25 and np.all(point>=np.asarray(e['box'])[:2]) and np.all(point<=np.asarray(e['box'])[2:]))
                            assert np.allclose(point,pr['position']) and np.allclose(fallback,pr['fallback']) and seed==pr['seed_valid'],(name,m,k,t,point,pr['position'],fallback,pr['fallback'],seed,pr['seed_valid'])
                            if seen[k,t]:assert np.allclose(point,p[k,t],rtol=1e-5,atol=1e-6)
                            if m=='hybrid_strict':assert current==int(ids[k])
                            tid=current
                        continue
                    for t in range(int(first[k])+1,frames):
                        current=int(active[k,t])
                        if current<0:continue
                        if current!=tid:
                            assert m=='keyframe_lineage'
                            assert any(edge['parent']==tid and edge['child']==current and round(edge['time']*30)==t for edge in trace['lineage'])
                        mover=table.get((t,tid),table[(t,current)])
                        old=np.asarray(mover['old_anchors']);new=np.asarray(mover['new_anchors'])
                        if len(old)>=protocol.get('min_inliers',4):
                            weights=1/np.maximum(np.linalg.norm(old-point,axis=1),1e-4)**2;weights/=weights.sum();point=point+(weights[:,None]*(new-old)).sum(0)
                        elif mover['source']=='detector' and mover.get('previous_box') is not None:
                            before=np.asarray(mover['previous_box']);after=np.asarray(mover['box']);extent=before[2:]-before[:2]
                            if np.all(extent>1e-6):point=after[:2]+(point-before[:2])/extent*(after[2:]-after[:2])
                        if seen[k,t]:assert np.allclose(point,p[k,t],rtol=1e-5,atol=1e-6),(name,m,k,t)
                        if m=='keyframe_strict':assert current==int(ids[k])
                        tid=current
                errors=np.linalg.norm((p-target)*256,axis=-1);matched=eligible&seen;good=matched&(errors<=8)
                total=totals[m];total['visible']+=int(eligible.sum());total['observed']+=int(matched.sum());total['correct']+=int(good.sum());total['squared_error']+=float(np.square(errors[matched]).sum())
                returns=0;recovered=0
                for k in range(count):
                    for t in range(int(first[k])+1,frames):
                        if hidden[k,t-1] and not hidden[k,t]:returns+=1;recovered+=int(good[k,t:min(frames,t+5)].any())
                total['returns']+=returns;total['recovered']+=recovered
                row=rows[(name,m)];assert row['visible_point_times']==int(eligible.sum()) and row['observed_visible_point_times']==int(matched.sum()) and row['observed_correct_8px_times']==int(good.sum())
                assert row['reference_occlusion_returns']==returns and row['recovered_returns_5frame']==recovered
    for meta in result['native']:
        path=out/(meta['episode_id']+'.observations.npz');assert digest(path)==meta['sha256'];hashes[path.name]=digest(path)
        with np.load(path) as z:
            assert z['values'].shape==(meta['observations'],28)
            assert np.isfinite(z['values']).all() and np.isin(z['evidence'],['detector','fb_lk']).all()
            if protocol.get('revision') in ('v3','v4'):
                lk=z['evidence']=='fb_lk';assert (z['fb_inliers'][lk]>=protocol['min_inliers']).all() and (z['fb_max'][lk]<protocol['fb_max_px']).all()
                if protocol.get('revision')=='v3':assert (z['ncc_min'][lk]>=.65).all()
                assert (z['motion_dt'][lk]>0).all() and np.isfinite(z['motion_delta']).all()
                from track1_fmr.belief_filter import Belief,advance
                previous={}
                for index,row in enumerate(z['values']):
                    tid=int(row[1]);cov=row[12:28].reshape(4,4)
                    assert np.linalg.eigvalsh(cov).min()>-1e-10
                    if z['evidence'][index]=='fb_lk':
                        old=previous[tid];prior=Belief(old[0],old[8:12],old[12:28].reshape(4,4),np.eye(2),'OBSERVED');prediction=advance(prior,row[0],protocol['tracker_q'])
                        expected=prediction.covariance.copy();expected[:2,:2]+=row[4:8].reshape(2,2)+np.eye(2)*protocol['lk_diffusion_density']*z['motion_dt'][index]
                        assert np.allclose(cov,expected,rtol=1e-6,atol=1e-10)
                        assert np.allclose(row[8:10],old[8:10]+z['motion_delta'][index],rtol=1e-6,atol=1e-8)
                    previous[tid]=row.copy()
    summaries=[]
    for m,total in totals.items():
        if not total['visible']:continue
        summaries.append(dict(method=m,**total,observed_visible_coverage=total['observed']/total['visible'],observed_point_agreement_8px=total['correct']/total['visible'],conditional_rmse_256px=float(np.sqrt(total['squared_error']/total['observed'])) if total['observed'] else None))
    mean_ms=1000*float(np.mean(latencies)) if latencies else None
    native_times=[t for e in result['native'] for t in e['timings']]
    native_ms=1000*float(np.mean(native_times)) if native_times else None
    checks=dict(timing=mean_ms is not None and mean_ms<=13.5,native_timing=native_ms is not None and native_ms<=13.5,rss=result['sampled_process_tree_peak_rss_gib']<=18,
        complete_point_denominator=frames_total==1999 and queries_total==650 and all(t['visible']==28174 for t in totals.values()))
    for r in summaries:
        checks[r['method']+'_coverage']=r['observed_visible_coverage']>=.62;checks[r['method']+'_agreement']=r['observed_point_agreement_8px']>=.42;checks[r['method']+'_reacquisition']=r['recovered']>=50 and r['returns']==471
    probe_path=out/'visor_detector_probes.json';probes=None
    if probe_path.exists():
        probes=json.loads(probe_path.read_text());assert digest(out/'visor_detector_probes.npz')==probes['arrays_sha256']
        assert probes['adapter_sha256']==protocol['code_hashes']['src/track1_fmr/keyframe_identity.py']
        assert probes['reference_selection_sha256']==protocol['reference_selection_sha256']
        spec=importlib.util.spec_from_file_location('visor_reference',ROOT/'scripts/exp_track1_fmr/02a_eval_visor_sparse.py');reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
        selection=json.loads((ROOT/'reports/track1_fmr/stage2_visor_v3/protocol.json').read_text())
        refs={(ep['video_id'],r['timestamp_us']):r for ep in selection['episodes'] for r in ep['references']}
        assert {(r['video_id'],r['timestamp_us']) for r in probes['rows']}==set(refs)
        with np.load(out/'visor_detector_probes.npz') as saved:
            for r in probes['rows']:
                truth=reference.raster(refs[(r['video_id'],r['timestamp_us'])]);masks=saved[r['array_key']]
                best=max([float((m&truth).sum()/max(1,(m|truth).sum())) for m in masks] or [0.])
                assert np.isclose(best,r['best_mask_iou'])
        hashes[probe_path.name]=digest(probe_path);hashes['visor_detector_probes.npz']=probes['arrays_sha256']
    for m in methods:
        v=[r for r in result['visor'] if r['method']==m]
        checks[m+'_visor_scheduled_detection']=len(v)==8 and sum(r['detector_best_mask_iou']>=.5 for r in v)/8>=.75
        checks[m+'_visor_detection']=probes is not None and len(probes['rows'])==8 and sum(r['best_mask_iou']>=.5 for r in probes['rows'])/8>=.75
        checks[m+'_visor_later_support']=all(any(r['video_id'].startswith(pid) and not r['initialization_frame'] and r['observed_fixed_id_mask_recall']>0 for r in v) for pid in ('P06','P08','P21'))
    verification=dict(passed=True,videos=len(protocol['case_names']),frames=frames_total,queries=queries_total,visible_point_times=next(iter(totals.values()))['visible'],source_checkpoint_reference_and_output_hashes_verified=True,fb_lk_inlier_evidence_verified=True,fb_lk_evidence_scope='DAVIS/native FB evidence and covariance equations; hybrid query acceptance and prediction/reseed provenance replayed' if protocol.get('revision')=='v4' else 'DAVIS and native inlier, FB and NCC checks; native LK covariance equations replayed' if protocol.get('revision')=='v3' else 'Saved DAVIS per-anchor evidence; native source tags and hashes checked',query_transport_replayed_without_later_reference_input=True,verifier_source_sha256=digest(Path(__file__)),coasting_never_credited_as_observation=True,denominators_recomputed_from_independent_reference=True,output_hashes=hashes,result_sha256=digest(out/'result.json'),scientific_acceptance=False)
    atomic_json(out/'verification.json',verification)
    atomic_json(out/'assessment.json',dict(point_metrics=summaries,frame_weighted_ingestion_ms=mean_ms,native_frame_weighted_ingestion_ms=native_ms,peak_rss_gib=result['sampled_process_tree_peak_rss_gib'],qualification_checks=checks,precalibration_gate_passed=all(value for key,value in checks.items() if not key.endswith('_visor_scheduled_detection')),visor=result['visor'],detector_only_visor_probes=probes,object_idf1=None,contact_f1=None,human_adjudicated=False,scientific_acceptance=False))
    print(json.dumps(dict(frames=frames_total,queries=queries_total,mean_ms=mean_ms,native_ms=native_ms,qualification_checks=checks)))
