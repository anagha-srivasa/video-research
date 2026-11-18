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
        # Detector support translated by measured flow; no fine-boundary claim.
        shift=(e.belief.mean[:2]-e.observation.cloud.centroid)*160
        return cv2.warpAffine(e.observation.mask.astype(np.uint8),np.array([[1,0,shift[0]],[0,1,shift[1]]],np.float32),(160,160),flags=cv2.INTER_NEAREST).astype(bool)
    m=np.zeros((160,160),bool);lo=np.clip((e.box[:2]*160).astype(int),0,160);hi=np.clip(np.ceil(e.box[2:]*160).astype(int),0,160);m[lo[1]:hi[1],lo[0]:hi[0]]=True;return m


def execute(args,helpers):
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
        protocol=dict(kind='keyframe_v2',methods=METHODS,case_names=names,dataset_sha256=digest(dataset),keyframe_stride=args.keyframe_stride,
            detector_confidence=.1,detector_size=640,grid=160,anchor_resolution=320,anchors=8,fb_max_px=1.5,min_inliers=4,
            code_hashes={str(p.resolve().relative_to(ROOT)):digest(p) for p in sources},reference_input_hashes=selection['input_hashes'],reference_selection_sha256=digest(selection_path),
            checkpoint_hashes={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'work/yoloe-26s-seg-pf.pt',ROOT/'work/yoloe-26s-seg.pt',ROOT/'work/mobileclip2_b.ts']},
            native_metadata_sha256=digest(metadata_path),native_frame_index_sha256=digest(frame_path),
            observation_policy='detector or >=4 FB-verified image correspondences; predictions never observed',
            point_policy='Initial query only; incremental inverse-distance anchor displacement. Lost anchors never predicted forward. No later reference point input.',
            lineage='Causal provisional split/merge edges; smallest active descendant support containing prior transported point. Strict IDs also scored.',
            q_units='continuous white acceleration spectral density',covariance_eigenvalue_cap=.25,
            timing='Isolated CPU, shared resident detector, every runtime keyframe/flow/association cost included; query evaluation excluded; decoder excluded; warmup separate',
            physical_identity_verified=False,human_adjudicated=False,scientific_acceptance=False)
        atomic_json(out/'protocol.json',protocol)
        for rel in protocol['code_hashes']:
            dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
        detector=KeyframeDetector();detector.warmup();guard()
        atomic_json(out/'warmup.json',dict(seconds=detector.warmup_seconds,text_cache_manifest=str(detector.cache_manifest.relative_to(ROOT)),text_cache_sha256=digest(detector.cache_manifest)))
        rows=[];visor=[];all_timings=[];native=[]
        metric=official_metrics()
        for ci,name in enumerate([] if args.visor_only else names):
            helpers.isolated();case=cases[name];video=case['video'];target=np.asarray(case['points'],float);hidden=np.asarray(case['occluded'],bool)
            keep=(~hidden).any(1);target=target[keep];hidden=hidden[keep];first=np.argmax(~hidden,1);count,frames=hidden.shape
            model=KeyframeObservationIdentityTracker(args.keyframe_stride)
            positions={m:np.full((count,frames,2),-1e4,np.float32) for m in METHODS}
            observed={m:np.zeros((count,frames),bool) for m in METHODS};carriers={m:np.zeros((count,frames),bool) for m in METHODS}
            evidence={m:np.zeros((count,frames),np.uint8) for m in METHODS};active_history={m:np.full((count,frames),-1,int) for m in METHODS};active={m:np.full(count,-1,int) for m in METHODS};fixed={m:np.full(count,-1,int) for m in METHODS}
            points={m:np.zeros((count,2)) for m in METHODS};frame_rows=[];times=[];trace=[]
            for frame,rgb in enumerate(video):
                guard();tick=time.perf_counter();states=model.step(rgb,frame/30,detector);elapsed=time.perf_counter()-tick;times.append(elapsed);all_timings.append(elapsed)
                record=dict(frame=frame,**model.last_telemetry);frame_rows.append(record)
                for e in states.values():
                    trace.append(dict(frame=frame,id=e.track_id,source=e.evidence,inliers=len(e.fb_errors),fb_max=float(e.fb_errors.max()) if len(e.fb_errors) else None,old_anchors=e.displacement_old.tolist(),new_anchors=e.displacement_new.tolist()))
                for m in METHODS:
                    for k in np.flatnonzero(first==frame):
                        tid=helpers.bind(states,target[k,frame]);fixed[m][k]=active[m][k]=tid;points[m][k]=target[k,frame]
                    for k in np.flatnonzero((first<frame)&(active[m]>=0)):
                        tid=int(active[m][k]);old=states.get(tid)
                        if m=='keyframe_lineage':tid=model.follow(tid,points[m][k]);active[m][k]=tid
                        e=states.get(tid)
                        if e is None:continue
                        # Preserve local query offset across detector reseeding, using this frame's verified old/new anchors.
                        mover=old if old is not None else e
                        points[m][k]=model.transport(points[m][k],mover.displacement_old,mover.displacement_new)
                        active_history[m][k,frame]=tid;carriers[m][k,frame]=True
                        if e.observed:
                            observed[m][k,frame]=True;positions[m][k,frame]=points[m][k];evidence[m][k,frame]=1 if e.evidence=='detector' else 2
                if frame%25==0:print(f'{name} {frame+1}/{frames}; elapsed={time.perf_counter()-started:.1f}s',flush=True)
            queries=np.array([[t,target[k,t,1]*256,target[k,t,0]*256] for k,t in enumerate(first)],np.float32)
            case_rows=[];arrays={}
            for m in METHODS:
                row=helpers.point_row(m,positions[m],observed[m],carriers[m],target,hidden,first,fixed[m]);row.update(video=name,mean_component_ms=1000*np.mean(times))
                official=metric(queries[None],hidden[None],(target*256)[None],(~observed[m])[None],(positions[m]*256)[None],query_mode='first')
                row['official_tapvid_observed_only']={k:float(v[0]) for k,v in official.items()};rows.append(row);case_rows.append(row)
                arrays.update({m+'_position':positions[m],m+'_observed':observed[m],m+'_carrier':carriers[m],m+'_fixed_ids':fixed[m],m+'_evidence':evidence[m],m+'_active_ids':active_history[m]})
            np.savez_compressed(out/(name+'.points.npz'),**arrays)
            atomic_json(out/(name+'.trace.json'),dict(frames=frame_rows,entities=trace,lineage=model.lineage))
            atomic_json(out/(name+'.json'),dict(frames=frames,queries=count,rows=case_rows,timings=times,points_sha256=digest(out/(name+'.points.npz')),trace_sha256=digest(out/(name+'.trace.json'))))
            atomic_json(out/'progress.json',dict(completed=ci+1,total=len(names),elapsed_seconds=time.perf_counter()-started));print('completed',ci+1,len(names),name,flush=True)
        spec=importlib.util.spec_from_file_location('visor_reference',ROOT/'scripts/exp_track1_fmr/02a_eval_visor_sparse.py');reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
        from track1_belief import data
        data.W=ROOT/'work/track1_fmr'
        for ep in ([] if args.skip_visor else selection['episodes']):
            model=KeyframeObservationIdentityTracker(args.keyframe_stride);bound={m:-1 for m in METHODS};point={m:None for m in METHODS};bytime={r['timestamp_us']:r for r in ep['references']}
            for index,rgb,_ in data.native_frames(ep['video_id'],ep['frames'],ep['video_id']+'.keyframe'):
                guard();states=model.step(rgb,ep['frames'][index]['timestamp_us']/1e6,detector)
                for m in METHODS:
                    if point[m] is not None:
                        old=states.get(bound[m]);tid=model.follow(bound[m],point[m]) if m.endswith('lineage') else bound[m]
                        if old is not None:point[m]=model.transport(point[m],old.displacement_old,old.displacement_new)
                        bound[m]=tid
                ts=ep['frames'][index]['timestamp_us']
                if ts not in bytime:continue
                truth=reference.raster(bytime[ts]);yy,xx=np.nonzero(truth);centre=np.array([xx.mean()/160,yy.mean()/160])
                for m in METHODS:
                    if index==0:bound[m]=helpers.bind(states,centre);point[m]=centre.copy()
                    e=states.get(bound[m]);pred=support(e) if e is not None and e.observed else np.zeros_like(truth)
                    direct=max([float((support(x)&truth).sum()/max(1,(support(x)|truth).sum())) for x in states.values() if x.evidence=='detector'] or [0.])
                    visor.append(dict(video_id=ep['video_id'],timestamp_us=ts,method=m,initialization_frame=index==0,fixed_id=bound[m],detector_best_mask_iou=direct,observed_fixed_id_mask_recall=float((pred&truth).sum()/truth.sum())))
            atomic_json(out/'visor_progress.json',visor);print('completed VISOR',ep['video_id'],flush=True)
        # New native cohort: inference reads RGB and timestamps only, never source centres or VLM labels.
        if not args.visor_only:
            import pyarrow.parquet as pq
            metadata=json.loads(metadata_path.read_text())['episodes'];frames_all=pq.read_table(frame_path).to_pylist()
            for meta in metadata:
                source=ROOT/meta['path']
                if digest(source)!=meta['sha256']:raise ValueError('Source ledger changed')
                with np.load(source,allow_pickle=False) as z:timestamps=z['timestamp_us'];timestamps=timestamps[timestamps>=meta['manual_seed_us']]
                requested=set(timestamps.tolist());frames=sorted([f for f in frames_all if f['video_id']==meta['video_id'] and f['timestamp_us'] in requested],key=lambda f:f['timestamp_us'])
                if [f['timestamp_us'] for f in frames]!=timestamps.tolist():raise ValueError('Native PTS mismatch')
                model=KeyframeObservationIdentityTracker(args.keyframe_stride);values=[];shapes=[];boxes=[];hits=[];sources=[];timings=[]
                for n,rgb,_ in data.native_frames(meta['video_id'],frames,meta['episode_id']+'.keyframe'):
                    guard()
                    if n%100==0:helpers.isolated();print('native',meta['episode_id'],n,len(frames),flush=True)
                    tick=time.perf_counter();states=model.step(rgb,frames[n]['timestamp_us']/1e6,detector);timings.append(time.perf_counter()-tick)
                    for tid,e in states.items():
                        if not e.observed:continue
                        values.append([e.belief.time,tid,*e.measurement,*e.measurement_covariance.ravel(),*e.belief.mean,*e.belief.covariance.ravel()]);shapes.append(e.belief.shape_scatter);boxes.append(e.box.copy());hits.append(e.hits);sources.append(e.evidence)
                path=out/(meta['episode_id']+'.observations.npz')
                np.savez_compressed(path,values=np.asarray(values).reshape(-1,28),support_shapes=np.asarray(shapes).reshape(-1,2,2),support_boxes=np.asarray(boxes).reshape(-1,4),observed_counts=np.asarray(hits),evidence=np.asarray(sources))
                native.append(dict(episode_id=meta['episode_id'],frames=len(frames),observations=len(values),sha256=digest(path),mean_ingestion_ms=1000*np.mean(timings),timings=timings));atomic_json(out/'native_progress.json',native)
        summaries=[]
        for m in METHODS:
            selected=[r for r in rows if r['method']==m];den=sum(r['visible_point_times'] for r in selected)
            if not den:continue
            summaries.append(dict(method=m,visible_point_times=den,observed_visible_coverage=sum(r['observed_visible_point_times'] for r in selected)/den,observed_point_agreement_8px=sum(r['observed_correct_8px_times'] for r in selected)/den,recovered_returns=sum(r['recovered_returns_5frame'] for r in selected),mean_ingestion_ms=1000*np.mean(all_timings)))
        atomic_json(out/'result.json',dict(protocol_sha256=digest(out/'protocol.json'),summaries=summaries,rows=rows,visor=visor,native=native,sampled_process_tree_peak_rss_gib=resource['peak']/2**30,elapsed_seconds=time.perf_counter()-started,scientific_acceptance=False))
    except Exception as exc:
        atomic_json(out/'failure.json',dict(error=repr(exc),peak_rss_gib=resource['peak']/2**30));raise
    finally:stop.set();thread.join()


def verify(out):
    protocol=json.loads((out/'protocol.json').read_text());result=json.loads((out/'result.json').read_text())
    assert digest(out/'protocol.json')==result['protocol_sha256']
    for rel,sha in protocol['code_hashes'].items():assert digest(out/'source'/rel)==sha,rel
    for group in ('checkpoint_hashes','reference_input_hashes'):
        for rel,sha in protocol[group].items():assert digest(ROOT/rel)==sha,rel
    dataset=ROOT/'work/track1_observation_revision/tapvid_davis/tapvid_davis.pkl';assert digest(dataset)==protocol['dataset_sha256']
    with dataset.open('rb') as f:cases=ArrayUnpickler(f).load()
    totals={m:dict(visible=0,observed=0,correct=0,returns=0,recovered=0,squared_error=0.) for m in METHODS};latencies=[];frames_total=0;queries_total=0;hashes={}
    rows={(r['video'],r['method']):r for r in result['rows']}
    for name in protocol['case_names']:
        case=cases[name];target=np.asarray(case['points'],float);hidden=np.asarray(case['occluded'],bool);keep=(~hidden).any(1);target=target[keep];hidden=hidden[keep];first=np.argmax(~hidden,1)
        count,frames=hidden.shape;frames_total+=frames;queries_total+=count
        eligible=(np.arange(frames)[None,:]>first[:,None])&~hidden
        marker=json.loads((out/(name+'.json')).read_text());trace=json.loads((out/(name+'.trace.json')).read_text());path=out/(name+'.points.npz')
        assert digest(path)==marker['points_sha256'];assert digest(out/(name+'.trace.json'))==marker['trace_sha256'];hashes[path.name]=digest(path)
        table={(r['frame'],r['id']):r for r in trace['entities']}
        for r in trace['entities']:
            if r['source']=='fb_lk':assert r['inliers']>=4 and r['fb_max']<1.5
        for edge in trace['lineage']:
            assert edge['parent']!=edge['child'] and edge['time']>=0 and edge['kind'] in ('split_hypothesis','merge_hypothesis')
        latencies.extend(marker['timings'])
        with np.load(path,allow_pickle=False) as saved:
            for m in METHODS:
                p=saved[m+'_position'];seen=saved[m+'_observed'];carrier=saved[m+'_carrier'];source=saved[m+'_evidence'];active=saved[m+'_active_ids'];ids=saved[m+'_fixed_ids']
                assert p.shape==(count,frames,2) and seen.shape==(count,frames) and np.isfinite(p).all()
                assert not (seen&~carrier).any() and np.all(p[~seen]==-1e4) and np.array_equal(seen,source>0)
                assert not (carrier&(ids<0)[:,None]).any()
                for k,t in zip(*np.nonzero(seen)):
                    r=table[(int(t),int(active[k,t]))];assert r['source']==('detector' if source[k,t]==1 else 'fb_lk')
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
    for m in METHODS:
        v=[r for r in result['visor'] if r['method']==m]
        checks[m+'_visor_detection']=len(v)==8 and sum(r['detector_best_mask_iou']>=.5 for r in v)/8>=.75
        checks[m+'_visor_later_support']=all(any(r['video_id'].startswith(pid) and not r['initialization_frame'] and r['observed_fixed_id_mask_recall']>0 for r in v) for pid in ('P06','P08','P21'))
    verification=dict(passed=True,videos=len(protocol['case_names']),frames=frames_total,queries=queries_total,visible_point_times=next(iter(totals.values()))['visible'],source_checkpoint_reference_and_output_hashes_verified=True,fb_lk_inlier_evidence_verified=True,coasting_never_credited_as_observation=True,denominators_recomputed_from_independent_reference=True,output_hashes=hashes,result_sha256=digest(out/'result.json'),scientific_acceptance=False)
    atomic_json(out/'verification.json',verification)
    atomic_json(out/'assessment.json',dict(point_metrics=summaries,frame_weighted_ingestion_ms=mean_ms,native_frame_weighted_ingestion_ms=native_ms,peak_rss_gib=result['sampled_process_tree_peak_rss_gib'],qualification_checks=checks,precalibration_gate_passed=all(checks.values()),visor=result['visor'],object_idf1=None,contact_f1=None,human_adjudicated=False,scientific_acceptance=False))
    print(json.dumps(dict(frames=frames_total,queries=queries_total,mean_ms=mean_ms,native_ms=native_ms,qualification_checks=checks)))
