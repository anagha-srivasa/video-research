"""Locked detector/association ablations on independent TAP-Vid points and VISOR masks."""
import argparse
import importlib.util
import json
import socket
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import psutil
import scipy.optimize  # Import solver before component timing.
from pipeline.common import ROOT,atomic_json,digest
from track1_neural_observation.benchmark import ArrayUnpickler,official_metrics,lk_tracks
from track1_fmr.observation_identity import DetectorObserver,ObservationIdentityCarrier
from track1_fmr.pipeline import Pipeline
from track1_fmr.dorsal_flux import PromptFreeHeartbeat

MODES=[f'{mode}_{representation}' for mode in ('detector','motion','appearance','flow') for representation in ('bbox','mask')]


def isolated():
    try:connection=socket.create_connection(('127.0.0.1',1234),timeout=.2)
    except OSError:return
    connection.close();raise RuntimeError('Stop the LM Studio server before this run')


def contains(box,point,mask=None):
    if np.any(point<box[:2]) or np.any(point>box[2:]):return False
    if mask is None:return True
    x,y=np.clip((point*np.array(mask.shape[::-1])).astype(int),0,np.array(mask.shape[::-1])-1)
    return bool(mask[y,x])


def bind(carriers,point):
    candidates=[]
    for tid,c in carriers.items():
        if not c.observed:continue
        mask=None if c.observation is None else c.observation.mask
        if contains(c.box,point,mask):candidates.append((float(np.maximum(0,c.box[2:]-c.box[:2]).prod()),tid))
    return min(candidates)[1] if candidates else -1


def legacy_snapshot(model,rgb,t):
    model.step(rgb,t);result={}
    for tid,state in model.tracker.tracks.items():
        extent=2*np.sqrt(np.diag(state.shape_scatter))
        box=np.clip(np.r_[state.mean[:2]-extent,state.mean[:2]+extent],0,1)
        result[tid]=SimpleNamespace(belief=state,box=box,observed=state.status=='OBSERVED',observation=None)
    return result


def point_row(name,positions,observed,carrier,target,hidden,first,fixed):
    post=np.arange(target.shape[1])[None,:]>first[:,None];visible=post & ~hidden
    errors=np.linalg.norm((positions-target)*256,axis=-1)
    good=visible & observed & (errors<=8)
    visible_n=int(visible.sum());total=int(post.sum());observed_n=int((visible & observed).sum())
    returns=0;recovered=0
    for k in range(len(first)):
        seen=False
        for t in range(int(first[k]),target.shape[1]):
            if not hidden[k,t]:
                if seen and t>first[k] and hidden[k,t-1]:
                    returns+=1;recovered+=int(good[k,t:min(t+5,target.shape[1])].any())
                seen=True
    return dict(method=name,visible_point_times=visible_n,total_point_times=total,
        initialized_queries=int((fixed>=0).sum()),queries=len(first),
        observed_visible_point_times=observed_n,observed_correct_8px_times=int(good.sum()),
        observed_visible_coverage=observed_n/visible_n if visible_n else None,
        observed_point_agreement_8px=int(good.sum())/visible_n if visible_n else None,
        carrier_point_times=int((carrier & post).sum()),carrier_coverage=int((carrier & post).sum())/total if total else None,
        observed_on_hidden=int((observed & hidden & post).sum()),
        conditional_observed_squared_error=float(np.square(errors[visible & observed]).sum()),
        conditional_observed_rmse_256px=float(np.sqrt(np.mean(np.square(errors[visible & observed])))) if observed_n else None,
        reference_occlusion_returns=returns,recovered_returns_5frame=recovered,
        reacquisition_point_agreement=recovered/returns if returns else None,
        verified_object_identity_coverage=None,object_idf1=None,object_id_switches=None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',default='observation_identity_v1');parser.add_argument('--videos',type=int,default=30)
    parser.add_argument('--skip-visor',action='store_true')
    parser.add_argument('--visor-only',action='store_true');args=parser.parse_args()
    if not args.tag.replace('_','').isalnum() or not 1<=args.videos<=30:raise ValueError('Invalid run arguments')
    isolated();out=ROOT/'reports/track1_fmr'/args.tag;out.mkdir(exist_ok=False)
    cv2.setNumThreads(4)
    import torch
    torch.set_num_threads(4)
    root=psutil.Process();stop=threading.Event();resource=dict(peak=0,failure=None)
    def monitor():
        while not stop.wait(.05):
            try:
                processes=[root]+root.children(recursive=True);rss=0
                for process in processes:
                    try:rss+=process.memory_info().rss
                    except psutil.NoSuchProcess:pass
                resource['peak']=max(resource['peak'],rss)
                if rss>18*2**30 or psutil.virtual_memory().available<6*2**30:resource['failure']='Memory guard exceeded'
            except psutil.Error as exc:resource['failure']=str(exc)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    def guard():
        if resource['failure']:raise MemoryError(resource['failure'])
    started=time.perf_counter();rows=[];visor=[]
    try:
        dataset=ROOT/'work/track1_observation_revision/tapvid_davis/tapvid_davis.pkl'
        receipt=json.loads((dataset.parent/'receipt.json').read_text())
        dataset_sha=digest(dataset)
        if dataset_sha!=receipt['uncompressed_sha256']:raise ValueError('Independent dataset checksum changed')
        with dataset.open('rb') as stream:cases=ArrayUnpickler(stream).load()
        names=sorted(cases)[:args.videos]
        selection_path=ROOT/'reports/track1_fmr/stage2_visor_v3/protocol.json'
        selection=json.loads(selection_path.read_text())
        for rel,sha in selection['input_hashes'].items():
            if digest(ROOT/rel)!=sha:raise ValueError('VISOR input changed: '+rel)
        supporting_sources=[ROOT/'src/track1_neural_observation/benchmark.py',ROOT/'src/track1_belief/data.py',
            ROOT/'src/track1_encoder_revision/source_stream.py',ROOT/'src/track1_reference/frame_checks.py',
            ROOT/'work/track1_observation_revision/tapnet_source/tapnet/tapvid/evaluation_datasets.py']
        protocol=dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            dataset_sha256=dataset_sha,reference_selection_sha256=digest(selection_path),reference_input_hashes=selection['input_hashes'],case_names=names,methods=MODES+['fmr_v5','LK_query_seeded'],
            code_hashes={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),*sorted((ROOT/'src/track1_fmr').glob('*.py')),*supporting_sources]},
            checkpoint_hashes={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'work/yoloe-26s-seg-pf.pt',ROOT/'work/yoloe-26s-seg.pt']},
            identity_labels='Independent TAP-Vid point trajectories; no complete object-instance identity labels',
            init='No queries enter detector or region pipelines. At query time bind smallest observed support containing point, then fixed ID and initial point offset.',
            denominator='All visible point-times strictly after each first-visible query, including unmatched queries; carrier denominator includes hidden point-times.',
            metric_resolution=256,assumed_davis_fps=30,detector_confidence=.25,detector_size=640,max_detections=40,tracker_measurement_floor=.02,tracker_acceleration_noise=.01,
            timing='One shared detector pass charged to each ablation, plus measured per-method processing; component estimates, not isolated deployed latency.',
            include_visor=not args.skip_visor,historically_consumed_benchmark=True,query_seeded_lk_has_different_initialization=True,
            physical_identity_verified=False,contact_f1=None,fine_boundary_lens_invocations=0,coarse_detector_segmentation_every_frame=True,
            representation_ablation='Moment geometry changes; detector/foreground color evidence shared. Not a bbox-only detector cost comparison',
            deferred=['Directional lobe/contact ablations gated on observation and identity validation',
                'Frozen spline has no unprivileged predictions on this point dataset; no fabricated same-input baseline'],
            max_sampled_process_tree_rss_gib=18)
        atomic_json(out/'protocol.json',protocol)
        for rel in protocol['code_hashes']:
            dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
        observer=DetectorObserver();tick=time.perf_counter();observer(np.zeros((640,640,3),np.uint8))
        heartbeat=PromptFreeHeartbeat(device='cpu');heartbeat(np.zeros((640,640,3),np.uint8))
        atomic_json(out/'warmup.json',dict(seconds=time.perf_counter()-tick))
        metric=official_metrics()
        for case_index,name in enumerate([] if args.visor_only else names):
            if '/' in name:raise ValueError('Invalid benchmark case name')
            guard();isolated();case=cases[name];video=case['video']
            target=np.asarray(case['points'],float);hidden=np.asarray(case['occluded'],bool)
            keep=(~hidden).any(1);target=target[keep];hidden=hidden[keep];first=np.argmax(~hidden,axis=1)
            count,frames=hidden.shape;systems={key:ObservationIdentityCarrier(*key.split('_')) for key in MODES}
            legacy=Pipeline(dorsal_grid=160,heartbeat=heartbeat)
            tracked=[key for key in MODES if not key.startswith('detector')]+['fmr_v5']
            positions={key:np.full((count,frames,2),-1e4,np.float32) for key in tracked}
            observed={key:np.zeros((count,frames),bool) for key in tracked};carrier={key:np.zeros((count,frames),bool) for key in tracked}
            fixed={key:np.full(count,-1,int) for key in tracked};offset={key:np.zeros((count,2)) for key in tracked}
            detection_support={rep:0 for rep in ('bbox','mask')};timings={key:[] for key in MODES+['fmr_v5']};detection_seconds=[]
            counts={key:dict(redetected=0,rejected=0) for key in MODES};nvisible=0;mask_available=0;detections_total=0
            for frame,rgb in enumerate(video):
                guard()
                if frame%25==0:isolated()
                tick=time.perf_counter();detections=observer(rgb);detection_time=time.perf_counter()-tick;detection_seconds.append(detection_time)
                detections_total+=len(detections);mask_available+=sum(pair[1].mask is not None for pair in detections)
                eligible=(frame>first)&~hidden[:,frame];nvisible+=int(eligible.sum())
                for rep,index in [('bbox',0),('mask',1)]:
                    for k in np.flatnonzero(eligible):
                        detection_support[rep]+=int(any(contains(pair[index].box,target[k,frame],pair[index].mask) for pair in detections))
                snapshots={}
                for key,system in systems.items():
                    tick=time.perf_counter();snapshots[key]=system.step(rgb,frame/30.,detections)
                    timings[key].append(detection_time+time.perf_counter()-tick)
                    counts[key]['redetected']+=system.last_telemetry['redetected'];counts[key]['rejected']+=system.last_telemetry['rejected']
                tick=time.perf_counter();snapshots['fmr_v5']=legacy_snapshot(legacy,rgb,frame/30.);timings['fmr_v5'].append(time.perf_counter()-tick)
                for key in tracked:
                    states=snapshots[key]
                    for k in np.flatnonzero(first==frame):
                        tid=bind(states,target[k,frame]);fixed[key][k]=tid
                        if tid>=0:offset[key][k]=target[k,frame]-states[tid].belief.mean[:2]
                    for k in np.flatnonzero((first<=frame)&(fixed[key]>=0)):
                        state=states.get(int(fixed[key][k]))
                        if state is None:continue
                        carrier[key][k,frame]=True
                        if state.observed:
                            observed[key][k,frame]=True;positions[key][k,frame]=state.belief.mean[:2]+offset[key][k]
                if frame%25==0:print(f'{name} {frame+1}/{frames} frames; elapsed={time.perf_counter()-started:.1f}s',flush=True)
            queries=np.array([[t,target[k,t,1]*256,target[k,t,0]*256] for k,t in enumerate(first)],np.float32)
            lk,lk_seen=lk_tracks(video,np.array([[t,target[k,t,1]*video.shape[1],target[k,t,0]*video.shape[2]] for k,t in enumerate(first)],np.float32))
            positions['LK_query_seeded']=(lk.transpose(1,0,2)/np.array(video.shape[1:3][::-1])).astype(np.float32)
            observed['LK_query_seeded']=lk_seen.T;carrier['LK_query_seeded']=lk_seen.T;fixed['LK_query_seeded']=np.arange(count)
            positions['LK_query_seeded'][~observed['LK_query_seeded']]=-1e4
            for key in MODES:
                if key.startswith('detector'):
                    rep=key.split('_')[1]
                    rows.append(dict(video=name,method=key,visible_point_times=nvisible,detector_supported_times=detection_support[rep],
                        detector_point_support_recall=detection_support[rep]/nvisible if nvisible else None,
                        mean_component_ms=1000*float(np.mean(timings[key])),tracking_metrics=None))
            case_rows=[]
            for key in positions:
                row=point_row(key,positions[key],observed[key],carrier[key],target,hidden,first,fixed[key]);row['video']=name
                official=metric(queries[None],hidden[None],(target*256)[None],(~observed[key])[None],(positions[key]*256)[None],query_mode='first')
                row['official_tapvid_observed_only']={k:float(v[0]) for k,v in official.items()}
                row['mean_component_ms']=None if key=='LK_query_seeded' else 1000*float(np.mean(timings[key]))
                if key in counts:row.update(counts[key])
                rows.append(row);case_rows.append(row)
            common=(np.arange(frames)[None,:]>first[:,None]) & ~hidden
            for key in tracked:common &= observed[key]
            common_result=dict(point_times=int(common.sum()),visible_point_times=nvisible,
                rmse_256px={key:float(np.sqrt(np.mean(np.sum(((positions[key]-target)*256)**2,axis=-1)[common]))) if common.any() else None for key in tracked})
            arrays={}
            for key in positions:arrays.update({key+'_position':positions[key],key+'_observed':observed[key],key+'_carrier':carrier[key],key+'_fixed_ids':fixed[key]})
            np.savez_compressed(out/(name+'.points.npz'),**arrays)
            atomic_json(out/(name+'.json'),dict(video=name,frames=frames,queries=count,rows=case_rows,common_support=common_result,
                detector_seconds=detection_seconds,raw_detections=detections_total,mask_observations=mask_available,
                points_sha256=digest(out/(name+'.points.npz'))))
            atomic_json(out/'progress.json',dict(completed=case_index+1,total=len(names),frames_completed=sum(len(json.loads(p.read_text())['detector_seconds']) for p in out.glob('*.json') if p.stem in names),elapsed_seconds=time.perf_counter()-started))
            print(f'completed {case_index+1}/{len(names)} {name}',flush=True)
        # Reuse the predeclared independent sparse-mask clips; references never enter inference.
        source=ROOT/'reports/track1_fmr/stage2_visor_v3/protocol.json';selection=json.loads(source.read_text())
        spec=importlib.util.spec_from_file_location('visor_reference',ROOT/'scripts/exp_track1_fmr/02a_eval_visor_sparse.py')
        reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
        from track1_belief import data
        data.W=ROOT/'work/track1_fmr'
        atomic_json(out/'visor_protocol.json',dict(selection_sha256=digest(source),input_hashes=selection['input_hashes'],scope='Previously consumed P02/P08 and reserved P06/P21 sequences, not fresh holdouts'))
        for ep in ([] if args.skip_visor else selection['episodes']):
            systems={key:ObservationIdentityCarrier(*key.split('_')) for key in MODES};bound={key:-1 for key in MODES};bytime={r['timestamp_us']:r for r in ep['references']}
            for index,rgb,_ in data.native_frames(ep['video_id'],ep['frames'],ep['video_id']+'.observation_identity'):
                guard();tick=time.perf_counter();detections=observer(rgb)
                snapshots={key:system.step(rgb,ep['frames'][index]['timestamp_us']/1e6,detections) for key,system in systems.items()}
                ts=ep['frames'][index]['timestamp_us']
                if ts not in bytime:continue
                mask=reference.raster(bytime[ts]);yy,xx=np.nonzero(mask);centre=np.array([xx.mean()/160,yy.mean()/160])
                truth_scatter=np.cov(np.c_[xx/160,yy/160],rowvar=False,bias=True)
                for key,states in snapshots.items():
                    if index==0 and not key.startswith('detector'):bound[key]=bind(states,centre)
                    predicted=states.get(bound[key]);observations=[pair[0 if key.endswith('bbox') else 1] for pair in detections]
                    def support(observation):
                        if observation.mask is not None:return cv2.resize(observation.mask.astype(np.uint8),(160,160),interpolation=cv2.INTER_NEAREST).astype(bool)
                        m=np.zeros((160,160),bool);lo=np.floor(observation.box[:2]*160).astype(int);hi=np.ceil(observation.box[2:]*160).astype(int);m[lo[1]:hi[1],lo[0]:hi[0]]=True;return m
                    direct=max([int((support(o)&mask).sum())/int((support(o)|mask).sum()) for o in observations] or [0.])
                    observation=predicted.observation if predicted is not None and predicted.observed else None
                    recall=0. if observation is None else int((support(observation)&mask).sum())/int(mask.sum())
                    visor.append(dict(video_id=ep['video_id'],timestamp_us=ts,method=key,initialization_frame=index==0,
                        fixed_id=None if bound[key]<0 else bound[key],detector_best_mask_iou=direct,observed_fixed_id_mask_recall=recall,
                        conditional_centroid_error_norm=None if observation is None else float(np.linalg.norm(observation.cloud.centroid-centre)),
                        conditional_scatter_error=None if observation is None else float(np.linalg.norm(observation.cloud.shape_scatter-truth_scatter))))
            atomic_json(out/'visor_progress.json',visor);print('completed VISOR',ep['video_id'],flush=True)
        summaries=[]
        for key in MODES+['fmr_v5','LK_query_seeded']:
            selected=[r for r in rows if r['method']==key]
            if not selected:continue
            denominator=sum(r['visible_point_times'] for r in selected)
            summary=dict(method=key,videos=len(selected),visible_point_times=denominator,
                mean_component_ms=None if key=='LK_query_seeded' else float(np.mean([r['mean_component_ms'] for r in selected])),
                verified_object_identity_coverage=None,object_idf1=None,physical_contact_f1=None)
            if key.startswith('detector'):
                summary['detector_point_support_recall']=sum(r['detector_supported_times'] for r in selected)/denominator
            else:
                for field in ['observed_visible_point_times','observed_correct_8px_times','carrier_point_times','total_point_times','reference_occlusion_returns','recovered_returns_5frame']:
                    summary[field]=sum(r[field] for r in selected)
                summary.update(observed_visible_coverage=summary['observed_visible_point_times']/denominator,
                    observed_point_agreement_8px=summary['observed_correct_8px_times']/denominator,
                    carrier_coverage=summary['carrier_point_times']/summary['total_point_times'],
                    mean_official_average_jaccard=float(np.mean([r['official_tapvid_observed_only']['average_jaccard'] for r in selected])))
            summaries.append(summary)
        atomic_json(out/'result.json',dict(protocol_sha256=digest(out/'protocol.json'),summaries=summaries,rows=rows,visor=visor,
            sampled_process_tree_peak_rss_gib=resource['peak']/2**30,elapsed_seconds=time.perf_counter()-started,
            independent_point_and_mask_references=True,complete_object_identity_reference=False,scientific_acceptance=False,
            blocked_downstream='No complete identity/contact truth or validated observation-identity gate; no lobe/contact/spline superiority claim'))
    except Exception as exc:
        atomic_json(out/'failure.json',dict(error=repr(exc),completed_rows=len(rows),peak_rss_gib=resource['peak']/2**30));raise
    finally:stop.set();thread.join()


if __name__=='__main__':main()
